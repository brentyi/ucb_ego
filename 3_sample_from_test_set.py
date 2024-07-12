""" """

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
import viser.transforms as vtf

from ucb_ego import fncsmpl, fncsmpl_extensions
from ucb_ego.constraint_optimizers import do_constraint_optimization
from ucb_ego.inference_utils import load_denoiser
from ucb_ego.sampling import run_sampling, run_sampling_with_stitching
from ucb_ego.transforms import SE3, SO3

T_cpf_camera = vtf.SE3(
    np.array(
        [
            0.7141628018541093,
            0.0544238619591747,
            0.04388365691196996,
            -0.6964795476920839,
            0.058931798855856754,
            0.006785269380978046,
            0.011996682057799284,
        ]
    )
)


def main(
    output_dir: Path,
    dataset_path: Path,
    floor_height_dir: Path,
    index_cond: str = "i>=0",
    checkpoint_dir: Path = Path("./experiments/april13/v0/checkpoints_3000000/"),
    body_npz_path: Path = Path("./data/smplh/neutral/model.npz"),
    dummy_test_path: Path = Path("./data/dummy_test.json"),
    glasses_x_angle_offset: float = 0.1,
    num_samples: int = 16,
) -> None:
    """Run a trained diffusion model.

    Args:
        output_dir: Directory to save the output results.
        dataset_path: Path to the EgoExo4D dataset.
        floor_height_dir: Directory to floor heights that were saved.
        index_cond: Condition to filter which takes to run.
        checkpoint_dir: Directory to the diffusion model checkpoint.
        body_npz_path: Path to the SMPL body model.
        dummy_test_path: Path to the dummy test JSON released by the challenge.
        glasses_x_angle_offset: Offset to the glasses X angle.
        num_samples: Number of samples per trajectory.
    """
    device = torch.device("cuda")
    denoiser_network = load_denoiser(checkpoint_dir).to(device)
    body_model = fncsmpl.SmplModel.load(body_npz_path).to(device)

    dummy_test = json.loads(dummy_test_path.read_text())
    camera_pose_dir = dataset_path / "annotations" / "ego_pose" / "test" / "camera_pose"

    output_dir.mkdir(exist_ok=True, parents=True)

    for i, take_uid in enumerate(sorted(dummy_test.keys())):
        assert (floor_height_dir / f"{take_uid}.txt").exists()

    for i, take_uid in enumerate(sorted(dummy_test.keys())):
        if not eval(index_cond, {"i": i}):
            print("Skipping", i)
            continue

        out_path = output_dir / f"{take_uid}.npz"
        if out_path.exists():
            print(f"{out_path} exists, skipping")

        camera_pose_raw: dict[str, Any] = json.loads(
            (camera_pose_dir / f"{take_uid}.json").read_text()
        )
        aria_key = [k for k in camera_pose_raw.keys() if k.startswith("aria")]
        assert len(aria_key) == 1
        (aria_key,) = aria_key
        ego_extrin: dict[str, list[list[float]]] = camera_pose_raw[aria_key][
            "camera_extrinsics"
        ]

        # Interpolate missing frames in trajectory.
        Ts_world_ariacam = dict[int, vtf.SE3]()
        prev_i = None
        for i in sorted(map(int, ego_extrin.keys())):
            delta = i - prev_i if prev_i is not None else 1

            # Interpolate missing poses.
            T_world_ariacam_i = vtf.SE3.from_matrix(
                np.array(ego_extrin[str(i)])
            ).inverse()
            if delta > 1:
                assert prev_i is not None
                T_world_ariacam_prev = Ts_world_ariacam[prev_i]
                T_ariacam_prev_ariacam = (
                    Ts_world_ariacam[prev_i].inverse() @ T_world_ariacam_i
                )
                for j in range(1, delta):
                    Ts_world_ariacam[prev_i + j] = T_world_ariacam_prev @ vtf.SE3.exp(
                        T_ariacam_prev_ariacam.log() * j / delta
                    )

            Ts_world_ariacam[i] = T_world_ariacam_i
            prev_i = i

        # Extrapolate trajectory if needed.
        start_frame = min(map(int, Ts_world_ariacam.keys()))
        start_frame_upper_bound = (
            min(map(int, dummy_test[take_uid]["body"].keys())) - 1
        )  # Off by one because we need to compute velocity...
        while start_frame > start_frame_upper_bound:
            Ts_world_ariacam[start_frame - 1] = (
                Ts_world_ariacam[start_frame]
                @ Ts_world_ariacam[start_frame + 1].inverse()
                @ Ts_world_ariacam[start_frame]
            )
            start_frame -= 1
            print("Extrapolating backward...")
        end_frame = max(map(int, Ts_world_ariacam.keys()))
        end_frame_lower_bound = max(map(int, dummy_test[take_uid]["body"].keys()))
        while end_frame < end_frame_lower_bound:
            Ts_world_ariacam[end_frame + 1] = (
                Ts_world_ariacam[end_frame]
                @ Ts_world_ariacam[end_frame - 1].inverse()
                @ Ts_world_ariacam[end_frame]
            )
            end_frame += 1
            print("Extrapolating forward...")

        # Don't overcompute...
        padding = 32
        Ts_world_ariacam = {
            k: v
            for k, v in Ts_world_ariacam.items()
            if k >= start_frame_upper_bound - padding
            and k <= end_frame_lower_bound + padding
        }
        frame_nums = sorted(Ts_world_ariacam.keys())

        print("Number of frames:", max(frame_nums) - min(frame_nums))

        Ts_world_cpf = (
            vtf.SE3(np.array([Ts_world_ariacam[k].wxyz_xyz for k in frame_nums]))
            @ T_cpf_camera.inverse()
            @ vtf.SE3.from_rotation(vtf.SO3.from_x_radians(glasses_x_angle_offset))
        ).parameters()
        floor_height = float((floor_height_dir / f"{take_uid}.txt").read_text())
        # points_data = np.load(floor_height_dir / f"{take_uid}_filtered_points.npz")[
        #     "points"
        # ]
        Ts_world_cpf[:, 6] -= floor_height
        traj = (
            run_sampling_with_stitching if Ts_world_cpf.shape[0] > 128 else run_sampling
        )(
            denoiser_network,
            body_model=body_model,
            guidance_constraint_optimizer=None,
            Ts_world_cpf=torch.from_numpy(Ts_world_cpf.astype(np.float32)).to(
                device=device
            ),
            num_samples=num_samples,
            device=device,
        )
        traj = do_constraint_optimization(
            Ts_world_cpf=torch.from_numpy(Ts_world_cpf[None, 1:, :].astype(np.float32))
            .to(device=device)
            .repeat(num_samples, 1, 1),
            body_model=body_model,
            traj=traj,
            optimizer="adam",
            optimizer_iters=200,
        )
        Ts_world_cpf[:, 6] += floor_height

        body_quats = SO3.from_matrix(traj.body_rotmats).wxyz
        assert traj.hand_rotmats is not None
        hand_quats = SO3.from_matrix(traj.hand_rotmats).wxyz
        left_hand_quats = hand_quats[..., :15, :]
        right_hand_quats = hand_quats[..., 15:30, :]

        shaped = body_model.with_shape(traj.betas)
        posed = shaped.with_pose_decomposed(
            T_world_root=SE3.identity(
                device=device, dtype=body_quats.dtype
            ).parameters(),
            body_quats=body_quats,
            left_hand_quats=left_hand_quats,
            right_hand_quats=right_hand_quats,
        )

        T_root_head = posed.Ts_world_joint[:, :, 14, :]
        pred_Ts_cpf_root = (
            SE3(T_root_head) @ SE3(fncsmpl_extensions.get_T_head_cpf(shaped))
        ).inverse()
        Ts_world_root = (
            SE3(
                torch.from_numpy(Ts_world_cpf[None, 1:, :].astype(np.float32)).to(
                    device=device
                )
            )
            @ pred_Ts_cpf_root
        )
        np.savez(
            out_path,
            Ts_world_cpf=Ts_world_cpf,
            Ts_world_root=Ts_world_root.parameters().numpy(force=True),
            body_quats=body_quats.numpy(force=True),
            left_hand_quats=left_hand_quats.numpy(force=True),
            right_hand_quats=right_hand_quats.numpy(force=True),
            betas=traj.betas.numpy(force=True),
            frame_nums=np.array(frame_nums[1:]),
        )


if __name__ == "__main__":
    tyro.cli(main)
