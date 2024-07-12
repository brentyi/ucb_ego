import contextlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from tqdm.auto import tqdm

from ucb_ego import fncsmpl


@contextlib.contextmanager
def stopwatch(message: str, verbose: bool = True):
    if verbose:
        print("[STOPWATCH]", message)
    start = time.time()
    yield
    if verbose:
        print("[STOPWATCH]", message, f"finished in {time.time() - start} seconds!")


def main(
    result_npz_dir: Path,
    output_json: Path,
    body_npz_path: Path = Path("./data/smplh/neutral/model.npz"),
    dummy_test_path: Path = Path("./data/dummy_test.json"),
    coco_from_smpl_regressor_path: Path = Path("./data/J_regressor_coco_from_HMR.npy"),
) -> None:
    """Converts the results from our diffusion model into the format expected
    by the EgoExo challenge.

    Args:
        result_npz_dir: Directory containing the npz files with the results.
        output_json: Path to save the output json file.
        body_npz_path: Path to the SMPL-H body model.
        dummy_test_path: Path to the dummy test json file released by the
            challenge.
        coco_from_smpl_regressor_path: Path to the regressor that maps SMPL-H
            vertices to COCO keypoints.
    """
    device = torch.device("cuda")
    body_model = fncsmpl.SmplModel.load(body_npz_path).to(device)

    json_struct: dict[str, Any] = json.loads(dummy_test_path.read_text())
    coco_from_smpl = np.load(coco_from_smpl_regressor_path)

    assert not output_json.exists()
    output_json.parent.mkdir(exist_ok=True, parents=True)

    with torch.inference_mode():
        for take_uid in tqdm(sorted(json_struct.keys())):
            result_npz_path = result_npz_dir / f"{take_uid}.npz"
            if not result_npz_path.exists():
                print(f"Skipping {result_npz_path}, we couldn't find it")
                continue
            print(f"Processing {result_npz_path}!")

            assert coco_from_smpl.shape == (17, 6890)

            # take_uid = result_npz.stem
            # floor_height = float((floor_height_dir / f"{take_uid}.txt").read_text())
            with stopwatch("Reading npz file..."):
                results = dict(**np.load(result_npz_path))

            vert_parts = []

            desired_frame_nums = np.array(
                list(map(int, json_struct[take_uid]["body"].keys()))
            )
            desired_timestep_indices = desired_frame_nums - results["frame_nums"][0]

            start = 0
            num_samples = results["body_quats"].shape[0]
            chunk_size = 512 // num_samples
            while start < len(desired_timestep_indices):
                idx_slice = desired_timestep_indices[start : start + chunk_size]

                posed = body_model.with_shape(
                    torch.from_numpy(results["betas"][:, idx_slice, :]).to(device)
                ).with_pose_decomposed(
                    torch.from_numpy(results["Ts_world_root"][:, idx_slice, :]).to(
                        device
                    ),
                    body_quats=torch.from_numpy(
                        results["body_quats"][:, idx_slice, :]
                    ).to(device),
                    left_hand_quats=torch.from_numpy(
                        results["left_hand_quats"][:, idx_slice, :]
                    ).to(device),
                    right_hand_quats=torch.from_numpy(
                        results["right_hand_quats"][:, idx_slice, :]
                    ).to(device),
                )
                mesh = posed.lbs()

                vert_parts.append(mesh.verts.numpy(force=True))
                start += chunk_size

            all_verts = np.concatenate(vert_parts, axis=1)
            assert (
                all_verts.shape[0] > 0 and len(all_verts.shape) == 4
            )  # (sample #, time, 6890, 3)
            all_verts = np.mean(all_verts, axis=0)
            assert all_verts.shape == (len(desired_frame_nums), 6890, 3)

            coco_keypoints = np.einsum("jv,tvi->tji", coco_from_smpl, all_verts)

            for i, frame_num in enumerate(desired_frame_nums):
                frame_num_str = str(int(frame_num))
                assert frame_num_str in json_struct[take_uid]["body"]
                assert len(json_struct[take_uid]["body"][frame_num_str]) == 17
                assert len(json_struct[take_uid]["body"][frame_num_str][0]) == 3
                json_struct[take_uid]["body"][frame_num_str] = coco_keypoints[
                    i
                ].tolist()
                assert len(json_struct[take_uid]["body"][frame_num_str]) == 17
                assert len(json_struct[take_uid]["body"][frame_num_str][0]) == 3

    output_json.write_text(json.dumps(json_struct))


if __name__ == "__main__":
    tyro.cli(main)
