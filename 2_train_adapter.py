"""Traj classifier.

As input, we have:
- A SLAM trajectory.

As output, we predict:
- Is this trajectory procedural, or physical?
- What is the floor height?
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
import tyro
import viser.transforms as vtf
from projectaria_tools.core import mps  # type: ignore
from scipy.stats import mode
from torch import nn
from tqdm.auto import tqdm

from ucb_ego.data.egopose_meta import TakeMeta
from ucb_ego.egoexo_classifier import EgoExoSequenceClassifier

device = torch.device("cuda")


# Approximate rigid transforms...
T_device_cpf = (
    vtf.SE3(
        np.array(
            [
                0.66340146,
                0.20968651,
                -0.2447417,
                0.67530109,
                0.00499302,
                -0.05146009,
                -0.04994988,
            ],
        )
    )
    .as_matrix()
    .astype(np.float32)
)
T_cpf_camera = (
    vtf.SE3(
        np.array(
            [
                0.7141628018541093,
                0.0544238619591747,
                0.04388365691196996,
                -0.6964795476920839,
                0.058931798855856754,
                0.006785269380978046,
                0.011996682057799284,
            ],
            dtype=np.float32,
        )
    )
    .as_matrix()
    .astype(np.float32)
)

# These are floor heights extracted from the train set. To start with the
# simplest possible thing we'll just do a nearest-neighbors-style
# classification problem!
DISCRETIZED_FLOOR_HEIGHTS = np.array(
    [
        -1.999,
        -1.934,
        -1.805,
        -1.659,
        -1.546,
        -1.448,
        -1.359,
        -1.280,
        -1.191,
        -1.095,
        -1.034,
        -0.959,
        -0.882,
    ]
)


class TorchDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_path: Path) -> None:
        # We'll train with the full egoex4d dataset, but this excludes the egopose test set.
        self.take_meta = TakeMeta.load(dataset_path)
        candidate_uids = sorted(
            self.take_meta.uids_from_split["train"]
            + self.take_meta.uids_from_split["val"],
            key=lambda x: hashlib.md5(x.encode("utf-8")).hexdigest(),
        )
        self._floor_height_from_uid = {
            uid: float(
                Path(
                    dataset_path / "brent_may29_floor_heights" / f"{uid}.txt"
                ).read_text()
            )
            for uid in candidate_uids
            if Path(dataset_path / "brent_may29_floor_heights" / f"{uid}.txt").exists()
        }
        self._floor_height_class_from_uid: dict[str, int] = {}

        self._uids = []
        for uid in candidate_uids:
            if uid not in self._floor_height_from_uid:
                continue
            floor_height = self._floor_height_from_uid[uid]
            deltas = np.abs(DISCRETIZED_FLOOR_HEIGHTS - floor_height)
            # if np.min(deltas) < 0.03:
            self._floor_height_class_from_uid[uid] = int(np.argmin(deltas))
            self._uids.append(uid)

        print("Dataset length:", len(self._uids))

        self._camera_pose_path = dataset_path

    def __iter__(self):
        return iter(self.generate())

    @functools.cache
    def get_traj(self, take_uid: str) -> np.ndarray:
        closed_loop_path = (
            self._camera_pose_path
            / "takes"
            / self.take_meta.name_from_uid[take_uid]
            / "trajectory"
            / "closed_loop_trajectory.csv"
        )
        downsample_cache_path = closed_loop_path.parent / "_T_world_device_30fps.npy"

        if downsample_cache_path.exists():
            Ts_world_device = np.load(downsample_cache_path)
        else:
            closed_loop_traj = mps.read_closed_loop_trajectory(str(closed_loop_path))  # type: ignore

            # Read timestamps in seconds and transforms as 4x4 matrices.
            timestamps_secs: list[float] = [
                it.tracking_timestamp.total_seconds() for it in closed_loop_traj
            ]

            aria_fps = len(timestamps_secs) / (timestamps_secs[-1] - timestamps_secs[0])
            num_frames = len(timestamps_secs)
            fps_ratio = 30.0 / aria_fps
            new_num_frames = int(fps_ratio * num_frames)
            downsamp_inds = np.linspace(
                0, num_frames - 1, num=new_num_frames, dtype=int
            )

            Ts_world_device = np.array(
                [
                    closed_loop_traj[i]
                    .transform_world_device.to_matrix()
                    .astype(np.float32)
                    for i in downsamp_inds
                ]
            )
            assert Ts_world_device.shape == (new_num_frames, 4, 4)
            np.save(downsample_cache_path, Ts_world_device)

        assert Ts_world_device.dtype == np.float32
        Ts_world_camera = np.einsum(
            "tij,jk,kl->til", Ts_world_device, T_device_cpf, T_cpf_camera
        )
        assert Ts_world_camera.dtype == np.float32
        return Ts_world_camera

    def generate(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        counter = worker_id
        while True:
            take_uid = self._uids[counter % len(self._uids)]
            Ts_world_camera = self.get_traj(take_uid)
            full_length = Ts_world_camera.shape[0]

            sample_length = 64
            start = np.random.randint(0, full_length - 64)

            positions = Ts_world_camera[start : start + sample_length, :3, 3].T
            assert positions.shape == (3, sample_length)

            yield {
                "positions": positions,
                "floor_height": self._floor_height_from_uid[take_uid],
                "floor_height_class": self._floor_height_class_from_uid[take_uid],
                "take_type": {"procedural": 0, "physical": 1}[
                    self.take_meta.type_from_uid[take_uid]
                ],
            }
            counter += num_workers


def main(
    output_dir: Path,
    dataset_path: Path,
    floor_height_dir: Path,
    dummy_test_path: Path,
    discretize_floor_heights: bool = True,
):
    """Train adapter network. The network is small, so we just save outputs and
    don't bother checkpointing.

    As input, we have a SLAM trajectory.

    As output, we predict:
    - Is this trajectory procedural, or physical?
    - What is the floor height?

    Args:
        output_dir: Directory to save predictions.
        dataset_path: Path to the EgoExo4D dataset.
        floor_height_dir: Directory to floor heights that were saved.
        dummy_test_path: Path to dummy test submission file released by challenge.
        discretize_floor_heights: Whether to treat floor heights as regression
            or classification.
    """
    dummy_test = json.loads(dummy_test_path.read_text())

    dataset = TorchDataset(dataset_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        num_workers=8,
    )

    network = EgoExoSequenceClassifier(
        num_floor_height_classes=len(DISCRETIZED_FLOOR_HEIGHTS)
    ).to("cuda")
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)

    take_meta = dataset.take_meta
    cross_entropy = nn.CrossEntropyLoss()

    output_dir.mkdir(exist_ok=True, parents=True)

    for i, batch in enumerate(tqdm(dataloader)):
        take_type_logits, floor_logits, floor_height = network(
            batch["positions"].to("cuda")
        )

        floor_height_class = batch["floor_height_class"].to("cuda")
        take_class = batch["take_type"].to("cuda")

        take_type_loss = cross_entropy(take_type_logits, take_class)
        if discretize_floor_heights:
            floor_height_loss = cross_entropy(floor_logits, floor_height_class)
        else:
            floor_height_loss = (
                torch.mean(
                    (batch["floor_height"].cuda() - floor_height) ** 2
                    * (take_class == 1)
                )
                * 100.0
            )

        loss = take_type_loss + floor_height_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(take_type_loss.item(), floor_height_loss.item())

            if i % 20_000 == 0:
                correct = 0
                incorrect = 0
                floor_height_errors = []

                for test_uid in dummy_test:
                    T_world_ariacam = dataset.get_traj(test_uid)

                    positions = T_world_ariacam[:, :3, 3]
                    assert positions.shape[0] >= 64
                    length = positions.shape[0] // 64 * 64
                    positions_for_network = positions[:length, :].reshape(
                        (length // 64, 64, 3)
                    )
                    positions_for_network = np.moveaxis(positions_for_network, 1, -1)
                    assert positions_for_network.shape == (length // 64, 3, 64)

                    # Forward pass through network.
                    take_type_logits, floor_logits, floor_height = network(
                        torch.tensor(
                            positions_for_network, device="cuda", dtype=torch.float32
                        )
                    )

                    # Get take type as string, floor height as float.
                    take_type = ["procedural", "physical"][
                        int(mode(take_type_logits.argmax(axis=-1).numpy(force=True))[0])
                    ]
                    if discretize_floor_heights:
                        floor_height_class = mode(
                            floor_logits.argmax(axis=-1).numpy(force=True)
                        )[0]
                        floor_height = DISCRETIZED_FLOOR_HEIGHTS[floor_height_class]
                    else:
                        floor_height = floor_height.median().item()
                    (output_dir / f"{test_uid}.txt").write_text(str(floor_height))
                    (output_dir / f"{test_uid}_type.txt").write_text(take_type)

                    gt_floor_height = float(
                        Path(floor_height_dir / f"{test_uid}.txt").read_text()
                    )

                    # Print uid, take type, floor height. Both predicted and ground-truth.
                    print(
                        test_uid,
                        "Take type (pred/gt):",
                        take_type,
                        take_meta.type_from_uid[test_uid],
                        "Floor height (pred/gt):",
                        floor_height,
                        gt_floor_height,
                    )
                    correct += take_type == take_meta.type_from_uid[test_uid]
                    incorrect += take_type != take_meta.type_from_uid[test_uid]

                    if take_type == "physical":
                        floor_height_errors.append(abs(floor_height - gt_floor_height))

                print(
                    f"{correct=} {incorrect=} {np.mean(floor_height_errors)} {floor_height_errors=}"
                )
                (output_dir / "_metrics.txt").write_text(
                    f"{correct=} {incorrect=} {np.mean(floor_height_errors)} {floor_height_errors=}"
                )


if __name__ == "__main__":
    tyro.cli(main)
