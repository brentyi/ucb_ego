"""Extract floor planes from EgoExo dataset via RANSAC."""

import contextlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from projectaria_tools.core import mps  # type: ignore
import tyro
from projectaria_tools.core.mps.utils import filter_points_from_confidence
from tqdm.auto import tqdm


@contextlib.contextmanager
def stopwatch(message: str, verbose: bool):
    if verbose:
        print("[STOPWATCH]", message)
    start = time.time()
    yield
    if verbose:
        print("[STOPWATCH]", message, f"finished in {time.time() - start} seconds!")


def load_point_cloud_and_find_ground(
    points_path: Path, output_filtered_points_path: Path, verbose: bool = False
) -> tuple[np.ndarray, float]:
    # Read world points as an Nx3 array.
    if output_filtered_points_path.exists():
        if verbose:
            print("Loading pre-filtered points")
        points_data = np.load(output_filtered_points_path)["points"]
    else:
        assert points_path.exists()
        with stopwatch("Loading points...", verbose=verbose):
            points_data = mps.read_global_point_cloud(str(points_path))  # type: ignore
        with stopwatch("Filtering points...", verbose=verbose):
            points_data = filter_points_from_confidence(
                points_data,
                # threshold_invdep=0.001,
                threshold_invdep=0.0001,
                # threshold_dep=0.05,
                threshold_dep=0.005,
            )

        with stopwatch("Converting points to numpy...", verbose=verbose):
            points_data = np.array([x.position_world for x in points_data])  # type: ignore

        output_filtered_points_path.parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(output_filtered_points_path, points=points_data)

    assert points_data.shape == (points_data.shape[0], 3)

    # RANSAC floor plane.
    # We consider points in the lowest 10% of the point cloud.
    with stopwatch("Finding floor...", verbose=verbose):
        filtered_zs = points_data[:, 2]

        zs = filtered_zs

        # Median-based outlier dropping, this doesn't work very well.
        # d = np.abs(zs - np.median(zs))
        # mdev = np.median(d)
        # zs = zs[d / mdev < 2.0]

        done = False
        best_z = 0.0
        while not done:
            # Slightly silly outlier dropping that just... works better.
            zs = np.sort(zs)[len(zs) // 10_000 : -len(zs) // 10_000]

            # Get bottom 10% or 15%.
            alpha = 0.1 if points_data.shape[0] < 10_000 else 0.15
            min_z = np.min(zs)
            max_z = np.max(zs)

            zs = zs[zs <= min_z + (max_z - min_z) * alpha]

            best_inliers = 0
            best_z = 0.0
            for _ in range(10_000):
                z = np.random.choice(zs)
                inliers_bool = np.abs(zs - z) < 0.01
                inliers = np.sum(inliers_bool)
                if inliers > best_inliers:
                    best_z = z
                    best_inliers = inliers

            looser_inliers = np.sum(np.abs(filtered_zs - best_z) <= 0.075)
            if looser_inliers <= 3:
                # If we found a really small group... seems like noise. Let's remove the inlier points and re-compute.
                filtered_zs = filtered_zs[np.abs(filtered_zs - best_z) >= 0.01]
                zs = filtered_zs
            else:
                done = True

        # Re-fit plane to inliers.
        floor_z = float(np.median(zs[np.abs(zs - best_z) < 0.01]))

    return points_data, floor_z


def process_take(
    points_path: Path, output_filtered_points_path: Path, output_txt_path: Path
) -> None:
    if not points_path.exists():
        print(f"{points_path} doesn't seem to exist, skipping!")
        return

    # Comment out to overwrite
    if output_txt_path.exists():
        print(f"{output_txt_path} already seems to exist, skipping!")
        return

    _, floor_z = load_point_cloud_and_find_ground(
        points_path, output_filtered_points_path
    )
    output_txt_path.parent.mkdir(exist_ok=True, parents=True)

    if output_txt_path.exists():
        old_floor_z = float(output_txt_path.read_text())
        if abs(old_floor_z - floor_z) >= 0.02:
            print(
                f"!!!! Discrepancy found for {output_txt_path.stem}, before {old_floor_z=}, now {floor_z=}"
            )
        output_txt_path.write_text(str(floor_z))
    else:
        output_txt_path.write_text(str(floor_z))


def main(
    output_dir: Path,
    dataset_path: Path,
    max_workers: int = 16,
) -> None:
    """Compute floor heights for EgoExo4D dataset. Only processes trajectories
    with ego pose annotations.

    Args:
        output_dir: Directory to save detected floor heights to.
        dataset_path: Path to EgoExo4D dataset root.
        max_workers: Processes to use for floor height computation.
    """
    takes_json_path = dataset_path / "takes.json"
    take_meta_from_uid: dict[str, dict[str, Any]] = {
        take_meta["take_uid"]: take_meta
        for take_meta in json.loads(takes_json_path.read_text())
    }
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for split in ("all",):
            if split != "all":
                ego_pose_split_dir = dataset_path / "annotations" / "ego_pose" / split
                camera_pose_dir = ego_pose_split_dir / "camera_pose"
                camera_pose_takes = sorted(
                    map(lambda x: x.stem, camera_pose_dir.iterdir())
                )
            else:
                camera_pose_takes = sorted(take_meta_from_uid.keys())

            points_paths = list[Path]()
            output_points_npz_paths = list[Path]()
            output_txt_paths = list[Path]()
            print(f"Processing {len(camera_pose_takes)} takes for {split=}")
            for i in range(len(camera_pose_takes)):
                take_uid = camera_pose_takes[i]
                take_meta = take_meta_from_uid[take_uid]
                points_paths.append(
                    dataset_path
                    / take_meta["root_dir"]
                    / "trajectory"
                    / "semidense_points.csv.gz"
                )
                output_points_npz_paths.append(
                    output_dir / f"{take_uid}_filtered_points.npz"
                )
                output_txt_paths.append(output_dir / f"{take_uid}.txt")

            # Process!
            tuple(
                tqdm(
                    executor.map(
                        process_take,
                        points_paths,
                        output_points_npz_paths,
                        output_txt_paths,
                        chunksize=1,
                    ),
                    desc=split,
                    total=len(points_paths),
                )
            )


if __name__ == "__main__":
    tyro.cli(main)
