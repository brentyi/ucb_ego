"""Visualize predicted floor heights. Can be used to double-check that our
floor height computation worked."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import tyro
import viser


def main(
    take_name: str,
    floor_height_dir: Path,
    dataset_path: Path,
    override_floor_height: float | None = None,
) -> None:
    """Visualize the computed floor height for a take.

    Args:
        take_name: Name of the take to visualize.
        floor_height_dir: Directory to floor heights that were saved.
        dataset_path: Path to the EgoExo4D dataset. Used for visualize cameras, SLAM poses.
        override_floor_height: If specified, ignore what was saved and
            visualize a different height instead.
    """
    takes_json_path = dataset_path / "takes.json"

    take_meta_from_name: dict[str, dict[str, Any]] = {
        take_meta["take_name"]: take_meta
        for take_meta in json.loads(takes_json_path.read_text())
    }
    take_uid = take_meta_from_name[take_name]["take_uid"]

    floor_height = (
        float((floor_height_dir / f"{take_uid}.txt").read_text())
        if override_floor_height is None
        else override_floor_height
    )
    print(f"{floor_height=}")
    points_data = np.load(floor_height_dir / f"{take_uid}_filtered_points.npz")[
        "points"
    ]

    server = viser.ViserServer()
    server.scene.add_grid("/ground", plane="xy", position=(0, 0, floor_height))
    server.gui.configure_theme(dark_mode=True)

    server.scene.add_point_cloud(
        "/aria_points",
        points=points_data,
        colors=np.cos(points_data + np.arange(3)) / 3.0
        + 0.7,  # Make points colorful :)
        point_size=0.01,
        point_shape="sparkle",
    )

    breakpoint()


if __name__ == "__main__":
    tyro.cli(main)
