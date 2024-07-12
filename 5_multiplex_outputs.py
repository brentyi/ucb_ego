import json
from pathlib import Path

import tyro


def main(
    physical_json: Path,
    procedural_json: Path,
    out_json: Path,
    adapter_output_dir: Path,
) -> None:
    out_json.parent.mkdir(exist_ok=True)
    physical = json.loads(physical_json.read_text())
    procedural = json.loads(procedural_json.read_text())

    for take_uid in physical:
        if (
            adapter_output_dir / f"{take_uid}_type.txt"
        ).read_text().strip() == "procedural":
            physical[take_uid] = procedural[take_uid]

    out_json.write_text(json.dumps(physical))


if __name__ == "__main__":
    tyro.cli(main)
