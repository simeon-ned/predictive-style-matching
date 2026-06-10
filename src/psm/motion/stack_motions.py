"""Stack extended clip NPZs into one G1 training bundle."""

from __future__ import annotations

from pathlib import Path

from psm.motion.conversion import tyro_cli
from psm.predictor.npz_schema import (
    list_clip_npz_files,
    stack_training_bundle,
    training_bundle_path,
)


def main(
    dataset_path: str = "data/motions",
    output_path: str | None = None,
    force: bool = False,
):
    root = Path(dataset_path).expanduser().resolve()
    out = Path(output_path).expanduser().resolve() if output_path else training_bundle_path(root)
    clips = list_clip_npz_files(root)
    if not clips:
        raise FileNotFoundError(f"No clip NPZ files in {root}")

    if not force and out.is_file():
        mtime = out.stat().st_mtime
        if all(mtime >= p.stat().st_mtime for p in clips):
            print(f"[INFO] Bundle up to date: {out}")
            return

    stack_training_bundle(clips=clips, output_path=out)


def cli() -> None:
    tyro_cli(main, bool_shorthand=("force",))


if __name__ == "__main__":
    cli()
