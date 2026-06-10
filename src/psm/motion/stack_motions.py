"""Stack extended clip NPZs into one G1 training bundle."""

from __future__ import annotations

from pathlib import Path

from psm.motion.conversion import tyro_cli
from psm.predictor.npz_schema import ensure_training_bundle


def main(
    dataset_path: str = "data/motions",
    output_path: str | None = None,
    force: bool = False,
):
    root = Path(dataset_path).expanduser().resolve()
    if output_path is not None:
        raise ValueError("output_path is deprecated; bundle is always data/motions/motions.npz")
    ensure_training_bundle(root, force=force)


def cli() -> None:
    tyro_cli(main, bool_shorthand=("force",))


if __name__ == "__main__":
    cli()
