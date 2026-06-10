"""Add PSM training arrays to existing G1 motion NPZ clips."""

from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from psm.motion.conversion import tyro_cli
from psm.predictor.npz_schema import BUNDLE_FILENAME, augment_clip_npz


def _npz_paths(input_path: str) -> list[Path]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.glob("*.npz") if p.name != BUNDLE_FILENAME)
        if not files:
            raise ValueError(f"No clip NPZ files in {path}")
        return files
    raise ValueError(f"Input path does not exist: {path}")


def main(input_path: str = "data/motions", overwrite: bool = False):
    paths = _npz_paths(input_path)
    updated = sum(augment_clip_npz(p, overwrite=overwrite) for p in tqdm(paths, desc="Augment NPZ"))
    print(f"[INFO] Augmented {updated}/{len(paths)} clip(s)")


def cli() -> None:
    tyro_cli(main, bool_shorthand=("overwrite",))


if __name__ == "__main__":
    cli()
