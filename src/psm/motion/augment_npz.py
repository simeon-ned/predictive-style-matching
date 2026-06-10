"""Add PSM training arrays to existing G1 motion NPZ clips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from psm.motion.conversion import tyro_cli
from psm.predictor.features import compute_predictor_features_from_config
from psm.predictor.npz_schema import (
    BUNDLE_FILENAME,
    clip_has_psm_training,
    expand_motion_npz,
    export_extended_clip,
)


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


def augment_one_npz(path: Path, *, overwrite: bool = False) -> bool:
    if not overwrite:
        with np.load(path, allow_pickle=True) as data:
            if clip_has_psm_training(data):
                return False

    with np.load(path, allow_pickle=True) as npz:
        motion = expand_motion_npz(npz)

    fps = float(motion["fps"])
    log = {
        "fps": [fps],
        "joint_pos": motion["joint_pos"],
        "joint_vel": motion["joint_vel"],
        "qpos": motion["qpos"],
        "qvel": motion["qvel"],
        "body_pos_w": motion["body_pos_w"],
        "body_quat_w": motion["body_quat_w"],
        "body_lin_vel_w": motion["body_lin_vel_w"],
        "body_ang_vel_w": motion["body_ang_vel_w"],
        "body_pos_r": motion["body_pos_r"],
        "body_quat_r": motion["body_quat_r"],
        "body_lin_vel_r": motion.get("body_lin_vel_r", motion["body_pos_r"] * 0),
        "body_ang_vel_r": motion.get("body_ang_vel_r", motion["body_pos_r"] * 0),
    }
    joint_names = list(motion["joint_names"])
    body_names = list(motion["body_names"])
    psm = compute_predictor_features_from_config({**motion, "fps": fps})
    export_extended_clip(
        output_dir=path.parent,
        source_path=path,
        log=log,
        psm=psm,
        joint_names=joint_names,
        body_names=body_names,
    )
    return True


def main(input_path: str = "data/motions", overwrite: bool = False):
    paths = _npz_paths(input_path)
    updated = sum(augment_one_npz(p, overwrite=overwrite) for p in tqdm(paths, desc="Augment NPZ"))
    print(f"[INFO] Augmented {updated}/{len(paths)} clip(s)")


def cli() -> None:
    tyro_cli(main, bool_shorthand=("overwrite",))


if __name__ == "__main__":
    cli()
