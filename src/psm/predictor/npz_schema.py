"""Motion NPZ schema: compact kinematics, extended PSM training arrays, stacking."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np

PSM_SCHEMA_VERSION = 1
ROOT_BODY_NAME = "pelvis"
BUNDLE_FILENAME = "motions.npz"


def clip_has_psm_training(npz: np.lib.npyio.NpzFile) -> bool:
    return "psm_lower_joints" in npz.files and "psm_body_features" in npz.files


def bundle_has_psm_training(npz: np.lib.npyio.NpzFile) -> bool:
    return clip_has_psm_training(npz) and "segment_start_idx" in npz.files


def body_pose_vel_in_root_frame(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    root_lin_vel: np.ndarray,
    root_ang_vel: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_lin_vel_w: np.ndarray,
    body_ang_vel_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Express body pose and velocity in the root frame (MuJoCo wxyz quats)."""
    conj_root = np.empty(4)
    mujoco.mju_negQuat(conj_root, root_quat)
    n = body_pos_w.shape[0]
    pos_r = np.empty((n, 3))
    quat_r = np.empty((n, 4))
    lin_vel_r = np.empty((n, 3))
    ang_vel_r = np.empty((n, 3))
    for i in range(n):
        diff = body_pos_w[i] - root_pos
        mujoco.mju_rotVecQuat(pos_r[i], diff, conj_root)
        mujoco.mju_mulQuat(quat_r[i], conj_root, body_quat_w[i])
        lin_rel_w = body_lin_vel_w[i] - root_lin_vel - np.cross(root_ang_vel, diff)
        mujoco.mju_rotVecQuat(lin_vel_r[i], lin_rel_w, conj_root)
        ang_diff = body_ang_vel_w[i] - root_ang_vel
        mujoco.mju_rotVecQuat(ang_vel_r[i], ang_diff, conj_root)
    return pos_r, quat_r, lin_vel_r, ang_vel_r


def _npz_fps(npz: np.lib.npyio.NpzFile, default: float = 50.0) -> float:
    if "fps" not in npz.files:
        return default
    fps_arr = np.asarray(npz["fps"], dtype=np.float64).reshape(-1)
    return float(fps_arr[0]) if fps_arr.size > 0 else default


@lru_cache(maxsize=1)
def g1_body_names() -> tuple[str, ...]:
    from mjlab.entity import Entity

    from psm.assets.unitree_g1.g1_constants import get_g1_robot_cfg

    return Entity(get_g1_robot_cfg()).body_names


def resolve_body_names(
    npz: np.lib.npyio.NpzFile,
    body_names: Sequence[str] | None = None,
) -> list[str]:
    if body_names is not None:
        return list(body_names)
    if "body_names" in npz.files:
        return [str(x) for x in npz["body_names"].tolist()]
    return list(g1_body_names())


def needs_schema_expansion(npz: np.lib.npyio.NpzFile) -> bool:
    """True when the file lacks root-frame / generalized-coord arrays."""
    return "body_pos_r" not in npz.files or "qpos" not in npz.files


def expand_motion_npz(
    npz: np.lib.npyio.NpzFile,
    *,
    body_names: Sequence[str] | None = None,
    root_body_name: str = ROOT_BODY_NAME,
) -> dict[str, Any]:
    """Return a dict with the full predictor schema, expanding compact NPZ if needed."""
    names = resolve_body_names(npz, body_names)
    joint_names = [str(x) for x in npz["joint_names"].tolist()]
    joint_pos = np.asarray(npz["joint_pos"], dtype=np.float64)
    body_pos_w = np.asarray(npz["body_pos_w"], dtype=np.float64)
    body_quat_w = np.asarray(npz["body_quat_w"], dtype=np.float64)
    body_lin_vel_w = np.asarray(npz["body_lin_vel_w"], dtype=np.float64)
    body_ang_vel_w = np.asarray(npz["body_ang_vel_w"], dtype=np.float64)

    if body_pos_w.shape[1] != len(names):
        raise ValueError(
            f"body_pos_w has {body_pos_w.shape[1]} bodies but body_names has {len(names)}"
        )

    out: dict[str, Any] = {
        "joint_names": joint_names,
        "joint_pos": joint_pos,
        "body_names": names,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "fps": _npz_fps(npz),
    }

    if "joint_vel" in npz.files:
        out["joint_vel"] = np.asarray(npz["joint_vel"], dtype=np.float64)
    else:
        out["joint_vel"] = _finite_diff_joint_vel(joint_pos, out["fps"])

    if not needs_schema_expansion(npz):
        out["qpos"] = np.asarray(npz["qpos"], dtype=np.float64)
        out["qvel"] = np.asarray(npz["qvel"], dtype=np.float64)
        out["body_pos_r"] = np.asarray(npz["body_pos_r"], dtype=np.float64)
        out["body_quat_r"] = np.asarray(npz["body_quat_r"], dtype=np.float64)
        if "body_lin_vel_r" in npz.files:
            out["body_lin_vel_r"] = np.asarray(npz["body_lin_vel_r"], dtype=np.float64)
        if "body_ang_vel_r" in npz.files:
            out["body_ang_vel_r"] = np.asarray(npz["body_ang_vel_r"], dtype=np.float64)
        return out

    try:
        root_idx = names.index(root_body_name)
    except ValueError as e:
        raise KeyError(
            f"Root body {root_body_name!r} not in body_names for NPZ schema expansion"
        ) from e

    root_pos = body_pos_w[:, root_idx, :]
    root_quat = body_quat_w[:, root_idx, :]
    root_lin_vel = body_lin_vel_w[:, root_idx, :]
    root_ang_vel = body_ang_vel_w[:, root_idx, :]

    out["qpos"] = np.concatenate([root_pos, root_quat, joint_pos], axis=1)
    out["qvel"] = np.concatenate(
        [root_lin_vel, root_ang_vel, out["joint_vel"]],
        axis=1,
    )

    n_frames = body_pos_w.shape[0]
    body_pos_r = np.empty_like(body_pos_w)
    body_quat_r = np.empty_like(body_quat_w)
    body_lin_vel_r = np.empty_like(body_lin_vel_w)
    body_ang_vel_r = np.empty_like(body_ang_vel_w)
    for t in range(n_frames):
        pos_r, quat_r, lin_r, ang_r = body_pose_vel_in_root_frame(
            root_pos[t],
            root_quat[t],
            root_lin_vel[t],
            root_ang_vel[t],
            body_pos_w[t],
            body_quat_w[t],
            body_lin_vel_w[t],
            body_ang_vel_w[t],
        )
        body_pos_r[t] = pos_r
        body_quat_r[t] = quat_r
        body_lin_vel_r[t] = lin_r
        body_ang_vel_r[t] = ang_r

    out["body_pos_r"] = body_pos_r
    out["body_quat_r"] = body_quat_r
    out["body_lin_vel_r"] = body_lin_vel_r
    out["body_ang_vel_r"] = body_ang_vel_r
    return out


def load_expanded_motion_npz(
    path: str | Path,
    *,
    body_names: Sequence[str] | None = None,
    root_body_name: str = ROOT_BODY_NAME,
) -> dict[str, Any]:
    npz = np.load(path, allow_pickle=True)
    try:
        return expand_motion_npz(
            npz,
            body_names=body_names,
            root_body_name=root_body_name,
        )
    finally:
        npz.close()


def _finite_diff_joint_vel(joint_pos: np.ndarray, fps: float) -> np.ndarray:
    dt = 1.0 / max(float(fps), 1e-6)
    v = np.empty_like(joint_pos)
    v[0] = (joint_pos[1] - joint_pos[0]) / dt
    v[-1] = (joint_pos[-1] - joint_pos[-2]) / dt
    if joint_pos.shape[0] > 2:
        v[1:-1] = (joint_pos[2:] - joint_pos[:-2]) / (2.0 * dt)
    return v


# --- paths (G1 / data/motions layout) ---


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate repo root (pyproject.toml)")


def motions_dir() -> Path:
    return repo_root() / "data" / "motions"


def training_bundle_path(dataset_root: Path | None = None) -> Path:
    root = (dataset_root or motions_dir()).resolve()
    return root / BUNDLE_FILENAME


def list_clip_npz_files(dataset_root: Path) -> list[Path]:
    return sorted(
        p for p in dataset_root.resolve().glob("*.npz") if p.is_file() and p.name != BUNDLE_FILENAME
    )


def resolve_conversion_paths(
    *,
    dataset: str | None = None,
    dataset_path: str | None = None,
    input_path: str | None = None,
    output_dir: str | None = None,
) -> tuple[str, str]:
    if dataset_path is not None and dataset is not None:
        raise ValueError("Use only one of --dataset or --dataset-path")

    if dataset_path is not None:
        root = Path(dataset_path).expanduser()
        root = (Path.cwd() / root).resolve() if not root.is_absolute() else root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {root}")
        out = str(output_dir or root)
        raw = root / "raw"
        inp = str(input_path or (raw if raw.is_dir() else root))
        return inp, out

    if dataset is not None:
        root = motions_dir() if dataset == "motions" else repo_root() / "data" / dataset
        out = str(output_dir or root)
        raw = root / "raw"
        inp = str(input_path or (raw if raw.is_dir() else root))
        return inp, out

    if input_path is None or output_dir is None:
        raise ValueError("Provide --dataset, --dataset-path, or --input-path and --output-dir")
    return input_path, output_dir


# --- extended clip I/O ---


def save_extended_clip_npz(
    *,
    output_path: Path,
    log: dict[str, Any],
    psm: dict[str, Any],
    joint_names: list[str],
    body_names: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "fps": np.asarray(log["fps"], dtype=np.float32),
        "joint_names": np.asarray(joint_names, dtype=object),
        "body_names": np.asarray(body_names, dtype=object),
        "robot": np.asarray("g1", dtype=object),
        "joint_pos": np.asarray(log["joint_pos"], dtype=np.float32),
        "joint_vel": np.asarray(log["joint_vel"], dtype=np.float32),
        "qpos": np.asarray(log["qpos"], dtype=np.float32),
        "qvel": np.asarray(log["qvel"], dtype=np.float32),
        "body_pos_w": np.asarray(log["body_pos_w"], dtype=np.float32),
        "body_quat_w": np.asarray(log["body_quat_w"], dtype=np.float32),
        "body_lin_vel_w": np.asarray(log["body_lin_vel_w"], dtype=np.float32),
        "body_ang_vel_w": np.asarray(log["body_ang_vel_w"], dtype=np.float32),
        "body_pos_r": np.asarray(log["body_pos_r"], dtype=np.float32),
        "body_quat_r": np.asarray(log["body_quat_r"], dtype=np.float32),
        "body_lin_vel_r": np.asarray(log["body_lin_vel_r"], dtype=np.float32),
        "body_ang_vel_r": np.asarray(log["body_ang_vel_r"], dtype=np.float32),
        "psm_schema_version": np.asarray([PSM_SCHEMA_VERSION], dtype=np.int64),
        "psm_lower_joints": np.asarray(psm["psm_lower_joints"], dtype=np.float32),
        "psm_lower_joint_vel": np.asarray(psm["psm_lower_joint_vel"], dtype=np.float32),
        "psm_upper_joints": np.asarray(psm["psm_upper_joints"], dtype=np.float32),
        "psm_foot_pos_hist": np.asarray(psm["psm_foot_pos_hist"], dtype=np.float32),
        "psm_foot_vel_hist": np.asarray(psm["psm_foot_vel_hist"], dtype=np.float32),
        "psm_body_vel": np.asarray(psm["psm_body_vel"], dtype=np.float32),
        "psm_cmd_features": np.asarray(psm["psm_cmd_features"], dtype=np.float32),
        "psm_body_features": np.asarray(psm["psm_body_features"], dtype=np.float32),
        "psm_lower_joint_names": np.asarray(psm["psm_lower_joint_names"], dtype=object),
        "psm_upper_joint_names": np.asarray(psm["psm_upper_joint_names"], dtype=object),
        "psm_body_feature_names": np.asarray(psm["psm_body_feature_names"], dtype=object),
        "psm_cmd_feature_names": np.asarray(psm["psm_cmd_feature_names"], dtype=object),
        "psm_cmd_traj_horizons": np.asarray(psm["psm_cmd_traj_horizons"], dtype=np.int64),
        "psm_cmd_traj_yaw_frame_deltas": np.asarray(
            psm["psm_cmd_traj_yaw_frame_deltas"], dtype=np.int64
        ),
    }
    np.savez(output_path, **payload)


def export_extended_clip(
    *,
    output_dir: str | Path,
    source_path: Path,
    log: dict[str, Any],
    psm: dict[str, Any],
    joint_names: list[str],
    body_names: list[str],
) -> Path:
    out = Path(output_dir).expanduser().resolve() / f"{source_path.stem}.npz"
    save_extended_clip_npz(
        output_path=out,
        log=log,
        psm=psm,
        joint_names=joint_names,
        body_names=body_names,
    )
    print(f"[INFO] Exported clip: {out}")
    return out


def write_extended_clip_from_log(
    *,
    output_dir: str | Path,
    source_path: Path,
    log: dict[str, Any],
    joint_names: list[str],
    body_names: list[str],
    fps: float,
) -> Path:
    """Compute PSM features from a FK log and write extended NPZ."""
    from psm.predictor.features import compute_predictor_features_from_config

    log = {**log, "fps": [fps]}
    motion = {
        "joint_names": joint_names,
        "body_names": body_names,
        "joint_pos": log["joint_pos"],
        "joint_vel": log["joint_vel"],
        "qpos": log["qpos"],
        "qvel": log["qvel"],
        "body_pos_w": log["body_pos_w"],
        "body_quat_w": log["body_quat_w"],
        "body_pos_r": log["body_pos_r"],
        "body_quat_r": log["body_quat_r"],
        "fps": fps,
    }
    psm = compute_predictor_features_from_config(motion)
    return export_extended_clip(
        output_dir=output_dir,
        source_path=source_path,
        log=log,
        psm=psm,
        joint_names=joint_names,
        body_names=body_names,
    )


def stack_training_bundle(*, clips: list[Path], output_path: Path) -> Path:
    """Concatenate extended clip NPZs into one training bundle."""
    from tqdm import tqdm

    if not clips:
        raise ValueError("No clips to stack")

    merged: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "psm_lower_joints",
            "psm_lower_joint_vel",
            "psm_upper_joints",
            "psm_foot_pos_hist",
            "psm_foot_vel_hist",
            "psm_body_vel",
            "psm_cmd_features",
            "psm_body_features",
        )
    }
    segment_start_idx: list[int] = []
    segment_length: list[int] = []
    segment_source: list[str] = []
    running = 0
    meta: dict[str, Any] = {}

    for path in tqdm(clips, desc="Stack clips"):
        data = np.load(path, allow_pickle=True)
        if not clip_has_psm_training(data):
            raise ValueError(f"{path} lacks PSM keys; run psm-augment-npz or psm-csv-to-npz first")
        length = int(data["psm_lower_joints"].shape[0])
        for key in merged:
            merged[key].append(np.asarray(data[key], dtype=np.float32))
        segment_start_idx.append(running)
        segment_length.append(length)
        segment_source.append(str(path.resolve()))
        running += length
        if not meta:
            meta = {
                "joint_names": data["joint_names"],
                "body_names": data.get("body_names"),
                "robot": data.get("robot", np.asarray("g1", dtype=object)),
                "fps": data["fps"],
                "psm_lower_joint_names": data["psm_lower_joint_names"],
                "psm_upper_joint_names": data["psm_upper_joint_names"],
                "psm_body_feature_names": data["psm_body_feature_names"],
                "psm_cmd_feature_names": data["psm_cmd_feature_names"],
                "psm_cmd_traj_horizons": data["psm_cmd_traj_horizons"],
                "psm_cmd_traj_yaw_frame_deltas": data["psm_cmd_traj_yaw_frame_deltas"],
            }

    payload: dict[str, Any] = {
        "psm_schema_version": np.asarray([PSM_SCHEMA_VERSION], dtype=np.int64),
        "segment_start_idx": np.asarray(segment_start_idx, dtype=np.int64),
        "segment_length": np.asarray(segment_length, dtype=np.int64),
        "segment_source": np.asarray(segment_source, dtype=object),
        **meta,
    }
    for key, parts in merged.items():
        payload[key] = np.concatenate(parts, axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    print(f"[INFO] Wrote training bundle: {output_path} ({len(clips)} clips, {running} frames)")
    return output_path
