"""Normalize motion NPZ files for PSM predictor training.

Supports both the full schema (``qpos``, ``body_pos_r``, …) and the compact
per-clip export (``joint_pos``, ``body_*_w`` only).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np


def _body_pose_vel_in_root_frame(
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


def _npz_scalar_str(npz: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in npz.files:
        return default
    arr = np.asarray(npz[key], dtype=object).reshape(-1)
    if arr.size == 0:
        return default
    return str(arr[0])


def _npz_fps(npz: np.lib.npyio.NpzFile, default: float = 50.0) -> float:
    if "fps" not in npz.files:
        return default
    fps_arr = np.asarray(npz["fps"], dtype=np.float64).reshape(-1)
    return float(fps_arr[0]) if fps_arr.size > 0 else default


@lru_cache(maxsize=1)
def default_g1_body_names() -> tuple[str, ...]:
    from mjlab.entity import Entity

    from psm.assets.unitree_g1.g1_constants import get_g1_robot_cfg

    return Entity(get_g1_robot_cfg()).body_names


def resolve_body_names(
    npz: np.lib.npyio.NpzFile,
    body_names: Sequence[str] | None,
) -> list[str]:
    if body_names is not None:
        return list(body_names)
    if "body_names" in npz.files:
        return [str(x) for x in npz["body_names"].tolist()]
    robot = _npz_scalar_str(npz, "robot", "g1").strip().lower()
    if robot in ("g1", "unitree_g1", "unitree_g1_with_hands"):
        return list(default_g1_body_names())
    raise ValueError(
        f"NPZ has no body_names and robot={robot!r} is unknown. "
        "Pass body_names= to load_motion_data_npz."
    )


def needs_schema_expansion(npz: np.lib.npyio.NpzFile) -> bool:
    """True when the file lacks root-frame / generalized-coord arrays."""
    return "body_pos_r" not in npz.files or "qpos" not in npz.files


def expand_motion_npz(
    npz: np.lib.npyio.NpzFile,
    *,
    body_names: Sequence[str] | None = None,
    root_body_name: str = "pelvis",
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
        pos_r, quat_r, lin_r, ang_r = _body_pose_vel_in_root_frame(
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
    root_body_name: str = "pelvis",
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
