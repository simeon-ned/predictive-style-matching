"""PSM predictor feature computation (offline preprocessing + legacy load path)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import find_peaks


def build_cmd_feature_names(
    cmd_traj_horizons: tuple[int, ...],
    *,
    cmd_traj_yaw_frame_deltas: bool = False,
) -> list[str]:
    names = ["root_vx", "root_vy", "root_wz"]
    for h in cmd_traj_horizons:
        names.extend(
            [
                f"traj_pos_{h}_x",
                f"traj_pos_{h}_y",
                f"traj_dir_{h}_x",
                f"traj_dir_{h}_y",
            ]
        )
        if cmd_traj_yaw_frame_deltas:
            for i in range(int(h)):
                names.append(f"traj_{h}_dyaw_{i}")
        else:
            names.append(f"traj_{h}_yaw")
    return names


def compute_predictor_features_for_clip(
    *,
    joint_names: list[str],
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    body_names: list[str],
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_pos_r: np.ndarray,
    body_quat_r: np.ndarray,
    fps: float,
    upper_joint_names: list[str],
    lower_joint_names: list[str],
    feet_bodies: tuple[str, str],
    cmd_traj_horizons: tuple[int, ...],
    cmd_traj_yaw_frame_deltas: bool = False,
) -> dict[str, Any]:
    """Per-clip training arrays (same layout as ``load_motion_data_npz`` output)."""
    from psm.predictor.utils import (
        _compute_cadence_and_double_support,
        _estimate_foot_contact_from_kinematics,
        _finite_diff_velocity_np,
        _postprocess_body_features,
        _smooth_series,
        compute_body_vel_from_qpos_qvel,
        compute_trajectory_features,
        quaternions_to_rpy,
    )

    upper_indices = [joint_names.index(n) for n in upper_joint_names]
    lower_indices = [joint_names.index(n) for n in lower_joint_names]

    lower_j = joint_pos[:, lower_indices].astype(np.float64)
    upper_j = joint_pos[:, upper_indices].astype(np.float64)
    lower_jv = joint_vel[:, lower_indices].astype(np.float64)

    root_pos = qpos[:, 0:3]
    root_quat = qpos[:, 3:7]

    root_vx, root_vy, root_wz = compute_body_vel_from_qpos_qvel(qpos, qvel)
    body_vel = np.stack([root_vx, root_vy, root_wz], axis=1)
    traj = compute_trajectory_features(
        qpos,
        horizons_frames=cmd_traj_horizons,
        traj_yaw_frame_deltas=cmd_traj_yaw_frame_deltas,
    )
    cmd_features = np.concatenate([body_vel, traj], axis=1)

    left_body, right_body = feet_bodies
    left_idx = body_names.index(left_body)
    right_idx = body_names.index(right_body)
    torso_idx = body_names.index("torso_link") if "torso_link" in body_names else None

    left_pos_r = body_pos_r[:, left_idx, :]
    right_pos_r = body_pos_r[:, right_idx, :]
    foot_pos_hist = np.concatenate([left_pos_r, right_pos_r], axis=1)
    foot_vel_hist = _finite_diff_velocity_np(foot_pos_hist, fps)

    body_feats: dict[str, np.ndarray] = {}
    pelvis_roll, pelvis_pitch, _ = quaternions_to_rpy(root_quat)
    body_feats["pelvis_roll"] = pelvis_roll
    body_feats["pelvis_pitch"] = pelvis_pitch

    if torso_idx is not None:
        torso_roll, torso_pitch, _ = quaternions_to_rpy(body_quat_w[:, torso_idx, :])
        body_feats["torso_roll"] = torso_roll
        body_feats["torso_pitch"] = torso_pitch

    _, left_pitch, left_rel_yaw = quaternions_to_rpy(body_quat_r[:, left_idx, :])
    _, right_pitch, right_rel_yaw = quaternions_to_rpy(body_quat_r[:, right_idx, :])
    body_feats["left_foot_pitch"] = left_pitch
    body_feats["right_foot_pitch"] = right_pitch
    body_feats["left_foot_rel_yaw"] = left_rel_yaw
    body_feats["right_foot_rel_yaw"] = right_rel_yaw

    step_length_raw = np.abs(right_pos_r[:, 0] - left_pos_r[:, 0])
    step_width_raw = np.abs(right_pos_r[:, 1] - left_pos_r[:, 1])

    dt = 1.0 / max(float(fps), 1e-6)
    n_samples = joint_pos.shape[0]
    left_contact = _estimate_foot_contact_from_kinematics(left_pos_r, dt)
    right_contact = _estimate_foot_contact_from_kinematics(right_pos_r, dt)
    cadence_hz, double_support_factor = _compute_cadence_and_double_support(
        left_contact, right_contact, dt=dt, window=25
    )

    x_full = np.arange(n_samples, dtype=np.float64)
    peaks, _ = find_peaks(step_length_raw, distance=10)
    if peaks.size < 2:
        peaks = np.array([0, n_samples - 1], dtype=np.int64)
    step_length_interp = np.interp(x_full, peaks, step_length_raw[peaks])
    step_width_interp = np.interp(x_full, peaks, step_width_raw[peaks])
    body_feats["step_length"] = _smooth_series(step_length_interp, window=31, poly=2)
    body_feats["step_width"] = _smooth_series(step_width_interp, window=31, poly=2)
    body_feats["cadence_hz"] = _smooth_series(cadence_hz, window=31, poly=2)
    body_feats["double_support_factor"] = _smooth_series(
        double_support_factor, window=31, poly=2
    )
    body_feats["root_height"] = root_pos[:, 2]

    body_feats = _postprocess_body_features(body_feats)
    body_feature_names = list(body_feats.keys())
    body_features = np.stack([body_feats[k] for k in body_feature_names], axis=1)

    return {
        "psm_lower_joints": lower_j,
        "psm_lower_joint_vel": lower_jv,
        "psm_upper_joints": upper_j,
        "psm_foot_pos_hist": foot_pos_hist,
        "psm_foot_vel_hist": foot_vel_hist,
        "psm_body_vel": body_vel,
        "psm_cmd_features": cmd_features,
        "psm_body_features": body_features,
        "psm_lower_joint_names": lower_joint_names,
        "psm_upper_joint_names": upper_joint_names,
        "psm_body_feature_names": body_feature_names,
        "psm_cmd_feature_names": build_cmd_feature_names(
            cmd_traj_horizons,
            cmd_traj_yaw_frame_deltas=cmd_traj_yaw_frame_deltas,
        ),
        "psm_cmd_traj_horizons": np.asarray(cmd_traj_horizons, dtype=np.int64),
        "psm_cmd_traj_yaw_frame_deltas": np.asarray(
            [int(cmd_traj_yaw_frame_deltas)], dtype=np.int64
        ),
        "root_height_mean": float(np.mean(root_pos[:, 2])),
    }


def compute_predictor_features_from_config(motion: dict[str, Any]) -> dict[str, Any]:
    """Run feature pipeline using joint/feet settings from ``predictor.config``."""
    from psm.predictor.config import (
        CMD_TRAJ_HORIZONS,
        CMD_TRAJ_YAW_FRAME_DELTAS,
        FEET_BODIES,
        LOWER_JOINT_NAMES,
        UPPER_JOINT_NAMES,
    )

    return compute_predictor_features_for_clip(
        joint_names=list(motion["joint_names"]),
        joint_pos=np.asarray(motion["joint_pos"], dtype=np.float64),
        joint_vel=np.asarray(motion["joint_vel"], dtype=np.float64),
        qpos=np.asarray(motion["qpos"], dtype=np.float64),
        qvel=np.asarray(motion["qvel"], dtype=np.float64),
        body_names=list(motion["body_names"]),
        body_pos_w=np.asarray(motion["body_pos_w"], dtype=np.float64),
        body_quat_w=np.asarray(motion["body_quat_w"], dtype=np.float64),
        body_pos_r=np.asarray(motion["body_pos_r"], dtype=np.float64),
        body_quat_r=np.asarray(motion["body_quat_r"], dtype=np.float64),
        fps=float(motion["fps"]),
        upper_joint_names=UPPER_JOINT_NAMES,
        lower_joint_names=LOWER_JOINT_NAMES,
        feet_bodies=(FEET_BODIES[0], FEET_BODIES[1]),
        cmd_traj_horizons=CMD_TRAJ_HORIZONS,
        cmd_traj_yaw_frame_deltas=CMD_TRAJ_YAW_FRAME_DELTAS,
    )
