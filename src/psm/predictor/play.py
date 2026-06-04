#!/usr/bin/env python3
"""
NPZ-based tester for the arm matching network.

- Loads a LAFAN-style NPZ motion (same schema as lafan_dataset).
- Runs the trained predictor in a receding-horizon fashion to overwrite upper joints.
- Visualizes recorded motion (solid) vs prediction (ghost) via ``run_viser_visualization``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import mujoco
import numpy as np
import torch

from psm.assets.unitree_g1.g1_constants import G1_XML

from .config import (
    DEVICE,
    HISTORY_HORIZON,
    PREDICTION_HORIZON,
    LOGS_DIR,
)
from .psm_predictor import PsmPredictor
from .utils import (
    _finite_diff_velocity_np,
    compute_body_vel_from_qpos_qvel,
    compute_trajectory_features,
    quaternions_to_rpy,
    load_config_yaml,
    load_metadata,
)
from .visualization import run_viser_visualization


def _find_latest_log_dir() -> str:
    logs_dir = Path(LOGS_DIR)
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")
    candidates = []
    for d in logs_dir.iterdir():
        if not d.is_dir():
            continue
        if (d / "predictor.pth").exists() and (d / "metadata.pkl").exists():
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(f"No complete log directories found in: {logs_dir}")
    latest = sorted(candidates, key=lambda p: p.name)[-1]
    return str(latest)


def _load_model_and_meta(log_dir: str) -> Tuple[PsmPredictor, dict, dict]:
    meta_path = os.path.join(log_dir, "metadata.pkl")
    model_path = os.path.join(log_dir, "predictor.pth")
    cfg_path = os.path.join(log_dir, "config.yaml")

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(meta_path)

    metadata = load_metadata(meta_path)
    config_yaml = load_config_yaml(cfg_path) if os.path.exists(cfg_path) else {}

    state_dict = torch.load(model_path, map_location=DEVICE)
    ctor = dict(metadata["constructor_params"])
    if "use_lower_joint_velocity" not in ctor and "use_lower_velocity" in ctor:
        ctor["use_lower_joint_velocity"] = bool(ctor["use_lower_velocity"])
    if "use_lower_joint_velocity" not in ctor:
        ctor["use_lower_joint_velocity"] = False
    if "use_foot_velocity" not in ctor:
        ctor["use_foot_velocity"] = False
    if "foot_vel_mean" not in ctor:
        ctor["foot_vel_mean"] = None
    if "foot_vel_std" not in ctor:
        ctor["foot_vel_std"] = None
    ctor.pop("use_lower_velocity", None)
    # Must match keys saved in ``metadata.pkl`` / ``train.py`` ctor (and ``psm`` load path).
    allowed = {
        "output_size",
        "y_mean",
        "y_std",
        "leg_pos_mean",
        "leg_pos_std",
        "foot_pos_mean",
        "foot_pos_std",
        "body_vel_mean",
        "body_vel_std",
        "cmd_mean",
        "cmd_std",
        "num_lower",
        "history_horizon",
        "prediction_horizon",
        "cmd_feature_dim",
        "history_input_mode",
        "joints_history_weight",
        "feet_history_weight",
        "use_lower_joint_velocity",
        "use_foot_velocity",
        "leg_vel_mean",
        "leg_vel_std",
        "foot_vel_mean",
        "foot_vel_std",
        "history_recency_decay",
        "encoder_type",
        "encoder_hidden_size",
        "conv1d_channels",
        "hidden_size",
        "gru_num_layers",
        "conv1d_kernel_size",
        "head_hidden_depth",
        "head_dropout",
        "activation",
    }
    ctor = {k: v for k, v in ctor.items() if k in allowed}
    if "hidden_size" not in ctor:
        raise KeyError("metadata['constructor_params'] must include 'hidden_size'.")
    model = PsmPredictor(**ctor).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata, config_yaml


def _default_mujoco_model_path() -> Path:
    """Default G1 MJCF from ``psm.assets.unitree_g1`` (same as RL env)."""
    if not G1_XML.is_file():
        raise FileNotFoundError(
            f"Could not find MuJoCo model at {G1_XML}. Pass --model-xml."
        )
    return G1_XML


def _build_joint_name_to_qpos_idx(model: mujoco.MjModel) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for j in range(model.njnt):
        name = model.joint(j).name
        if not name:
            continue
        adr = int(model.jnt_qposadr[j])
        mapping[name] = adr
    return mapping


def _extract_lower_upper_from_npz(
    npz_path: str,
    lower_joint_names,
    upper_joint_names,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    joint_names = [str(x) for x in data["joint_names"].tolist()]
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    qpos = np.asarray(data["qpos"], dtype=np.float64)

    lower_idx = [joint_names.index(n) for n in lower_joint_names]
    upper_idx = [joint_names.index(n) for n in upper_joint_names]

    lower = joint_pos[:, lower_idx]
    upper = joint_pos[:, upper_idx]
    return lower, upper, qpos


def _extract_foot_pos_hist_from_npz(npz_path: str, feet_bodies: tuple[str, str]) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    body_names = [str(x) for x in data["body_names"].tolist()]
    body_pos_r = np.asarray(data["body_pos_r"], dtype=np.float64)
    lb, rb = feet_bodies
    li = body_names.index(lb)
    ri = body_names.index(rb)
    left_pos_r = body_pos_r[:, li, :]
    right_pos_r = body_pos_r[:, ri, :]
    return np.concatenate([left_pos_r, right_pos_r], axis=1)


def _extract_foot_vel_from_npz(npz_path: str, feet_bodies: tuple[str, str]) -> np.ndarray:
    """Root-frame foot velocities from finite differences (matches training ``foot_vel_hist``)."""
    data = np.load(npz_path, allow_pickle=True)
    fp = _extract_foot_pos_hist_from_npz(npz_path, feet_bodies)
    if "fps" in data:
        fps_arr = np.asarray(data["fps"], dtype=np.float64).reshape(-1)
        fps = float(fps_arr[0]) if fps_arr.size > 0 else 50.0
    else:
        fps = 50.0
    return _finite_diff_velocity_np(fp, fps)


def _extract_lower_vel_from_npz(npz_path: str, lower_joint_names) -> np.ndarray:
    """Per-timestep lower joint velocities (matches ``utils.load_motion_data_npz``)."""
    data = np.load(npz_path, allow_pickle=True)
    joint_names = [str(x) for x in data["joint_names"].tolist()]
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    lower_idx = [joint_names.index(n) for n in lower_joint_names]
    lp = joint_pos[:, lower_idx]
    if "joint_vel" in data.files:
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
        return joint_vel[:, lower_idx]
    if "fps" in data:
        fps_arr = np.asarray(data["fps"], dtype=np.float64).reshape(-1)
        fps = float(fps_arr[0]) if fps_arr.size > 0 else 50.0
    else:
        fps = 50.0
    dt = 1.0 / max(fps, 1e-6)
    v = np.empty_like(lp)
    v[0] = (lp[1] - lp[0]) / dt
    v[-1] = (lp[-1] - lp[-2]) / dt
    if lp.shape[0] > 2:
        v[1:-1] = (lp[2:] - lp[:-2]) / (2.0 * dt)
    return v


def _build_velocity_series_from_npz(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild velocity series from NPZ.

    Returns:
        body_root_vel: (T, 3)  [root_vx, root_vy, root_wz] in root body frame
        yaw_vel3:   (T, 3)  [vx, vy, wz] yaw-aligned (for Viser arrow only)
    """
    data = np.load(npz_path, allow_pickle=True)
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    qvel = np.asarray(data["qvel"], dtype=np.float64)
    vx_local, vy_local, wz_local = compute_body_vel_from_qpos_qvel(qpos, qvel)
    body_root_vel = np.stack([vx_local, vy_local, wz_local], axis=1)
    yaw_vel3 = np.stack([vx_local, vy_local, wz_local], axis=1)
    return body_root_vel, yaw_vel3


def _compute_body_features_gt_from_npz(npz_path: str) -> dict[str, np.ndarray]:
    """Compute ground-truth body features from NPZ (same formulas as utils)."""
    from .config import FEET_BODIES
    from scipy.signal import find_peaks, savgol_filter

    data = np.load(npz_path, allow_pickle=True)
    qpos = np.asarray(data["qpos"], dtype=np.float64)
    root_pos = qpos[:, 0:3]
    root_quat = qpos[:, 3:7]  # wxyz

    body_names = [str(x) for x in data["body_names"].tolist()]
    body_pos_r = np.asarray(data["body_pos_r"], dtype=np.float64)
    body_quat_r = np.asarray(data["body_quat_r"], dtype=np.float64)
    body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float64)

    left_body, right_body = FEET_BODIES
    left_idx = body_names.index(left_body)
    right_idx = body_names.index(right_body)

    # quaternions_to_rpy returns (roll, pitch, yaw)
    _, left_pitch, left_rel_yaw = quaternions_to_rpy(body_quat_r[:, left_idx, :])
    _, right_pitch, right_rel_yaw = quaternions_to_rpy(body_quat_r[:, right_idx, :])

    # Step metrics from root-relative positions (match utils.py peak-interpolated definition)
    left_pos_r = body_pos_r[:, left_idx, :]
    right_pos_r = body_pos_r[:, right_idx, :]
    body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float64)
    left_pos_w = body_pos_w[:, left_idx, :]
    right_pos_w = body_pos_w[:, right_idx, :]
    step_length_raw = np.abs(right_pos_r[:, 0] - left_pos_r[:, 0])
    step_width_raw = np.abs(right_pos_r[:, 1] - left_pos_r[:, 1])

    # Match utils.py smoothing (Savitzky-Golay).
    def _smooth_series(series: np.ndarray, window: int = 21, poly: int = 3) -> np.ndarray:
        series = np.asarray(series, dtype=np.float64)
        n = series.shape[0]
        if n < 5:
            return series
        window_eff = min(window, n if n % 2 == 1 else n - 1)
        if window_eff < 5:
            return series
        try:
            return savgol_filter(
                series,
                window_length=window_eff,
                polyorder=min(poly, window_eff - 2),
            )
        except Exception:
            return series

    from .utils import (
        _compute_cadence_and_double_support,
        _estimate_foot_contact_from_kinematics,
    )

    if "fps" in data.files:
        fps = float(np.asarray(data["fps"], dtype=np.float64).reshape(-1)[0])
    else:
        fps = 50.0
    dt = 1.0 / max(fps, 1e-6)
    left_contact = _estimate_foot_contact_from_kinematics(left_pos_r, dt)
    right_contact = _estimate_foot_contact_from_kinematics(right_pos_r, dt)
    n_samples = step_length_raw.shape[0]
    x_full = np.arange(n_samples, dtype=np.float64)
    peaks, _ = find_peaks(step_length_raw, distance=10)
    if peaks.size < 2:
        peaks = np.array([0, n_samples - 1], dtype=np.int64)
    step_length_interp = np.interp(x_full, peaks, step_length_raw[peaks])
    step_width_interp = np.interp(x_full, peaks, step_width_raw[peaks])
    step_length_interp = _smooth_series(step_length_interp, window=31, poly=2)
    step_width_interp = _smooth_series(step_width_interp, window=31, poly=2)
    cadence_hz, double_support_factor = _compute_cadence_and_double_support(
        left_contact, right_contact, dt=dt, window=25
    )
    torso_idx = body_names.index("torso_link") if "torso_link" in body_names else None
    pelvis_roll, pelvis_pitch, _ = quaternions_to_rpy(root_quat)
    if torso_idx is not None:
        torso_roll, torso_pitch, _ = quaternions_to_rpy(body_quat_w[:, torso_idx, :])
    else:
        torso_roll = np.full_like(pelvis_roll, np.nan)
        torso_pitch = np.full_like(pelvis_pitch, np.nan)

    return {
        "left_foot_pitch": left_pitch,
        "right_foot_pitch": right_pitch,
        "left_foot_rel_yaw": left_rel_yaw,
        "right_foot_rel_yaw": right_rel_yaw,
        "step_length": step_length_interp,
        "step_width": step_width_interp,
        "cadence_hz": cadence_hz,
        "double_support_factor": double_support_factor,
        "root_height": root_pos[:, 2],
        "pelvis_roll": pelvis_roll,
        "pelvis_pitch": pelvis_pitch,
        "torso_roll": torso_roll,
        "torso_pitch": torso_pitch,
    }


def _receding_horizon_predict(
    model: PsmPredictor,
    npz_path: str,
    metadata: dict,
) -> Tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Run receding-horizon prediction over a single NPZ motion.

    Returns:
        original_qpos, predicted_qpos, predicted_body_features
    """
    from .config import HISTORY_HORIZON as H, PREDICTION_HORIZON as P

    lower_names = metadata["lower_order"]
    upper_names = metadata["upper_order"]
    feet_bodies = tuple(metadata.get("feet_bodies", ("left_ankle_roll_link", "right_ankle_roll_link")))
    body_feature_names = metadata.get("body_feature_names", [])
    hist_mode = str(metadata.get("history_input_mode", "both")).lower()
    use_joint_hist = hist_mode in ("joints", "both")
    use_foot_hist = hist_mode in ("feet", "both")

    lower, _, qpos = _extract_lower_upper_from_npz(npz_path, lower_names, upper_names)
    foot_pos_hist_all = _extract_foot_pos_hist_from_npz(npz_path, feet_bodies)
    foot_vel_hist_all = _extract_foot_vel_from_npz(npz_path, feet_bodies)
    lower_vel = _extract_lower_vel_from_npz(npz_path, lower_names)
    cp = metadata.get("constructor_params", {})
    use_lower_vel = bool(
        metadata.get("use_lower_joint_velocity", cp.get("use_lower_joint_velocity", cp.get("use_lower_velocity", False)))
    )
    use_foot_vel = bool(metadata.get("use_foot_velocity", cp.get("use_foot_velocity", False)))
    data = np.load(npz_path, allow_pickle=True)
    qvel = np.asarray(data["qvel"], dtype=np.float64)
    root_vx, root_vy, root_wz = compute_body_vel_from_qpos_qvel(qpos, qvel)
    body_vel_series = np.stack([root_vx, root_vy, root_wz], axis=1)
    traj_horizons = tuple(int(h) for h in metadata.get("cmd_traj_horizons", (8, 16, 24)))
    traj_yaw_frame_deltas = bool(metadata.get("cmd_traj_yaw_frame_deltas", False))
    traj_series = compute_trajectory_features(
        qpos,
        horizons_frames=traj_horizons,
        traj_yaw_frame_deltas=traj_yaw_frame_deltas,
    )

    T = lower.shape[0]
    predicted_qpos = qpos.copy()
    pred_body_feats = {
        name: np.full(T, np.nan, dtype=np.float64) for name in body_feature_names
    }

    # Build name->joint index mapping inside joint block of qpos
    joint_names = [str(x) for x in data["joint_names"].tolist()]
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    # qpos layout: [root(7), joints(in joint_names order)]
    joint_start = 7
    name_to_qpos_idx = {
        n: joint_start + joint_names.index(n) for n in joint_names
    }
    lower_qpos_idx = [name_to_qpos_idx[n] for n in lower_names]
    upper_qpos_idx = [name_to_qpos_idx[n] for n in upper_names]

    # Match training: only predict frames where the full prediction horizon exists.
    # In training, start indices are sampled so future steps [t, t+P-1] are valid.
    t_end = T - P + 1
    for t in range(H, t_end):
        # History window [t-H, ..., t-1]; base timestep (last history index) = t-1.
        hist_lower = lower[t - H : t, :]  # (H, n_lower)
        hist_foot = foot_pos_hist_all[t - H : t, :]  # (H, 6)
        hist_lower_vel = lower_vel[t - H : t, :]
        hist_foot_vel = foot_vel_hist_all[t - H : t, :]
        vel_hist = body_vel_series[t - H : t, :]  # (H, 3)

        # Command-style conditioning at the base timestep (t-1):
        # [root_vx,root_vy,root_wz] + root-frame trajectory descriptor.
        k = t - 1
        vel_future = np.concatenate([body_vel_series[k], traj_series[k]], axis=0)

        with torch.no_grad():
            lp = (
                torch.from_numpy(hist_lower).float().to(DEVICE).unsqueeze(0)
                if use_joint_hist
                else None
            )
            fp = (
                torch.from_numpy(hist_foot).float().to(DEVICE).unsqueeze(0)
                if use_foot_hist
                else None
            )
            bvh = torch.from_numpy(vel_hist).float().to(DEVICE).unsqueeze(0)
            bvf = torch.from_numpy(vel_future).float().to(DEVICE).unsqueeze(0)
            lv = (
                torch.from_numpy(hist_lower_vel).float().to(DEVICE).unsqueeze(0)
                if use_lower_vel
                else None
            )
            fv = (
                torch.from_numpy(hist_foot_vel).float().to(DEVICE).unsqueeze(0)
                if use_foot_vel
                else None
            )
            y = model.predict(lp, fp, bvh, bvf, lv, fv).cpu().numpy().squeeze()

        n_upper = len(upper_names)
        n_body_feats = len(body_feature_names)
        # Training output layout:
        #   [upper_future_flat (P*n_upper), body_future_flat (P*n_body_feats)]
        upper_flat = y[: P * n_upper]
        body_flat = y[P * n_upper :]

        # Receding horizon uses only the first prediction step (k=0).
        upper_pred_step0 = upper_flat[:n_upper]
        body_pred_step0 = body_flat[:n_body_feats] if n_body_feats > 0 else np.array([])

        for i, idx in enumerate(upper_qpos_idx):
            predicted_qpos[t, idx] = upper_pred_step0[i]

        # Store predicted body feature values for this frame (t) using first step.
        if body_feature_names:
            if body_pred_step0.shape[0] == len(body_feature_names):
                for i, name in enumerate(body_feature_names):
                    pred_body_feats[name][t] = float(body_pred_step0[i])

    return qpos, predicted_qpos, pred_body_feats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test arm matching predictor on a LAFAN NPZ motion with Viser visualization.",
    )
    parser.add_argument(
        "--npz",
        type=str,
        required=True,
        help="Path to input NPZ motion (lafan_dataset-style).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Log directory containing predictor.pth and metadata.pkl (defaults to latest).",
    )
    parser.add_argument(
        "--model-xml",
        type=str,
        default=None,
        help="Path to MuJoCo XML (defaults to psm.assets.unitree_g1 xmls/g1.xml).",
    )
    args = parser.parse_args()

    log_dir = args.log_dir or _find_latest_log_dir()
    print(f"Using log directory: {log_dir}")
    model, metadata, cfg = _load_model_and_meta(log_dir)

    original_qpos, predicted_qpos, pred_body_feats = _receding_horizon_predict(
        model, args.npz, metadata
    )

    data_npz = np.load(args.npz, allow_pickle=True)
    if "fps" in data_npz:
        fps_arr = np.asarray(data_npz["fps"], dtype=np.float64).reshape(-1)
        fps = float(fps_arr[0]) if fps_arr.size > 0 else 50.0
    else:
        fps = 50.0

    model_xml = args.model_xml or str(_default_mujoco_model_path())
    print(f"Using MuJoCo model XML: {model_xml}")

    # Build commanded velocity series and body feature time series for plotting.
    _, body_vel_series = _build_velocity_series_from_npz(
        args.npz
    )  # yaw-aligned (T,3) for Viser arrow
    # Ground-truth body features for debug/verification.
    gt_body_feats = _compute_body_features_gt_from_npz(args.npz)

    # Plot all predicted and GT traces with distinct term names.
    body_features_for_plot: dict[str, np.ndarray] = {}
    for k, pred_arr in pred_body_feats.items():
        body_features_for_plot[f"{k}_pred"] = pred_arr
    for k, gt_arr in gt_body_feats.items():
        body_features_for_plot[f"{k}_gt"] = gt_arr

    run_viser_visualization(
        model_xml,
        original_qpos,
        predicted_qpos,
        fps=fps,
        body_vel_series=body_vel_series,
        body_features=body_features_for_plot,
    )


if __name__ == "__main__":
    main()
