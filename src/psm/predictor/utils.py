import glob
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial.transform import Rotation as R


def _yaw_raw_and_unwrapped(root_quat_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yaw_raw = R.from_quat(root_quat_wxyz, scalar_first=True).as_euler("xyz")[:, 2]
    return yaw_raw, np.unwrap(yaw_raw)


def compute_trajectory_features(
    qpos: np.ndarray,
    horizons_frames: tuple[int, ...] = (8, 16, 24),
    *,
    traj_yaw_frame_deltas: bool = False,
) -> np.ndarray:
    """Future trajectory in current root yaw frame (pos/dir/yaw per horizon)."""
    qpos = np.asarray(qpos, dtype=np.float64)
    T = qpos.shape[0]
    root_pos = qpos[:, 0:3]
    root_quat = qpos[:, 3:7]
    yaw_r, yaw_u = _yaw_raw_and_unwrapped(root_quat)
    tidx = np.arange(T, dtype=np.int64)
    parts: list[np.ndarray] = []
    for nf in horizons_frames:
        end_idx = np.minimum(tidx + int(nf), T - 1)
        disp_w = root_pos[end_idx, :2] - root_pos[:, :2]
        cy = np.cos(yaw_r)
        sy = np.sin(yaw_r)
        pos_x = cy * disp_w[:, 0] + sy * disp_w[:, 1]
        pos_y = -sy * disp_w[:, 0] + cy * disp_w[:, 1]
        dyaw_tot = yaw_u[end_idx] - yaw_u[tidx]
        dir_x = np.cos(dyaw_tot)
        dir_y = np.sin(dyaw_tot)
        base = np.stack([pos_x, pos_y, dir_x, dir_y], axis=1)
        if traj_yaw_frame_deltas:
            cols: list[np.ndarray] = []
            for i in range(int(nf)):
                i0 = np.minimum(tidx + i, T - 1)
                i1 = np.minimum(tidx + i + 1, T - 1)
                cols.append((yaw_u[i1] - yaw_u[i0])[:, None])
            dyaw_blk = np.concatenate(cols, axis=1) if cols else np.zeros((T, 0))
            parts.append(np.concatenate([base, dyaw_blk], axis=1))
        else:
            parts.append(np.concatenate([base, dyaw_tot[:, None]], axis=1))
    return np.concatenate(parts, axis=1)


def _finite_diff_velocity_np(x: np.ndarray, fps: float) -> np.ndarray:
    """Per-row velocity from positions (T, C), same scheme as lower joint vel in NPZ loading."""
    x = np.asarray(x, dtype=np.float64)
    dt = 1.0 / max(float(fps), 1e-6)
    v = np.empty_like(x)
    v[0] = (x[1] - x[0]) / dt
    v[-1] = (x[-1] - x[-2]) / dt
    if x.shape[0] > 2:
        v[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    return v


def _symmetry_project_mean_std(
    mean: torch.Tensor,
    std: torch.Tensor,
    names: list[str],
    symmetry_spec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project channel stats onto mirror-consistent constraints.

    For a pair (left, right, sign):
      mean[left] = sign * mean[right], std[left] = std[right].
    For center antisymmetric channels (name == mirror_name, sign == -1):
      mean[name] = 0.
    """
    if symmetry_spec is None:
        return mean, std

    m = mean.clone()
    s = std.clone()
    name_to_idx = {n: i for i, n in enumerate(names)}
    touched: set[int] = set()

    for left_name, right_name, sign in symmetry_spec:
        if left_name not in name_to_idx or right_name not in name_to_idx:
            continue
        li = name_to_idx[left_name]
        ri = name_to_idx[right_name]
        sg = float(sign)

        # Center channel (no swap).
        if li == ri:
            if sg < 0:
                m[li] = 0.0
            continue

        if li in touched or ri in touched:
            continue
        touched.add(li)
        touched.add(ri)

        m_li = m[li]
        m_ri = m[ri]
        pair_mean = 0.5 * (m_li + sg * m_ri)
        m[li] = pair_mean
        m[ri] = sg * pair_mean

        pair_std = 0.5 * (s[li] + s[ri])
        s[li] = pair_std
        s[ri] = pair_std

    s[s == 0] = 1.0
    return m, s


def should_flip_cmd_feature_for_mirror(name: str) -> bool:
    """Whether a command channel negates under a sagittal (left/right) mirror.

    Uses explicit patterns so new trajectory keys are not flipped by accident
    (e.g. ``traj_*_x`` stays unchanged in the yaw frame).
    """
    if name in ("root_vy", "root_wz"):
        return True
    if "_dyaw_" in name:
        return True
    if name.startswith("traj_") and name.endswith("_yaw"):
        return True
    if name.startswith(("traj_pos_", "traj_dir_")) and name.endswith("_y"):
        return True
    return False


def make_cmd_flip_signs(
    cmd_feature_names: list[str], *, device, dtype=torch.float32
) -> torch.Tensor:
    """Per-channel +/-1 multiplier for mirroring command features in the root yaw frame."""
    vals = [-1.0 if should_flip_cmd_feature_for_mirror(n) else 1.0 for n in cmd_feature_names]
    return torch.tensor(vals, device=device, dtype=dtype)


def make_body_vel_flip_signs(
    body_vel_names: list[str], *, device, dtype=torch.float32
) -> torch.Tensor:
    """Per-channel +/-1 for mirroring root velocity history under a sagittal mirror."""
    vals = [-1.0 if n in ("root_vy", "root_wz") else 1.0 for n in body_vel_names]
    return torch.tensor(vals, device=device, dtype=dtype)


def _validate_mirror_lists(indices: list[int], signs: list[float]) -> None:
    """Ensure mirror map is self-inverse: applying the mirror twice recovers the input."""
    for i in range(len(indices)):
        j = indices[i]
        if indices[j] != i:
            raise ValueError(
                f"Mirror map is not an involution: idx[{i}]={j}, idx[{j}]={indices[j]} "
                "(check SYMMETRY_SPEC for this joint/feature list)"
            )
        if abs(signs[i] * signs[j] - 1.0) > 1e-5:
            raise ValueError(
                f"Mirror signs not self-inverse at index {i}: "
                f"signs[{i}]={signs[i]}, signs[{j}]={signs[j]}"
            )


def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    """Wrap angles (rad) to the interval [-pi, pi] in a vectorized way."""
    angles = np.asarray(angles, dtype=np.float64)
    return (angles + np.pi) % (2 * np.pi) - np.pi


def quaternions_to_rpy(quats_wxyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized conversion from scalar-first quaternion (w,x,y,z) to roll, pitch, yaw [rad]."""
    quats_wxyz = np.asarray(quats_wxyz, dtype=np.float64)
    if quats_wxyz.shape[-1] != 4:
        raise ValueError(f"Expected quaternions (...,4), got {quats_wxyz.shape}")
    r = R.from_quat(quats_wxyz, scalar_first=True)
    euler = r.as_euler("xyz", degrees=False)
    return euler[..., 0], euler[..., 1], wrap_to_pi(euler[..., 2])


def _smooth_series(series: np.ndarray, window: int = 21, poly: int = 3) -> np.ndarray:
    """Apply gentle temporal smoothing (Savitzky-Golay) to a 1D series."""
    series = np.asarray(series, dtype=np.float64)
    n = series.shape[0]
    if n < 5:
        return series
    # Ensure odd window and not longer than the sequence.
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < 5:
        return series
    try:
        return savgol_filter(series, window_length=window, polyorder=min(poly, window - 2))
    except Exception:
        return series


def _smooth_angle_series(angles: np.ndarray, window: int = 21, poly: int = 3) -> np.ndarray:
    """Smooth wrapped angular series by unwrapping, filtering, then re-wrapping."""
    angles = np.asarray(angles, dtype=np.float64)
    unwrapped = np.unwrap(angles)
    smoothed = _smooth_series(unwrapped, window=window, poly=poly)
    return wrap_to_pi(smoothed)


def _postprocess_body_features(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Smooth/clamp features to reduce spikes and keep physical ranges."""
    out: dict[str, np.ndarray] = {}
    for name, arr in features.items():
        a = np.asarray(arr, dtype=np.float64)
        if name in ("left_foot_rel_yaw", "right_foot_rel_yaw"):
            a = _smooth_angle_series(a, window=21, poly=2)
        else:
            a = _smooth_series(a, window=21, poly=2)

        # Non-negative/boxed features.
        if name in ("step_length", "step_width", "cadence_hz", "root_height"):
            a = np.clip(a, 0.0, None)
        if name == "double_support_factor":
            a = np.clip(a, 0.0, 1.0)
        if name in ("pelvis_roll", "pelvis_pitch", "torso_roll", "torso_pitch"):
            a = np.clip(a, -np.pi, np.pi)

        out[name] = a
    return out


def _estimate_foot_contact_from_kinematics(
    foot_pos_r: np.ndarray, dt: float
) -> np.ndarray:
    """Heuristic contact estimate from root-relative foot trajectory.

    Contact is true when the foot is low and vertical velocity is small.
    """
    z = np.asarray(foot_pos_r[:, 2], dtype=np.float64)
    vz = np.gradient(z, dt)
    z_thr = float(np.percentile(z, 25.0) + 0.02)
    return (z <= z_thr) & (np.abs(vz) < 0.35)


def _compute_cadence_and_double_support(
    left_contact: np.ndarray,
    right_contact: np.ndarray,
    dt: float,
    window: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute windowed cadence (Hz) and double-support fraction.

    Cadence counts contact onsets (heel-strike-like events) for both feet.
    """
    T = int(left_contact.shape[0])
    cadence = np.zeros(T, dtype=np.float64)
    double_support = np.zeros(T, dtype=np.float64)
    both = left_contact & right_contact

    for t in range(T):
        s = max(0, t - window + 1)
        l_win = left_contact[s : t + 1]
        r_win = right_contact[s : t + 1]
        b_win = both[s : t + 1]
        wlen = max(1, t - s + 1)

        l_edges = np.sum((~l_win[:-1]) & l_win[1:]) if wlen > 1 else 0
        r_edges = np.sum((~r_win[:-1]) & r_win[1:]) if wlen > 1 else 0
        cadence[t] = float(l_edges + r_edges) / (wlen * dt)
        double_support[t] = float(np.mean(b_win.astype(np.float64)))

    cadence = _smooth_series(cadence, window=21, poly=2)
    double_support = _smooth_series(double_support, window=21, poly=2)
    double_support = np.clip(double_support, 0.0, 1.0)
    return cadence, double_support


def _swing_peak_height_trace(foot_z_w: np.ndarray, in_contact: np.ndarray) -> np.ndarray:
    """Per-timestep swing peak tracker (resets on contact)."""
    z = np.asarray(foot_z_w, dtype=np.float64)
    c = np.asarray(in_contact, dtype=bool)
    out = np.zeros_like(z)
    peak = float(z[0]) if z.size > 0 else 0.0
    for i in range(z.shape[0]):
        zi = float(z[i])
        if c[i]:
            peak = zi
        else:
            peak = max(peak, zi)
        out[i] = peak
    return out


def _swing_peak_height_from_step_peaks(
    step_length_raw: np.ndarray,
    left_z_w: np.ndarray,
    right_z_w: np.ndarray,
    *,
    peak_distance: int = 10,
    smooth_window: int = 31,
    smooth_poly: int = 2,
) -> np.ndarray:
    """Step-synchronous swing clearance from step-length peaks.

    For each detected step-length peak, compute the max of both feet world-z
    over that step interval (midpoint between neighboring peaks), then interpolate
    and smooth to obtain a dense per-frame target.
    """
    step = np.asarray(step_length_raw, dtype=np.float64)
    lz = np.asarray(left_z_w, dtype=np.float64)
    rz = np.asarray(right_z_w, dtype=np.float64)
    n = int(step.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    x_full = np.arange(n, dtype=np.float64)
    peaks, _ = find_peaks(step, distance=int(peak_distance))
    if peaks.size < 2:
        peaks = np.array([0, n - 1], dtype=np.int64)
    peaks = np.unique(np.clip(peaks.astype(np.int64), 0, n - 1))
    if peaks.size < 2:
        peaks = np.array([0, n - 1], dtype=np.int64)

    mids = np.rint((peaks[:-1] + peaks[1:]) / 2.0).astype(np.int64)
    starts = np.empty_like(peaks)
    ends = np.empty_like(peaks)
    starts[0] = 0
    starts[1:] = mids
    ends[:-1] = mids
    ends[-1] = n

    peak_vals = np.zeros(peaks.shape[0], dtype=np.float64)
    for i in range(peaks.shape[0]):
        s = int(max(0, starts[i]))
        e = int(min(n, max(starts[i] + 1, ends[i])))
        seg = np.maximum(lz[s:e], rz[s:e])
        peak_vals[i] = float(np.max(seg)) if seg.size > 0 else 0.0

    clearance = np.interp(x_full, peaks.astype(np.float64), peak_vals)
    clearance = _smooth_series(clearance, window=smooth_window, poly=smooth_poly)
    return np.clip(clearance, 0.0, None)


def compute_body_vel_from_qpos_qvel(
    qpos: np.ndarray, qvel: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (root_vx, root_vy, root_wz) in yaw-only root frame from qpos/qvel."""
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)

    root_quat = qpos[:, 3:7]  # wxyz
    root_lin_vel_w = qvel[:, 0:3]
    root_ang_vel_w = qvel[:, 3:6]

    root_rot_full = R.from_quat(root_quat, scalar_first=True)
    root_euler = root_rot_full.as_euler("xyz", degrees=False)  # (T, 3)
    # Zero roll/pitch, keep yaw.
    root_euler_only_yaw = np.zeros_like(root_euler)
    root_euler_only_yaw[:, 2] = root_euler[:, 2]
    yaw_only_rot = R.from_euler("xyz", root_euler_only_yaw, degrees=False)

    root_lin_vel_yaw = yaw_only_rot.inv().apply(root_lin_vel_w)
    vx_local = root_lin_vel_yaw[:, 0]
    vy_local = root_lin_vel_yaw[:, 1]
    wz_local = root_ang_vel_w[:, 2]
    return vx_local, vy_local, wz_local


def compute_body_vel_vx_wz_from_qpos_qvel(
    qpos: np.ndarray, qvel: np.ndarray
) -> np.ndarray:
  """Root linear vx and angular wz in **body** frame (legacy PKL pipeline).

  Returns:
      (T, 2) with columns ``[root_vx, root_wz]``.
  """
  qpos = np.asarray(qpos, dtype=np.float64)
  qvel = np.asarray(qvel, dtype=np.float64)

  root_quat = qpos[:, 3:7]  # wxyz
  root_lin_vel_w = qvel[:, 0:3]

  root_rot = R.from_quat(root_quat, scalar_first=True)
  lin_b = root_rot.inv().apply(root_lin_vel_w)
  ang_b = qvel[:, 3:6].copy()
  return np.stack([lin_b[:, 0], ang_b[:, 2]], axis=1)


def compute_local_6d_vel_from_qpos_qvel(
    qpos: np.ndarray, qvel: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute full local-frame linear/angular velocity (xyz, xyz) from qpos/qvel.

    Returns:
        lin_local: (T, 3) root linear velocity in full local/root frame
        ang_local: (T, 3) root angular velocity in full local/root frame
    """
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)

    root_quat = qpos[:, 3:7]  # wxyz
    root_lin_vel_w = qvel[:, 0:3]

    root_rot_full = R.from_quat(root_quat, scalar_first=True)
    lin_local = root_rot_full.inv().apply(root_lin_vel_w)
    ang_local = qvel[:, 3:6]
    return lin_local, ang_local


def mirror_lower_sequence_batch(
    lower_seq: torch.Tensor,
    lower_indices: torch.Tensor,
    lower_signs: torch.Tensor,
) -> torch.Tensor:
  """Apply left/right mirroring to lower joint history (B, H, n_lower)."""
  return lower_seq[:, :, lower_indices] * lower_signs.view(1, 1, -1)


def get_mirror_indices_scales(
    lower_joint_names,
    upper_joint_names,
    body_feature_names,
    history_horizon,
    prediction_horizon,
    device,
    symmetry_spec=None,
):
    """
    Return scales and index permutations for batch mirroring
    Args:
        lower_joint_names: List of lower body joint names
        upper_joint_names: List of upper body joint names
        history_horizon: Number of history timesteps
        prediction_horizon: Number of prediction timesteps
        device: device name
        symmetry_spec: List of tuples (left_name, right_name, sign) defining mirroring

    Returns:
        output_indices_exp, output_signs_exp, lower_indices (per-timestep joint perm), lower_signs
    """

    # Create mirroring maps for each joint type
    def create_mirror_map(joint_names):
        n_joints = len(joint_names)
        name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
        indices = list(range(n_joints))
        signs = [1.0] * n_joints

        if symmetry_spec is not None:
            for left_name, right_name, sign in symmetry_spec:
                if left_name in name_to_idx and right_name in name_to_idx:
                    li, ri = name_to_idx[left_name], name_to_idx[right_name]
                    indices[li], indices[ri] = ri, li
                    signs[li] = signs[ri] = float(sign)

        _validate_mirror_lists(indices, signs)
        return torch.tensor(indices, device=device), torch.tensor(signs, device=device)

    # Get mirroring maps
    lower_indices, lower_signs = create_mirror_map(lower_joint_names)
    upper_indices, upper_signs = create_mirror_map(upper_joint_names)
    body_indices, body_signs = create_mirror_map(body_feature_names)

    # Output vector: [upper joints over P] + [body features over P]
    upper_indices_exp = torch.cat(
        [upper_indices + i * len(upper_joint_names) for i in range(prediction_horizon)]
    )
    upper_signs_exp = upper_signs.repeat(prediction_horizon)

    body_indices_pred_exp = torch.cat(
        [body_indices + i * len(body_feature_names) for i in range(prediction_horizon)]
    )
    body_signs_pred_exp = body_signs.repeat(prediction_horizon)

    upper_pred_size = len(upper_joint_names) * prediction_horizon
    output_indices_exp = torch.cat(
        [upper_indices_exp, body_indices_pred_exp + upper_pred_size]
    )
    output_signs_exp = torch.cat([upper_signs_exp, body_signs_pred_exp])

    return output_indices_exp, output_signs_exp, lower_indices, lower_signs


def _validate_psm_joint_config(
    stored_lower: list[str],
    stored_upper: list[str],
    lower_joint_names: list[str],
    upper_joint_names: list[str],
    source: str,
) -> None:
    if stored_lower != lower_joint_names or stored_upper != upper_joint_names:
        raise ValueError(
            f"PSM NPZ joint lists in {source!r} do not match config. "
            f"Re-run psm-csv-to-npz or psm-augment-npz with current config, or update config."
        )


def _psm_arrays_to_training_dict(
    *,
    lower_joints: np.ndarray,
    lower_joint_vel: np.ndarray,
    upper_joints: np.ndarray,
    foot_pos_hist: np.ndarray,
    foot_vel_hist: np.ndarray,
    body_vel: np.ndarray,
    cmd_features: np.ndarray,
    body_features_arr: np.ndarray,
    body_feature_names: list[str],
    cmd_feature_names: list[str],
    cmd_traj_horizons: tuple[int, ...],
    cmd_traj_yaw_frame_deltas: bool,
    motion_offsets: list[int],
    motion_lengths: list[int],
    upper_joint_names: list[str],
    lower_joint_names: list[str],
    motion_files: list[str],
    root_height_mean: float,
) -> dict:
    body_features = {
        name: body_features_arr[:, i].astype(np.float64)
        for i, name in enumerate(body_feature_names)
    }
    return {
        "lower_joints": lower_joints.astype(np.float64),
        "lower_joint_vel": lower_joint_vel.astype(np.float64),
        "upper_joints": upper_joints.astype(np.float64),
        "foot_pos_hist": foot_pos_hist.astype(np.float64),
        "foot_vel_hist": foot_vel_hist.astype(np.float64),
        "motion_offsets": motion_offsets,
        "motion_lengths": motion_lengths,
        "upper_joint_names": upper_joint_names,
        "lower_joint_names": lower_joint_names,
        "body_features": body_features,
        "body_feature_names": body_feature_names,
        "body_vel": body_vel.astype(np.float64),
        "body_vel_names": ["root_vx", "root_vy", "root_wz"],
        "cmd_features": cmd_features.astype(np.float64),
        "cmd_feature_names": cmd_feature_names,
        "cmd_traj_horizons": cmd_traj_horizons,
        "cmd_traj_yaw_frame_deltas": cmd_traj_yaw_frame_deltas,
        "motion_files": motion_files,
        "root_height_mean": root_height_mean,
    }


def _load_psm_training_bundle(
    bundle_path: str,
    *,
    upper_joint_names: list[str],
    lower_joint_names: list[str],
) -> dict:
    from psm.predictor.npz_schema import bundle_has_psm_training

    data = np.load(bundle_path, allow_pickle=True)
    if not bundle_has_psm_training(data):
        raise ValueError(f"{bundle_path} is not a PSM training bundle")

    stored_lower = [str(x) for x in data["psm_lower_joint_names"].tolist()]
    stored_upper = [str(x) for x in data["psm_upper_joint_names"].tolist()]
    _validate_psm_joint_config(
        stored_lower, stored_upper, lower_joint_names, upper_joint_names, bundle_path
    )

    body_feature_names = [str(x) for x in data["psm_body_feature_names"].tolist()]
    cmd_feature_names = [str(x) for x in data["psm_cmd_feature_names"].tolist()]
    horizons = tuple(int(x) for x in np.asarray(data["psm_cmd_traj_horizons"]).tolist())
    yaw_deltas = bool(int(np.asarray(data["psm_cmd_traj_yaw_frame_deltas"]).reshape(-1)[0]))

    root_h = float(np.mean(data["psm_body_features"][:, body_feature_names.index("root_height")]))
    sources = [str(x) for x in data["segment_source"].tolist()]

    return _psm_arrays_to_training_dict(
        lower_joints=data["psm_lower_joints"],
        lower_joint_vel=data["psm_lower_joint_vel"],
        upper_joints=data["psm_upper_joints"],
        foot_pos_hist=data["psm_foot_pos_hist"],
        foot_vel_hist=data["psm_foot_vel_hist"],
        body_vel=data["psm_body_vel"],
        cmd_features=data["psm_cmd_features"],
        body_features_arr=data["psm_body_features"],
        body_feature_names=body_feature_names,
        cmd_feature_names=cmd_feature_names,
        cmd_traj_horizons=horizons,
        cmd_traj_yaw_frame_deltas=yaw_deltas,
        motion_offsets=[int(x) for x in data["segment_start_idx"].tolist()],
        motion_lengths=[int(x) for x in data["segment_length"].tolist()],
        upper_joint_names=upper_joint_names,
        lower_joint_names=lower_joint_names,
        motion_files=sources,
        root_height_mean=root_h,
    )


def _load_psm_clips_precomputed(
    motion_files: list[str],
    *,
    upper_joint_names: list[str],
    lower_joint_names: list[str],
) -> dict:
    from psm.predictor.npz_schema import clip_has_psm_training

    parts: dict[str, list[np.ndarray]] = {
        "psm_lower_joints": [],
        "psm_lower_joint_vel": [],
        "psm_upper_joints": [],
        "psm_foot_pos_hist": [],
        "psm_foot_vel_hist": [],
        "psm_body_vel": [],
        "psm_cmd_features": [],
        "psm_body_features": [],
    }
    motion_offsets: list[int] = []
    motion_lengths: list[int] = []
    offset = 0
    meta: dict | None = None
    root_heights: list[np.ndarray] = []

    for path in motion_files:
        data = np.load(path, allow_pickle=True)
        if not clip_has_psm_training(data):
            raise ValueError(f"{path} lacks PSM keys; run psm-augment-npz first")
        if meta is None:
            meta = {
                "body_feature_names": [str(x) for x in data["psm_body_feature_names"].tolist()],
                "cmd_feature_names": [str(x) for x in data["psm_cmd_feature_names"].tolist()],
                "horizons": tuple(int(x) for x in np.asarray(data["psm_cmd_traj_horizons"]).tolist()),
                "yaw_deltas": bool(
                    int(np.asarray(data["psm_cmd_traj_yaw_frame_deltas"]).reshape(-1)[0])
                ),
                "stored_lower": [str(x) for x in data["psm_lower_joint_names"].tolist()],
                "stored_upper": [str(x) for x in data["psm_upper_joint_names"].tolist()],
            }
            _validate_psm_joint_config(
                meta["stored_lower"],
                meta["stored_upper"],
                lower_joint_names,
                upper_joint_names,
                path,
            )
        length = int(data["psm_lower_joints"].shape[0])
        motion_offsets.append(offset)
        motion_lengths.append(length)
        offset += length
        for key in parts:
            parts[key].append(np.asarray(data[key]))
        rh_idx = meta["body_feature_names"].index("root_height")
        root_heights.append(np.asarray(data["psm_body_features"][:, rh_idx], dtype=np.float64))

    assert meta is not None
    body_features_arr = np.concatenate(parts["psm_body_features"], axis=0)
    return _psm_arrays_to_training_dict(
        lower_joints=np.concatenate(parts["psm_lower_joints"], axis=0),
        lower_joint_vel=np.concatenate(parts["psm_lower_joint_vel"], axis=0),
        upper_joints=np.concatenate(parts["psm_upper_joints"], axis=0),
        foot_pos_hist=np.concatenate(parts["psm_foot_pos_hist"], axis=0),
        foot_vel_hist=np.concatenate(parts["psm_foot_vel_hist"], axis=0),
        body_vel=np.concatenate(parts["psm_body_vel"], axis=0),
        cmd_features=np.concatenate(parts["psm_cmd_features"], axis=0),
        body_features_arr=body_features_arr,
        body_feature_names=meta["body_feature_names"],
        cmd_feature_names=meta["cmd_feature_names"],
        cmd_traj_horizons=meta["horizons"],
        cmd_traj_yaw_frame_deltas=meta["yaw_deltas"],
        motion_offsets=motion_offsets,
        motion_lengths=motion_lengths,
        upper_joint_names=upper_joint_names,
        lower_joint_names=lower_joint_names,
        motion_files=motion_files,
        root_height_mean=float(np.mean(np.concatenate(root_heights))),
    )


def load_motion_data_npz(
    motion_files_pattern: str,
    *,
    upper_joint_names: list[str],
    lower_joint_names: list[str],
    feet_bodies: Tuple[str, str],
    cmd_traj_horizons: tuple[int, ...] = (8, 16, 24),
    cmd_traj_yaw_frame_deltas: bool = False,
    body_names: list[str] | None = None,
    root_body_name: str = "pelvis",
):
    """Load and process motion data from NPZ for predictor training.

    Accepts the full schema (``qpos``, ``body_pos_r``, …) and the compact per-clip export
    (``joint_pos``, ``body_*_w`` only). Missing fields are derived at load time
    (see ``psm.predictor.npz_schema``).

    Minimum NPZ keys (compact export):
    - joint_names, joint_pos, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
    - optional: joint_vel, fps, robot

    Fast path: merged ``motions.npz`` or per-clip files with ``psm_*`` keys (from
    ``psm-csv-to-npz`` / ``psm-augment-npz`` + ``psm-stack-motions``).
    """
    from psm.predictor.features import compute_predictor_features_from_config
    from psm.predictor.npz_schema import (
        bundle_has_psm_training,
        clip_has_psm_training,
        expand_motion_npz,
    )

    motion_files = sorted(glob.glob(motion_files_pattern))
    if motion_files_pattern.endswith(".npz") and not motion_files:
        # Explicit single-file path (not a glob).
        candidate = motion_files_pattern
        if Path(candidate).is_file():
            motion_files = [candidate]

    if not lower_joint_names:
        raise ValueError("lower_joint_names must be non-empty.")
    if not motion_files:
        raise FileNotFoundError(f"No motion files matched pattern: {motion_files_pattern!r}")

    if len(motion_files) == 1:
        npz = np.load(motion_files[0], allow_pickle=True)
        is_bundle = bundle_has_psm_training(npz)
        npz.close()
        if is_bundle:
            print(f"[INFO] Fast load: PSM training bundle {motion_files[0]}")
            return _load_psm_training_bundle(
                motion_files[0],
                upper_joint_names=upper_joint_names,
                lower_joint_names=lower_joint_names,
            )

    sample = np.load(motion_files[0], allow_pickle=True)
    try:
        all_precomputed = clip_has_psm_training(sample)
    finally:
        sample.close()
    if all_precomputed:
        for path in motion_files[1:]:
            check = np.load(path, allow_pickle=True)
            try:
                if not clip_has_psm_training(check):
                    all_precomputed = False
                    break
            finally:
                check.close()
    if all_precomputed:
        print(f"[INFO] Fast load: {len(motion_files)} precomputed PSM clip(s)")
        return _load_psm_clips_precomputed(
            motion_files,
            upper_joint_names=upper_joint_names,
            lower_joint_names=lower_joint_names,
        )

    print("[INFO] Slow load: computing features from kinematics (consider psm-augment-npz)")

    parts: dict[str, list[np.ndarray]] = {
        "psm_lower_joints": [],
        "psm_lower_joint_vel": [],
        "psm_upper_joints": [],
        "psm_foot_pos_hist": [],
        "psm_foot_vel_hist": [],
        "psm_body_vel": [],
        "psm_cmd_features": [],
        "psm_body_features": [],
    }
    motion_offsets: list[int] = []
    motion_lengths: list[int] = []
    offset = 0
    meta: dict | None = None
    root_heights: list[np.ndarray] = []

    for motion_path in motion_files:
        npz = np.load(motion_path, allow_pickle=True)
        try:
            motion = expand_motion_npz(
                npz,
                body_names=body_names,
                root_body_name=root_body_name,
            )
        finally:
            npz.close()

        psm = compute_predictor_features_from_config(motion)
        if meta is None:
            meta = {
                "body_feature_names": list(psm["psm_body_feature_names"]),
                "cmd_feature_names": list(psm["psm_cmd_feature_names"]),
                "horizons": tuple(int(x) for x in np.asarray(psm["psm_cmd_traj_horizons"]).tolist()),
                "yaw_deltas": bool(
                    int(np.asarray(psm["psm_cmd_traj_yaw_frame_deltas"]).reshape(-1)[0])
                ),
            }
        length = int(psm["psm_lower_joints"].shape[0])
        motion_offsets.append(offset)
        motion_lengths.append(length)
        offset += length
        for key in parts:
            parts[key].append(np.asarray(psm[key]))
        rh_idx = meta["body_feature_names"].index("root_height")
        root_heights.append(np.asarray(psm["psm_body_features"][:, rh_idx], dtype=np.float64))

    assert meta is not None
    body_features_arr = np.concatenate(parts["psm_body_features"], axis=0)
    return _psm_arrays_to_training_dict(
        lower_joints=np.concatenate(parts["psm_lower_joints"], axis=0),
        lower_joint_vel=np.concatenate(parts["psm_lower_joint_vel"], axis=0),
        upper_joints=np.concatenate(parts["psm_upper_joints"], axis=0),
        foot_pos_hist=np.concatenate(parts["psm_foot_pos_hist"], axis=0),
        foot_vel_hist=np.concatenate(parts["psm_foot_vel_hist"], axis=0),
        body_vel=np.concatenate(parts["psm_body_vel"], axis=0),
        cmd_features=np.concatenate(parts["psm_cmd_features"], axis=0),
        body_features_arr=body_features_arr,
        body_feature_names=meta["body_feature_names"],
        cmd_feature_names=meta["cmd_feature_names"],
        cmd_traj_horizons=meta["horizons"],
        cmd_traj_yaw_frame_deltas=meta["yaw_deltas"],
        motion_offsets=motion_offsets,
        motion_lengths=motion_lengths,
        upper_joint_names=upper_joint_names,
        lower_joint_names=lower_joint_names,
        motion_files=motion_files,
        root_height_mean=float(np.mean(np.concatenate(root_heights))),
    )


def prepare_training_data(
    data,
    history_horizon,
    prediction_horizon,
    device="cuda",
    *,
    use_lower_joint_velocity: bool = True,
    use_foot_velocity: bool = False,
    symmetry_spec=None,
):
    """Prepare training data by converting to tensors and computing normalization parameters."""
    lower_joints = torch.from_numpy(data["lower_joints"]).float().to(device=device)
    lower_joint_vel = torch.from_numpy(data["lower_joint_vel"]).float().to(device=device)
    upper_joints = torch.from_numpy(data["upper_joints"]).float().to(device=device)
    foot_pos_hist = torch.from_numpy(np.asarray(data["foot_pos_hist"])).float().to(device=device)
    if "foot_vel_hist" in data:
        foot_vel_hist = torch.from_numpy(np.asarray(data["foot_vel_hist"])).float().to(device=device)
    else:
        foot_vel_hist = torch.from_numpy(
            _finite_diff_velocity_np(np.asarray(data["foot_pos_hist"], dtype=np.float64), 50.0)
        ).float().to(device=device)
    motion_offsets = torch.tensor(data["motion_offsets"], dtype=torch.int64).to(device=device)
    motion_lengths = torch.tensor(data["motion_lengths"], dtype=torch.int64).to(device=device)

    # Body feature matrix
    body_feature_names = list(data["body_features"].keys())
    body_features_list = [
        torch.from_numpy(np.asarray(data["body_features"][k])).float().to(device=device)
        for k in body_feature_names
    ]
    body_features = torch.stack(body_features_list, dim=1)  # (T, F)
    num_body_features = int(body_features.shape[1])

    # Body velocity (root_vx, root_vy, root_wz) in root body frame.
    body_vel = torch.from_numpy(np.asarray(data["body_vel"])).float().to(device=device)
    body_vel_names = list(data.get("body_vel_names", ["root_vx", "root_vy", "root_wz"]))
    cmd_features = torch.from_numpy(np.asarray(data["cmd_features"])).float().to(device=device)
    cmd_feature_names = list(data.get("cmd_feature_names", ["root_vx", "root_vy", "root_wz"]))

    # Filter motions too short
    min_required = history_horizon + prediction_horizon
    valid = [i for i, L in enumerate(data["motion_lengths"]) if L >= min_required]
    if len(valid) < len(data["motion_lengths"]):
        motion_offsets = motion_offsets[valid]
        motion_lengths = motion_lengths[valid]

    # Per-channel normalization (applied inside the model; leg uses same stats at each timestep).
    leg_pos_mean = lower_joints.mean(dim=0)
    leg_pos_std = lower_joints.std(dim=0)
    leg_pos_std[leg_pos_std == 0] = 1
    leg_pos_mean, leg_pos_std = _symmetry_project_mean_std(
        leg_pos_mean,
        leg_pos_std,
        list(data["lower_joint_names"]),
        symmetry_spec,
    )

    leg_vel_mean = lower_joint_vel.mean(dim=0)
    leg_vel_std = lower_joint_vel.std(dim=0)
    leg_vel_std[leg_vel_std == 0] = 1
    leg_vel_mean, leg_vel_std = _symmetry_project_mean_std(
        leg_vel_mean,
        leg_vel_std,
        list(data["lower_joint_names"]),
        symmetry_spec,
    )
    foot_pos_mean = foot_pos_hist.mean(dim=0)
    foot_pos_std = foot_pos_hist.std(dim=0)
    foot_pos_std[foot_pos_std == 0] = 1
    # Sagittal mirror: x/z keep sign, y flips, and feet are swapped (L<->R).
    if foot_pos_mean.numel() == 6:
        l = foot_pos_mean[0:3].clone()
        r = foot_pos_mean[3:6].clone()
        foot_pos_mean[0] = 0.5 * (l[0] + r[0])
        foot_pos_mean[1] = 0.0
        foot_pos_mean[2] = 0.5 * (l[2] + r[2])
        foot_pos_mean[3] = 0.5 * (r[0] + l[0])
        foot_pos_mean[4] = 0.0
        foot_pos_mean[5] = 0.5 * (r[2] + l[2])
        # Match std across L/R feet so normalize(mirror(x)) ~= mirror(normalize(x)).
        sx = 0.5 * (foot_pos_std[0] + foot_pos_std[3])
        sy = 0.5 * (foot_pos_std[1] + foot_pos_std[4])
        sz = 0.5 * (foot_pos_std[2] + foot_pos_std[5])
        foot_pos_std[0] = foot_pos_std[3] = torch.clamp(sx, min=1e-8)
        foot_pos_std[1] = foot_pos_std[4] = torch.clamp(sy, min=1e-8)
        foot_pos_std[2] = foot_pos_std[5] = torch.clamp(sz, min=1e-8)

    foot_vel_mean = foot_vel_hist.mean(dim=0)
    foot_vel_std = foot_vel_hist.std(dim=0)
    foot_vel_std[foot_vel_std == 0] = 1
    if foot_vel_mean.numel() == 6:
        l = foot_vel_mean[0:3].clone()
        r = foot_vel_mean[3:6].clone()
        foot_vel_mean[0] = 0.5 * (l[0] + r[0])
        foot_vel_mean[1] = 0.0
        foot_vel_mean[2] = 0.5 * (l[2] + r[2])
        foot_vel_mean[3] = 0.5 * (r[0] + l[0])
        foot_vel_mean[4] = 0.0
        foot_vel_mean[5] = 0.5 * (r[2] + l[2])
        sx = 0.5 * (foot_vel_std[0] + foot_vel_std[3])
        sy = 0.5 * (foot_vel_std[1] + foot_vel_std[4])
        sz = 0.5 * (foot_vel_std[2] + foot_vel_std[5])
        foot_vel_std[0] = foot_vel_std[3] = torch.clamp(sx, min=1e-8)
        foot_vel_std[1] = foot_vel_std[4] = torch.clamp(sy, min=1e-8)
        foot_vel_std[2] = foot_vel_std[5] = torch.clamp(sz, min=1e-8)

    body_vel_mean = body_vel.mean(dim=0)
    body_vel_std = body_vel.std(dim=0)
    body_vel_std[body_vel_std == 0] = 1
    # Mirror in sagittal plane keeps vx and flips vy/wz; enforce zero-mean lateral/yaw channels.
    body_vel_name_to_idx = {n: i for i, n in enumerate(body_vel_names)}
    for n in ("root_vy", "root_wz"):
        if n in body_vel_name_to_idx:
            body_vel_mean[body_vel_name_to_idx[n]] = 0.0
    cmd_mean = cmd_features.mean(dim=0)
    cmd_std = cmd_features.std(dim=0)
    cmd_std[cmd_std == 0] = 1
    # Mirror constraints for command channels (zero-mean on negated channels).
    for i, name in enumerate(cmd_feature_names):
        if should_flip_cmd_feature_for_mirror(name):
            cmd_mean[i] = 0.0

    # Output normalization: [upper_future, body_features_future]
    upper_mean = upper_joints.mean(dim=0)
    upper_std = upper_joints.std(dim=0)
    upper_std[upper_std == 0] = 1
    upper_mean, upper_std = _symmetry_project_mean_std(
        upper_mean,
        upper_std,
        list(data["upper_joint_names"]),
        symmetry_spec,
    )
    upper_y_mean = upper_mean.repeat(prediction_horizon)
    upper_y_std = upper_std.repeat(prediction_horizon)
    upper_y_std[upper_y_std == 0] = 1

    body_mean = body_features.mean(dim=0)
    body_std = body_features.std(dim=0)
    body_std[body_std == 0] = 1
    body_mean, body_std = _symmetry_project_mean_std(
        body_mean,
        body_std,
        body_feature_names,
        symmetry_spec,
    )
    body_y_mean = body_mean.repeat(prediction_horizon)
    body_y_std = body_std.repeat(prediction_horizon)
    body_y_std[body_y_std == 0] = 1

    y_mean = torch.cat([upper_y_mean, body_y_mean])
    y_std = torch.cat([upper_y_std, body_y_std])

    return {
        "lower_joints": lower_joints,
        "lower_joint_vel": lower_joint_vel,
        "upper_joints": upper_joints,
        "foot_pos_hist": foot_pos_hist,
        "body_features": body_features,
        "body_feature_names": body_feature_names,
        "num_body_features": num_body_features,
        "body_vel": body_vel,
        "body_vel_names": body_vel_names,
        "cmd_features": cmd_features,
        "cmd_feature_names": cmd_feature_names,
        "cmd_traj_horizons": tuple(data.get("cmd_traj_horizons", (8, 16, 24))),
        "cmd_traj_yaw_frame_deltas": bool(data.get("cmd_traj_yaw_frame_deltas", False)),
        "cmd_mean": cmd_mean,
        "cmd_std": cmd_std,
        "motion_offsets": motion_offsets,
        "motion_lengths": motion_lengths,
        "leg_pos_mean": leg_pos_mean,
        "leg_pos_std": leg_pos_std,
        "leg_vel_mean": leg_vel_mean if use_lower_joint_velocity else None,
        "leg_vel_std": leg_vel_std if use_lower_joint_velocity else None,
        "foot_pos_mean": foot_pos_mean,
        "foot_pos_std": foot_pos_std,
        "foot_vel_mean": foot_vel_mean if use_foot_velocity else None,
        "foot_vel_std": foot_vel_std if use_foot_velocity else None,
        "body_vel_mean": body_vel_mean,
        "body_vel_std": body_vel_std,
        "use_lower_joint_velocity": use_lower_joint_velocity,
        "use_foot_velocity": use_foot_velocity,
        "foot_vel_hist": foot_vel_hist,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def create_loss_weights(
    upper_joint_names,
    body_feature_names,
    prediction_horizon,
    loss_weight_config,
    device="cuda",
):
    """Create a per-output-feature weight vector for weighted MSE loss."""
    import fnmatch

    all_feature_names = []
    for _ in range(prediction_horizon):
        all_feature_names.extend(upper_joint_names)
    for _ in range(prediction_horizon):
        all_feature_names.extend(body_feature_names)

    weights = [1.0] * len(all_feature_names)
    for idx, feature_name in enumerate(all_feature_names):
        for pattern, w in loss_weight_config.items():
            if fnmatch.fnmatch(feature_name, pattern):
                weights[idx] = float(w)
                break
    return torch.tensor(weights, dtype=torch.float32, device=device)


def create_feature_regularization_weights(
    upper_joint_names,
    body_feature_names,
    prediction_horizon,
    regularization_config,
    device,
):
    """Create a per-output-feature L2 regularization weight vector (0 = no regularization)."""
    from fnmatch import fnmatch

    all_feature_names = []
    for t in range(prediction_horizon):
        for name in upper_joint_names:
            all_feature_names.append(f"{name}_t{t}")
    for t in range(prediction_horizon):
        for name in body_feature_names:
            all_feature_names.append(f"{name}_t{t}")

    reg = torch.zeros(len(all_feature_names), device=device)
    for pattern, weight in regularization_config.items():
        for i, fname in enumerate(all_feature_names):
            base = "_".join(fname.split("_")[:-1]) if fname.split("_")[-1].startswith("t") else fname
            if fnmatch(base, pattern) or fnmatch(fname, pattern):
                reg[i] = float(weight)
    return reg


def save_metadata(metadata, filepath="metadata.pkl"):
    """Save metadata to pickle file (for constructor params with tensors)."""
    with open(filepath, "wb") as f:
        pickle.dump(metadata, f)


def save_config_yaml(config_data, filepath="config.yaml"):
    """Save readable configuration data to YAML file."""
    with open(filepath, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)


def load_metadata(filepath="metadata.pkl"):
    """Load metadata from pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_config_yaml(filepath="config.yaml"):
    """Load configuration from YAML file."""
    with open(filepath, "r") as f:
        return yaml.safe_load(f)
