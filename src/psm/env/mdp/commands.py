from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from .velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
)

from psm.predictor.psm_predictor import PsmPredictor
from psm.env.weights.resolve import effective_data_path

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class PsmVelocityCommandCfg(UniformVelocityCommandCfg):
  """Velocity command with integrated PSM predictor."""

  predictor_path: str
  command_name: str = "twist"
  feet_contact_sensor_name: str = "feet_ground_contact"
  warmup_history_steps: int | None = None

  # Ghost visualization parameters (used by debug viewers).
  ghost_upper_prediction: bool = True
  ghost_alpha: float = 0.4
  show_command_trajectory: bool = True
  command_trajectory_scale: float = 1.0
  command_trajectory_z_offset: float = 0.03
  command_trajectory_max_intermediate: int = 4

  def build(self, env: ManagerBasedRlEnv) -> "PsmVelocityCommand":
    return PsmVelocityCommand(self, env)


class PsmVelocityCommand(UniformVelocityCommand):
  """Uniform velocity command that also owns the PSM predictor and its state."""

  cfg: PsmVelocityCommandCfg

  predictor: PsmPredictor
  horizon_h: int
  horizon_p: int
  _pred_horizon_k: int
  n_vel_hist: int

  lower_order: list[str]
  upper_order: list[str]
  body_feature_names: list[str]
  body_vel_hist_names: list[str]
  body_vel_future_names: list[str]
  cmd_feature_names: list[str]
  cmd_traj_horizons: tuple[int, ...]
  cmd_traj_yaw_frame_deltas: bool
  history_input_mode: str

  lower_indices: list[int]
  upper_indices: list[int]
  _predictor_use_lower_hist: bool
  _predictor_use_feet_hist: bool
  _predictor_use_lower_joint_vel: bool
  _predictor_use_foot_vel: bool
  left_foot_body_idx: int
  right_foot_body_idx: int

  pred_upper_targets: torch.Tensor
  pred_body_feature_targets: torch.Tensor

  _upper_joint_q_adr_np: np.ndarray
  _twist_cmd_index_map: dict[str, int]

  _lower_joint_limits_min: torch.Tensor
  _lower_joint_limits_max: torch.Tensor
  _upper_joint_limits_min: torch.Tensor
  _upper_joint_limits_max: torch.Tensor

  _body_feature_name_to_index: dict[str, int]

  _ghost_model: Any | None
  _ghost_qpos_cache: np.ndarray | None

  _warmup_steps: torch.Tensor

  def __init__(self, cfg: PsmVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._env = env
    # Per-env predictor diagnostics, mirroring tracking's MotionCommand.metrics.
    self.metrics["upper_joint_rmse"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["feet_pitch_rmse"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["feet_yaw_rmse"] = torch.zeros(self.num_envs, device=self.device)
    self._ghost_model = None
    self._ghost_qpos_cache = None
    self._psm_predictor_init_from_cfg()

  # --------------------------------------------------------------------------- #
  # Initialisation from predictor metadata
  # --------------------------------------------------------------------------- #

  def _psm_predictor_init_from_cfg(self) -> None:
    import os
    import pickle

    device = self.device
    robot = self._env.scene[self.cfg.entity_name]

    pred_root, used_log_snap = effective_data_path(self.cfg.predictor_path)
    if used_log_snap:
      self.cfg.predictor_path = pred_root

    meta_path = os.path.join(pred_root, "metadata.pkl")
    model_path = os.path.join(pred_root, "predictor.pth")
    if not os.path.exists(meta_path):
      raise FileNotFoundError(f"Predictor metadata not found: {meta_path}")
    if not os.path.exists(model_path):
      raise FileNotFoundError(f"Predictor model not found: {model_path}")

    with open(meta_path, "rb") as f:
      metadata = pickle.load(f)

    self.horizon_h = int(metadata["horizon"])
    self.lower_order = list(metadata["lower_order"])
    self.upper_order = list(metadata["upper_order"])
    self.body_feature_names = list(metadata.get("body_feature_names", []))

    self.body_vel_hist_names = list(
      metadata.get(
        "body_vel_names",
        metadata.get("body_vel_hist_names", ["vx_local", "wz_local"]),
      )
    )
    self.body_vel_future_names = list(
      metadata.get("body_vel_future_names", ["root_vx", "root_vy", "root_wz"])
    )
    self.cmd_feature_names = list(
      metadata.get("cmd_feature_names", self.body_vel_future_names)
    )
    self.cmd_traj_horizons = tuple(
      int(h) for h in metadata.get("cmd_traj_horizons", (8, 16, 24))
    )
    self.cmd_traj_yaw_frame_deltas = bool(
      metadata.get("cmd_traj_yaw_frame_deltas", False)
    )
    self.n_vel_hist = len(self.body_vel_hist_names)
    cp = metadata["constructor_params"]
    _him = str(metadata.get("history_input_mode", cp.get("history_input_mode", "both"))).lower()
    if _him not in ("joints", "feet", "both"):
      raise ValueError(
        f"Predictor history_input_mode must be joints|feet|both, got {_him!r}"
      )
    self.history_input_mode = _him
    self._predictor_use_lower_hist = _him in ("joints", "both")
    self._predictor_use_feet_hist = _him in ("feet", "both")
    self._twist_cmd_index_map = {
      "lin_vel_x_local": 0,
      "lin_vel_y_local": 1,
      "ang_vel_z_local": 2,
      "vx_local": 0,
      "vy_local": 1,
      "wz_local": 2,
      "root_vx": 0,
      "root_vy": 1,
      "root_wz": 2,
    }

    self.lower_indices, self.upper_indices = self._resolve_joint_indices_from_metadata()
    self._upper_joint_q_adr_np = (
      robot.indexing.joint_q_adr[self.upper_indices].detach().cpu().numpy()
    )

    if "prediction_h" in cp:
      self.horizon_p = int(cp["prediction_h"])
    else:
      out_sz = int(cp["output_size"])
      n_out = len(self.upper_order) + len(self.body_feature_names)
      self.horizon_p = out_sz // max(1, n_out)
    self._pred_horizon_k = 0

    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    ctor_allowed = {
      "output_size",
      "y_mean",
      "y_std",
      "leg_pos_mean",
      "leg_pos_std",
      "body_vel_mean",
      "body_vel_std",
      "foot_pos_mean",
      "foot_pos_std",
      "num_lower",
      "history_horizon",
      "prediction_horizon",
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
      "cmd_mean",
      "cmd_std",
      "cmd_feature_dim",
    }
    ctor_full = dict(metadata["constructor_params"])
    if "use_lower_joint_velocity" not in ctor_full and "use_lower_velocity" in ctor_full:
      ctor_full["use_lower_joint_velocity"] = bool(ctor_full["use_lower_velocity"])
    if "use_lower_joint_velocity" not in ctor_full:
      ctor_full["use_lower_joint_velocity"] = False
    if "use_foot_velocity" not in ctor_full:
      ctor_full["use_foot_velocity"] = False
    if "foot_vel_mean" not in ctor_full:
      ctor_full["foot_vel_mean"] = None
    if "foot_vel_std" not in ctor_full:
      ctor_full["foot_vel_std"] = None
    ctor = {k: v for k, v in ctor_full.items() if k in ctor_allowed}
    self.predictor = PsmPredictor(**ctor).to(device)
    self.predictor.load_state_dict(state_dict)
    self.predictor.eval()
    self._predictor_use_lower_joint_vel = bool(self.predictor.use_lower_joint_velocity)
    self._predictor_use_foot_vel = bool(self.predictor.use_foot_velocity)

    feet_bodies = metadata.get("feet_bodies", None)
    if not feet_bodies or len(feet_bodies) != 2:
      raise KeyError(
        "Predictor metadata must contain `feet_bodies = [left_body, right_body]`."
      )
    left_body, right_body = feet_bodies
    self.left_foot_body_idx = robot.body_names.index(left_body)
    self.right_foot_body_idx = robot.body_names.index(right_body)
    self._ghost_qpos_cache = np.zeros(
      (self.num_envs, int(self._env.sim.mj_model.nq)), dtype=np.float32
    )

    n_lower = len(self.lower_indices)
    n_upper = len(self.upper_indices)
    n_body = len(self.body_feature_names)

    self.lower_buffer = torch.zeros(
      (self.num_envs, self.horizon_h, n_lower), device=device, dtype=torch.float32
    )
    self.lower_vel_buffer: torch.Tensor | None = None
    if self._predictor_use_lower_joint_vel:
      self.lower_vel_buffer = torch.zeros(
        (self.num_envs, self.horizon_h, n_lower), device=device, dtype=torch.float32
      )
    self.body_vel_hist_buffer = torch.zeros(
      (self.num_envs, self.horizon_h, self.n_vel_hist), device=device, dtype=torch.float32
    )
    self.foot_pos_hist_buffer = torch.zeros(
      (self.num_envs, self.horizon_h, 6), device=device, dtype=torch.float32
    )
    self.pred_upper_targets = torch.zeros(
      (self.num_envs, n_upper), device=device, dtype=torch.float32
    )
    self.pred_body_feature_targets = torch.zeros(
      (self.num_envs, n_body), device=device, dtype=torch.float32
    )

    self._step_length_inst = torch.zeros(
      self.num_envs, device=device, dtype=torch.float32
    )
    self._step_length_smooth = torch.zeros(
      self.num_envs, device=device, dtype=torch.float32
    )
    self._body_feature_name_to_index = {
      name: i for i, name in enumerate(self.body_feature_names)
    }
    self._setup_joint_limits()

    self._warmup_steps = torch.zeros(
      self.num_envs, device=device, dtype=torch.float32
    )

  def _resolve_joint_indices_from_metadata(self) -> tuple[list[int], list[int]]:
    robot = self._env.scene["robot"]
    robot_joint_names = list(robot.joint_names)
    robot_name_to_index = {name: i for i, name in enumerate(robot_joint_names)}

    if len(set(self.lower_order)) != len(self.lower_order):
      raise ValueError("Predictor metadata `lower_order` contains duplicate joint names.")
    if len(set(self.upper_order)) != len(self.upper_order):
      raise ValueError("Predictor metadata `upper_order` contains duplicate joint names.")

    missing_lower = [n for n in self.lower_order if n not in robot_name_to_index]
    missing_upper = [n for n in self.upper_order if n not in robot_name_to_index]
    if missing_lower or missing_upper:
      raise KeyError(
        "Predictor joint names not found in robot model. "
        f"Missing lower: {missing_lower}. Missing upper: {missing_upper}."
      )

    lower_indices = [robot_name_to_index[n] for n in self.lower_order]
    upper_indices = [robot_name_to_index[n] for n in self.upper_order]

    resolved_lower = [robot_joint_names[i] for i in lower_indices]
    resolved_upper = [robot_joint_names[i] for i in upper_indices]
    if resolved_lower != self.lower_order or resolved_upper != self.upper_order:
      raise ValueError(
        "Predictor joint order mismatch between metadata and environment mapping."
      )

    return lower_indices, upper_indices

  def _setup_joint_limits(self) -> None:
    lower_n = len(self.lower_order)
    upper_n = len(self.upper_order)
    self._lower_joint_limits_min = torch.full((lower_n,), -float("inf"), device=self.device)
    self._lower_joint_limits_max = torch.full((lower_n,), float("inf"), device=self.device)
    self._upper_joint_limits_min = torch.full((upper_n,), -float("inf"), device=self.device)
    self._upper_joint_limits_max = torch.full((upper_n,), float("inf"), device=self.device)

    joint_limits = getattr(self._env.cfg, "joint_limits", None)
    if not isinstance(joint_limits, dict):
      return
    for i, name in enumerate(self.lower_order):
      if name in joint_limits:
        lo, hi = joint_limits[name]
        self._lower_joint_limits_min[i] = float(lo)
        self._lower_joint_limits_max[i] = float(hi)
    for i, name in enumerate(self.upper_order):
      if name in joint_limits:
        lo, hi = joint_limits[name]
        self._upper_joint_limits_min[i] = float(lo)
        self._upper_joint_limits_max[i] = float(hi)

  def _clip_lower_joints(self, x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, self._lower_joint_limits_min, self._lower_joint_limits_max)

  def _clip_upper_joints(self, x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, self._upper_joint_limits_min, self._upper_joint_limits_max)

  # --------------------------------------------------------------------------- #
  # Exposed helpers used by rewards / metrics
  # --------------------------------------------------------------------------- #

  @property
  def predictor_ready_mask(self) -> torch.Tensor:
    """Float mask [B] that is 1.0 for envs whose predictor has warmed up."""
    horizon = self.horizon_h
    if horizon <= 0:
      return torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
    return (self._warmup_steps >= float(horizon)).float()

  def get_upper_joint_target(self) -> torch.Tensor:
    return self.pred_upper_targets

  def get_upper_joint_indices(self) -> list[int]:
    return self.upper_indices

  def get_upper_joint_names(self) -> list[str]:
    return list(self.upper_order)

  def get_body_feature_targets(self) -> torch.Tensor:
    return self.pred_body_feature_targets

  def get_body_feature_index(self, feature_name: str) -> int:
    return self._body_feature_name_to_index.get(feature_name, -1)

  def _get_body_feature_target(self, feature_name: str) -> torch.Tensor | None:
    idx = self._body_feature_name_to_index.get(feature_name, -1)
    if idx < 0:
      return None
    return self.pred_body_feature_targets[:, idx]

  def get_step_length_target(self) -> torch.Tensor | None:
    return self._get_body_feature_target("step_length")

  def get_step_width_target(self) -> torch.Tensor | None:
    return self._get_body_feature_target("step_width")

  def get_step_length_instant(self) -> torch.Tensor:
    """Fore-aft foot separation in root frame (updated every sim step)."""
    return self._step_length_inst

  def get_step_length_smooth(self) -> torch.Tensor:
    """EMA of `get_step_length_instant`; closer to offline peak-smoothed step_length labels."""
    return self._step_length_smooth

  def get_cadence_hz_target(self) -> torch.Tensor | None:
    return self._get_body_feature_target("cadence_hz")

  def get_double_support_fraction_target(self) -> torch.Tensor | None:
    return self._get_body_feature_target("double_support_factor")

  def get_swing_peak_height_target(self) -> torch.Tensor | None:
    return self._get_body_feature_target("swing_peak_height")

  def get_root_height_target(self) -> torch.Tensor | None:
    return self._get_body_feature_target("root_height")

  def get_pelvis_roll_pitch_target(self) -> torch.Tensor | None:
    roll = self._get_body_feature_target("pelvis_roll")
    pitch = self._get_body_feature_target("pelvis_pitch")
    if roll is None or pitch is None:
      return None
    return torch.stack([roll, pitch], dim=1)

  def get_torso_roll_pitch_target(self) -> torch.Tensor | None:
    roll = self._get_body_feature_target("torso_roll")
    pitch = self._get_body_feature_target("torso_pitch")
    if roll is None or pitch is None:
      return None
    return torch.stack([roll, pitch], dim=1)

  def get_cadence_hz_contact_estimate(self) -> torch.Tensor:
    """Disabled for now: cadence feature is not used in the current model."""
    return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

  def get_double_support_contact_estimate(self) -> torch.Tensor:
    """Disabled for now: double-support feature is not used in the current model."""
    return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

  def get_contact_phase_sin_cos(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Phase outputs are disabled for this command variant."""
    z = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
    o = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
    return z, o

  def get_gait_phase_sin_cos_actual(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Phase outputs are disabled for this command variant."""
    z = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
    o = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
    return z, o

  def get_feet_pitch_targets(self) -> torch.Tensor | None:
    l_idx = self._body_feature_name_to_index.get("left_foot_pitch", -1)
    r_idx = self._body_feature_name_to_index.get("right_foot_pitch", -1)
    if l_idx < 0 and r_idx < 0:
      return None
    left = (
      self.pred_body_feature_targets[:, l_idx]
      if l_idx >= 0
      else torch.zeros(self.num_envs, device=self.device)
    )
    right = (
      self.pred_body_feature_targets[:, r_idx]
      if r_idx >= 0
      else torch.zeros(self.num_envs, device=self.device)
    )
    return torch.stack([left, right], dim=1)

  def get_feet_relative_yaw_targets(self) -> torch.Tensor | None:
    l_idx = self._body_feature_name_to_index.get("left_foot_rel_yaw", -1)
    r_idx = self._body_feature_name_to_index.get("right_foot_rel_yaw", -1)
    if l_idx < 0 and r_idx < 0:
      return None
    left = (
      self.pred_body_feature_targets[:, l_idx]
      if l_idx >= 0
      else torch.zeros(self.num_envs, device=self.device)
    )
    right = (
      self.pred_body_feature_targets[:, r_idx]
      if r_idx >= 0
      else torch.zeros(self.num_envs, device=self.device)
    )
    return torch.stack([left, right], dim=1)

  # --------------------------------------------------------------------------- #
  # Internal feature computation and predictor update
  # --------------------------------------------------------------------------- #

  def _compute_body_vel_hist_features(self) -> torch.Tensor:
    robot = self._env.scene["robot"]
    lin_b = robot.data.root_link_lin_vel_b
    ang_b = robot.data.root_link_ang_vel_b
    vel_map = {
      "lin_vel_x_local": lin_b[:, 0],
      "lin_vel_y_local": lin_b[:, 1],
      "lin_vel_z_local": lin_b[:, 2],
      "vx_local": lin_b[:, 0],
      "vy_local": lin_b[:, 1],
      "vz_local": lin_b[:, 2],
      "ang_vel_x_local": ang_b[:, 0],
      "ang_vel_y_local": ang_b[:, 1],
      "ang_vel_z_local": ang_b[:, 2],
      "wx_local": ang_b[:, 0],
      "wy_local": ang_b[:, 1],
      "wz_local": ang_b[:, 2],
      "root_vx": lin_b[:, 0],
      "root_vy": lin_b[:, 1],
      "root_wz": ang_b[:, 2],
    }
    unknown = [n for n in self.body_vel_hist_names if n not in vel_map]
    if unknown:
      raise KeyError(
        f"Unsupported `body_vel_hist_names` in predictor metadata: {unknown}"
      )
    return torch.stack([vel_map[n] for n in self.body_vel_hist_names], dim=1)

  def _integrate_cmd_traj_features(
    self,
    vx: torch.Tensor,
    vy: torch.Tensor,
    wz: torch.Tensor,
  ) -> torch.Tensor:
    dt = float(self._env.step_dt)
    B = vx.shape[0]
    if len(self.cmd_traj_horizons) == 0:
      return torch.zeros((B, 0), device=self.device, dtype=torch.float32)
    outs: list[torch.Tensor] = []
    for h in self.cmd_traj_horizons:
      h_i = int(h)
      if h_i <= 0:
        base = torch.zeros((B, 4), device=self.device, dtype=torch.float32)
        outs.append(base if self.cmd_traj_yaw_frame_deltas else torch.cat([base, torch.zeros((B, 1), device=self.device, dtype=torch.float32)], dim=1))
        continue
      t = float(h_i) * dt
      dyaw_tot = wz * t
      sin_t = torch.sin(dyaw_tot)
      cos_t = torch.cos(dyaw_tot)
      # Closed-form SE(2) integration for constant body twist over horizon t.
      wz_safe = torch.where(torch.abs(wz) < 1.0e-6, torch.ones_like(wz), wz)
      px_turn = (vx * sin_t + vy * (cos_t - 1.0)) / wz_safe
      py_turn = (vx * (1.0 - cos_t) + vy * sin_t) / wz_safe
      px_lin = vx * t
      py_lin = vy * t
      small_w = torch.abs(wz) < 1.0e-6
      px = torch.where(small_w, px_lin, px_turn)
      py = torch.where(small_w, py_lin, py_turn)
      base = torch.stack([px, py, torch.cos(dyaw_tot), torch.sin(dyaw_tot)], dim=1)
      if self.cmd_traj_yaw_frame_deltas:
        dyaw = (wz * dt).unsqueeze(1).repeat(1, h_i)
        outs.append(torch.cat([base, dyaw], dim=1))
      else:
        outs.append(torch.cat([base, dyaw_tot.unsqueeze(1)], dim=1))
    return torch.cat(outs, dim=1)

  def _extract_command_features(self) -> torch.Tensor:
    cmd = self._env.command_manager.get_command(self.cfg.command_name)
    if cmd is None:
      raise KeyError(
        f"Command '{self.cfg.command_name}' not found in command_manager."
      )
    # New checkpoints: explicit command features with trajectory block.
    if self.cmd_feature_names and any(n.startswith("traj_") for n in self.cmd_feature_names):
      vx = cmd[:, self._twist_cmd_index_map["root_vx"]]
      vy = cmd[:, self._twist_cmd_index_map["root_vy"]]
      wz = cmd[:, self._twist_cmd_index_map["root_wz"]]
      traj = self._integrate_cmd_traj_features(vx, vy, wz)
      full = torch.cat([vx.unsqueeze(1), vy.unsqueeze(1), wz.unsqueeze(1), traj], dim=1)
      if len(self.cmd_feature_names) != int(full.shape[1]):
        raise ValueError(
          f"Command feature size mismatch: metadata has {len(self.cmd_feature_names)} names, runtime built {int(full.shape[1])} features."
        )
      return full

    # Old checkpoints: stack listed command channels directly.
    names = self.body_vel_future_names
    unknown = [n for n in names if n not in self._twist_cmd_index_map]
    if unknown:
      raise KeyError(
        f"Unsupported `body_vel_future_names` in predictor metadata: {unknown}"
      )
    return torch.stack([cmd[:, self._twist_cmd_index_map[n]] for n in names], dim=1)

  def _command_traj_frame_stops(self) -> list[int]:
    # Match loco_gen "reference trajectory points": only configured horizons.
    return sorted({int(h) for h in self.cmd_traj_horizons if int(h) > 0})

  def _dense_traj_frame_stops(self) -> list[int]:
    hs = self._command_traj_frame_stops()
    if not hs:
      return []
    n_mid = max(0, int(self.cfg.command_trajectory_max_intermediate))
    out: list[int] = []
    prev = 0
    for h in hs:
      if h <= prev:
        continue
      if n_mid > 0 and h > prev + 1:
        raw = np.linspace(float(prev), float(h), n_mid + 2, dtype=np.float64)[1:-1]
        for x in raw:
          k = int(round(float(x)))
          k = max(prev + 1, min(h - 1, k))
          if not out or out[-1] != k:
            out.append(k)
      if not out or out[-1] != h:
        out.append(h)
      prev = h
    return out

  def _integrate_cmd_world_traj(
    self,
    *,
    root_pos_w: np.ndarray,
    root_yaw: float,
    vx: float,
    vy: float,
    wz: float,
    frame_stops: list[int],
    dt: float,
    z_off: float,
    scale: float,
  ) -> tuple[np.ndarray, np.ndarray]:
    if not frame_stops:
      return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    max_n = max(frame_stops)
    target = set(frame_stops)
    px_w = 0.0
    py_w = 0.0
    psi = float(root_yaw)
    pts: list[list[float]] = []
    yaws: list[float] = []
    for step in range(1, max_n + 1):
      c, s = np.cos(psi), np.sin(psi)
      px_w += (c * vx - s * vy) * dt * scale
      py_w += (s * vx + c * vy) * dt * scale
      psi += wz * dt
      if step in target:
        pts.append([
          float(root_pos_w[0] + px_w),
          float(root_pos_w[1] + py_w),
          float(root_pos_w[2] + z_off),
        ])
        yaws.append(float(psi))
    return np.asarray(pts, dtype=np.float64), np.asarray(yaws, dtype=np.float64)

  def _compute_foot_pos_hist_features(self) -> torch.Tensor:
    robot = self._env.scene["robot"]
    body_pos_w = robot.data.body_link_pos_w
    root_pos_w = robot.data.root_link_pos_w
    root_quat_w = robot.data.root_link_quat_w
    left_pos_w = body_pos_w[:, self.left_foot_body_idx, :]
    right_pos_w = body_pos_w[:, self.right_foot_body_idx, :]
    left_pos_r = quat_apply_inverse(root_quat_w, left_pos_w - root_pos_w)
    right_pos_r = quat_apply_inverse(root_quat_w, right_pos_w - root_pos_w)
    return torch.cat([left_pos_r, right_pos_r], dim=1)

  def _foot_vel_from_pos_hist(self, fp: torch.Tensor) -> torch.Tensor:
    """Finite-difference foot velocities in root frame (matches offline ``foot_vel_hist``)."""
    dt = float(self._env.step_dt)
    if dt <= 0:
      dt = 1.0 / 200.0
    fv = torch.zeros_like(fp)
    H = int(fp.shape[1])
    if H <= 1:
      return fv
    fv[:, 0] = (fp[:, 1] - fp[:, 0]) / dt
    fv[:, -1] = (fp[:, -1] - fp[:, -2]) / dt
    if H > 2:
      fv[:, 1:-1] = (fp[:, 2:] - fp[:, :-2]) / (2.0 * dt)
    return fv

  def _update_history_buffers(
    self,
    *,
    current_lower: torch.Tensor,
    current_lower_vel: torch.Tensor | None,
    current_foot_pos: torch.Tensor,
    current_vel_hist: torch.Tensor,
    env_ids: torch.Tensor | slice,
  ) -> None:
    self.lower_buffer[env_ids, :-1] = self.lower_buffer[env_ids, 1:].clone()
    self.lower_buffer[env_ids, -1] = self._clip_lower_joints(current_lower[env_ids]).clone()
    if self.lower_vel_buffer is not None and current_lower_vel is not None:
      self.lower_vel_buffer[env_ids, :-1] = self.lower_vel_buffer[env_ids, 1:].clone()
      self.lower_vel_buffer[env_ids, -1] = current_lower_vel[env_ids].clone()
    self.foot_pos_hist_buffer[env_ids, :-1] = self.foot_pos_hist_buffer[env_ids, 1:].clone()
    self.foot_pos_hist_buffer[env_ids, -1] = current_foot_pos[env_ids].clone()
    self.body_vel_hist_buffer[env_ids, :-1] = self.body_vel_hist_buffer[env_ids, 1:].clone()
    self.body_vel_hist_buffer[env_ids, -1] = current_vel_hist[env_ids].clone()

  def _write_prediction_targets(
    self, prediction: torch.Tensor, env_ids: torch.Tensor | slice
  ) -> None:
    n_upper = len(self.upper_indices)
    n_body = len(self.body_feature_names)
    k = self._pred_horizon_k
    upper_flat = prediction[:, : self.horizon_p * n_upper]
    body_flat = prediction[:, self.horizon_p * n_upper :] if n_body > 0 else None

    upper_pred = upper_flat.reshape(-1, self.horizon_p, n_upper)
    self.pred_upper_targets[env_ids] = self._clip_upper_joints(upper_pred[:, k, :])

    if n_body > 0 and body_flat is not None:
      body_pred = body_flat.reshape(-1, self.horizon_p, n_body)
      self.pred_body_feature_targets[env_ids] = body_pred[:, k, :]
    else:
      self.pred_body_feature_targets[env_ids].zero_()

  @torch.no_grad()
  def _update_predictor_targets(self, env_ids: torch.Tensor | slice | None = None) -> None:
    robot = self._env.scene["robot"]
    if env_ids is None:
      env_ids = slice(None)

    current_lower = robot.data.joint_pos[:, self.lower_indices].to(
      device=self.device, dtype=torch.float32
    )
    current_lower_vel = (
      robot.data.joint_vel[:, self.lower_indices].to(device=self.device, dtype=torch.float32)
      if self._predictor_use_lower_joint_vel
      else None
    )
    current_foot_pos = self._compute_foot_pos_hist_features()
    current_vel_hist = self._compute_body_vel_hist_features()

    self._update_history_buffers(
      current_lower=current_lower,
      current_lower_vel=current_lower_vel,
      current_foot_pos=current_foot_pos,
      current_vel_hist=current_vel_hist,
      env_ids=env_ids,
    )

    cmd = self._extract_command_features()
    cmd_sel = cmd[env_ids] if not isinstance(env_ids, slice) else cmd
    vel_future_input = cmd_sel
    lp = self.lower_buffer[env_ids] if self._predictor_use_lower_hist else None
    fp = self.foot_pos_hist_buffer[env_ids] if self._predictor_use_feet_hist else None
    bvh = self.body_vel_hist_buffer[env_ids]
    lv = self.lower_vel_buffer[env_ids] if self._predictor_use_lower_joint_vel else None
    fv = (
      self._foot_vel_from_pos_hist(fp)
      if self._predictor_use_foot_vel and fp is not None
      else None
    )
    prediction = self.predictor.predict(lp, fp, bvh, vel_future_input, lv, fv)
    self._write_prediction_targets(prediction, env_ids)

  def _fill_buffers_from_current_state(self, env_ids: torch.Tensor | slice) -> None:
    robot = self._env.scene["robot"]
    current_lower = self._clip_lower_joints(
      robot.data.joint_pos[:, self.lower_indices].to(device=self.device, dtype=torch.float32)
    )
    current_foot_pos = self._compute_foot_pos_hist_features()
    current_vel_hist = self._compute_body_vel_hist_features()

    self.lower_buffer[env_ids, :, :] = (
      current_lower[env_ids].unsqueeze(1).expand(-1, self.horizon_h, -1).clone()
    )
    self.foot_pos_hist_buffer[env_ids, :, :] = (
      current_foot_pos[env_ids].unsqueeze(1).expand(-1, self.horizon_h, -1).clone()
    )
    self.body_vel_hist_buffer[env_ids, :, :] = (
      current_vel_hist[env_ids].unsqueeze(1).expand(-1, self.horizon_h, -1).clone()
    )
    if self.lower_vel_buffer is not None:
      jv = robot.data.joint_vel[:, self.lower_indices].to(
        device=self.device, dtype=torch.float32
      )
      self.lower_vel_buffer[env_ids, :, :] = (
        jv[env_ids].unsqueeze(1).expand(-1, self.horizon_h, -1).clone()
      )
    self.pred_upper_targets[env_ids, :].zero_()
    self.pred_body_feature_targets[env_ids, :].zero_()
    self._step_length_inst[env_ids].zero_()
    self._step_length_smooth[env_ids].zero_()
    self._warmup_steps[env_ids].zero_()

  @torch.no_grad()
  def _update_step_metrics(self) -> None:
    robot = self._env.scene["robot"]
    body_pos_w = robot.data.body_link_pos_w
    root_pos_w = robot.data.root_link_pos_w
    root_quat_w = robot.data.root_link_quat_w

    left_pos_w = body_pos_w[:, self.left_foot_body_idx, :]
    right_pos_w = body_pos_w[:, self.right_foot_body_idx, :]
    left_pos_r = quat_apply_inverse(root_quat_w, left_pos_w - root_pos_w)
    right_pos_r = quat_apply_inverse(root_quat_w, right_pos_w - root_pos_w)

    step_length_inst = torch.abs(right_pos_r[:, 0] - left_pos_r[:, 0])
    self._step_length_inst = step_length_inst
    # EMA tracks slowly-varying separation; offline labels use peak interpolation + smoothing.
    ema_alpha = 0.05
    self._step_length_smooth = (1.0 - ema_alpha) * self._step_length_smooth + ema_alpha * step_length_inst

  # --------------------------------------------------------------------------- #
  # CommandTerm interface
  # --------------------------------------------------------------------------- #

  def _update_command(self) -> None:
    """Called by the command manager every env step."""
    # Preserve base velocity-command behaviour (heading control, world-frame logic).
    super()._update_command()
    # Then update predictor state on top of the new command.
    if self.num_envs > 0:
      self._warmup_steps += 1.0
    self._update_predictor_targets()
    self._update_step_metrics()
    self._update_metrics()

  def _update_metrics(self) -> None:
    """Update both base velocity metrics and predictor tracking diagnostics."""
    # First, keep original velocity-command metrics (error_vel_xy / error_vel_yaw).
    super()._update_metrics()

    robot = self._env.scene[self.cfg.entity_name]
    mask = self.predictor_ready_mask

    if mask.sum() == 0:
      self.metrics["upper_joint_rmse"].zero_()
      self.metrics["feet_pitch_rmse"].zero_()
      self.metrics["feet_yaw_rmse"].zero_()
      return

    upper_actual = robot.data.joint_pos[:, self.upper_indices].to(
      self.device, dtype=torch.float32
    )
    upper_pred = self.get_upper_joint_target().to(self.device, dtype=torch.float32)
    mse_upper = torch.mean((upper_actual - upper_pred) ** 2, dim=1)
    rmse_upper = torch.sqrt(mse_upper + 1e-9)
    self.metrics["upper_joint_rmse"][:] = rmse_upper * mask


    pitch_targets = self.get_feet_pitch_targets()
    if pitch_targets is not None:
      root_quat_w = robot.data.root_link_quat_w
      body_quat_w = robot.data.body_link_quat_w
      root_inv = quat_inv(root_quat_w)

      left_rel = quat_mul(root_inv, body_quat_w[:, self.left_foot_body_idx, :])
      _, left_pitch, _ = euler_xyz_from_quat(left_rel)
      right_rel = quat_mul(root_inv, body_quat_w[:, self.right_foot_body_idx, :])
      _, right_pitch, _ = euler_xyz_from_quat(right_rel)

      actual_pitch = torch.stack([left_pitch, right_pitch], dim=1)
      mse_pitch = torch.mean((actual_pitch - pitch_targets) ** 2, dim=1)
      rmse_pitch = torch.sqrt(mse_pitch + 1e-9)
      self.metrics["feet_pitch_rmse"][:] = rmse_pitch * mask
    else:
      self.metrics["feet_pitch_rmse"].zero_()

    yaw_targets = self.get_feet_relative_yaw_targets()
    if yaw_targets is not None:
      root_quat_w = robot.data.root_link_quat_w
      body_quat_w = robot.data.body_link_quat_w
      root_inv = quat_inv(root_quat_w)

      left_rel = quat_mul(root_inv, body_quat_w[:, self.left_foot_body_idx, :])
      right_rel = quat_mul(root_inv, body_quat_w[:, self.right_foot_body_idx, :])
      fwd = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=torch.float32).view(1, 3)
      left_fwd = quat_apply(left_rel, fwd.expand(self.num_envs, -1))
      right_fwd = quat_apply(right_rel, fwd.expand(self.num_envs, -1))
      actual_fwd_xy = torch.stack([left_fwd[:, :2], right_fwd[:, :2]], dim=1)
      actual_fwd_xy = torch.nn.functional.normalize(actual_fwd_xy, p=2, dim=-1, eps=1e-6)

      yaw_targets_wrapped = (yaw_targets + math.pi) % (2 * math.pi) - math.pi
      target_fwd_xy = torch.stack(
        [torch.cos(yaw_targets_wrapped), torch.sin(yaw_targets_wrapped)], dim=-1
      )
      target_fwd_xy = torch.nn.functional.normalize(target_fwd_xy, p=2, dim=-1, eps=1e-6)
      cross_z = actual_fwd_xy[..., 0] * target_fwd_xy[..., 1] - actual_fwd_xy[..., 1] * target_fwd_xy[..., 0]
      dot = (actual_fwd_xy * target_fwd_xy).sum(dim=-1)
      yaw_err = torch.atan2(cross_z, dot)
      mse_yaw = torch.mean(yaw_err ** 2, dim=1)
      rmse_yaw = torch.sqrt(mse_yaw + 1e-9)
      self.metrics["feet_yaw_rmse"][:] = rmse_yaw * mask
    else:
      self.metrics["feet_yaw_rmse"].zero_()

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Velocity arrows + predictor ghost + command trajectory frame overlay."""
    # First, keep the base velocity visualization (command vs actual velocities).
    super()._debug_vis_impl(visualizer)

    if not self.cfg.ghost_upper_prediction:
      return

    # Lazily construct a tinted ghost model, mirroring PsmEnv.
    if self._ghost_model is None:
      self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
      ghost_rgba = self._ghost_model.geom_rgba.copy()
      ghost_rgba[:, 0] = 1.0  # light red tint
      ghost_rgba[:, 1] *= 0.6
      ghost_rgba[:, 2] *= 0.6
      self._ghost_model.geom_rgba[:] = ghost_rgba

    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self._ghost_qpos_cache is None:
      self._ghost_qpos_cache = np.zeros(
        (self.num_envs, int(self._env.sim.mj_model.nq)), dtype=np.float32
      )

    cmd_all = self._env.command_manager.get_command(self.cfg.command_name)
    # Draw ghost and command trajectory for each selected env.
    for batch in env_indices:
      if batch < 0 or batch >= self.num_envs:
        continue
      qpos = self._ghost_qpos_cache[batch]
      np.copyto(qpos, self._env.sim.data.qpos[batch].detach().cpu().numpy())
      qpos[self._upper_joint_q_adr_np] = (
        self.pred_upper_targets[batch].detach().cpu().numpy()
      )
      visualizer.add_ghost_mesh(
        qpos=qpos,
        model=self._ghost_model,
        alpha=float(self.cfg.ghost_alpha),
      )
      if not self.cfg.show_command_trajectory or cmd_all is None:
        continue
      root_pos_w = self._env.scene["robot"].data.root_link_pos_w[batch].detach().cpu().numpy()
      root_quat_w = self._env.scene["robot"].data.root_link_quat_w[batch]
      _, _, root_yaw = euler_xyz_from_quat(root_quat_w.unsqueeze(0))
      yaw0 = float(root_yaw[0].detach().cpu().item())
      vx = float(cmd_all[batch, 0].detach().cpu().item())
      vy = float(cmd_all[batch, 1].detach().cpu().item())
      wz = float(cmd_all[batch, 2].detach().cpu().item())
      stops_dense = self._dense_traj_frame_stops()
      stops_ref = self._command_traj_frame_stops()
      pts_dense, yaws_dense = self._integrate_cmd_world_traj(
        root_pos_w=root_pos_w,
        root_yaw=yaw0,
        vx=vx,
        vy=vy,
        wz=wz,
        frame_stops=stops_dense,
        dt=float(self._env.step_dt),
        z_off=float(self.cfg.command_trajectory_z_offset),
        scale=float(self.cfg.command_trajectory_scale),
      )
      pts_ref, yaws_ref = self._integrate_cmd_world_traj(
        root_pos_w=root_pos_w,
        root_yaw=yaw0,
        vx=vx,
        vy=vy,
        wz=wz,
        frame_stops=stops_ref,
        dt=float(self._env.step_dt),
        z_off=float(self.cfg.command_trajectory_z_offset),
        scale=float(self.cfg.command_trajectory_scale),
      )
      if pts_dense.shape[0] == 0:
        continue
      p0 = np.array(
        [
          root_pos_w[0],
          root_pos_w[1],
          root_pos_w[2] + float(self.cfg.command_trajectory_z_offset),
        ],
        dtype=np.float64,
      )
      # Colors follow loco_gen.viz.overlays RefTrajOverlay:
      # arc=(158,118,116), heading shaft=(188,42,42), heading head=(205,52,52).
      arc_rgba = (158.0 / 255.0, 118.0 / 255.0, 116.0 / 255.0, 0.35)
      head_rgba = (205.0 / 255.0, 52.0 / 255.0, 52.0 / 255.0, 1.0)
      # Dense, transparent, thin path.
      prev = p0
      for i in range(pts_dense.shape[0]):
        cur = pts_dense[i]
        visualizer.add_arrow(prev, cur, color=arc_rgba, width=0.006)
        prev = cur
      # Sparse bold points + heading arrows at CMD_TRAJ_HORIZONS.
      for i in range(pts_ref.shape[0]):
        cur = pts_ref[i]
        # point proxy (short vertical tick)
        # tick_to = cur + np.array([0.0, 0.0, 0.015], dtype=np.float64)
        visualizer.add_sphere(cur, 0.02, color=head_rgba)
        head_len = 0.12 * float(self.cfg.command_trajectory_scale)
        hy = float(yaws_ref[i])
        hto = cur + np.array(
          [np.cos(hy) * head_len, np.sin(hy) * head_len, 0.0],
          dtype=np.float64,
        )
        visualizer.add_arrow(cur, hto, color=head_rgba, width=0.012)
      return

