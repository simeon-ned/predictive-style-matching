"""PSM predictor tracking rewards (upper body, gait targets from ``PsmVelocityCommand``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_apply_yaw,
  quat_inv,
  quat_mul,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values

from ..commands import PsmVelocityCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
def _get_predictor_command(env: "ManagerBasedRlEnv") -> PsmVelocityCommand:
  cmd = env.command_manager.get_term("twist")
  if cmd is None:
    raise KeyError("PSM velocity command 'twist' not found.")
  return cast(PsmVelocityCommand, cmd)


def _apply_tracking_kernel(
  err_sq_scalar: torch.Tensor,
  err_abs_scalar: torch.Tensor,
  kernel: str,
  std: float | None = None,
) -> torch.Tensor:
  if kernel == "exp":
    if std is None:
      raise ValueError("std must be provided when kernel='exp'.")
    return torch.exp(-err_sq_scalar / std**2)
  if kernel == "l1":
    return -err_abs_scalar
  raise ValueError(f"Unsupported tracking kernel '{kernel}'. Expected one of: ['exp', 'l1'].")


def _predictor_ready_mask(env: "ManagerBasedRlEnv") -> torch.Tensor:
  cmd = _get_predictor_command(env)
  return cmd.predictor_ready_mask.to(env.device, dtype=torch.float32)


def predictor_upper_joint_tracking(
  env: "ManagerBasedRlEnv",
  std: float = 0.25,
  kernel: str = "exp",
  scale_by_limits: bool = False,
  limit_range_eps: float = 1e-4,
  scales: dict[str, float] | None = None,
) -> torch.Tensor:
  mask = _predictor_ready_mask(env)
  if mask.sum() == 0:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  robot = env.scene["robot"]
  cmd = _get_predictor_command(env)
  joint_names = cmd.get_upper_joint_names()
  jids = cmd.get_upper_joint_indices()
  upper_actual = robot.data.joint_pos[:, jids].to(env.device, dtype=torch.float32)
  upper_pred = cmd.get_upper_joint_target().to(env.device, dtype=torch.float32)
  err = upper_actual - upper_pred
  n = len(jids)
  divisor = torch.ones(n, device=env.device, dtype=torch.float32)
  if scale_by_limits:
    sj = robot.data.soft_joint_pos_limits
    assert sj is not None
    divisor *= (sj[0, jids, 1] - sj[0, jids, 0]).clamp(min=limit_range_eps)
  if scales:
    idx_list, _, vals = resolve_matching_names_values(data=scales, list_of_strings=joint_names)
    for idx, v in zip(idx_list, vals):
      divisor[idx] *= max(float(v), limit_range_eps)
  err = err / divisor.unsqueeze(0)
  mse = torch.mean(err**2, dim=1)
  l1 = torch.mean(torch.abs(err), dim=1)
  return _apply_tracking_kernel(mse, l1, kernel=kernel, std=std) * mask


def _feet_local_xy_separation_from_predictor(
  env: "ManagerBasedRlEnv",
  cmd: PsmVelocityCommand,
) -> tuple[torch.Tensor, torch.Tensor]:
  robot = env.scene["robot"]
  body_pos_w = robot.data.body_link_pos_w
  root_pos_w = robot.data.root_link_pos_w
  root_quat_w = robot.data.root_link_quat_w
  left_pos_w = body_pos_w[:, cmd.left_foot_body_idx, :]
  right_pos_w = body_pos_w[:, cmd.right_foot_body_idx, :]
  left_pos_r = quat_apply_inverse(root_quat_w, left_pos_w - root_pos_w)
  right_pos_r = quat_apply_inverse(root_quat_w, right_pos_w - root_pos_w)
  dx = torch.abs(right_pos_r[:, 0] - left_pos_r[:, 0])
  dy = torch.abs(right_pos_r[:, 1] - left_pos_r[:, 1])
  return dx, dy


def step_width_matching(
  env: "ManagerBasedRlEnv",
  std: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  offset: float = 0.0,
  command_name: str | None = None,
  ang_vel_threshold: float = 0.1,
  kernel: str = "exp",
  method: str = "instant",
) -> torch.Tensor:
  """Track lateral foot separation vs predictor target.

  ``method``: ``instant``.
  """
  mask = _predictor_ready_mask(env)
  cmd = _get_predictor_command(env)
  del asset_cfg
  target_raw = cmd.get_step_width_target()
  if target_raw is None or mask.sum() == 0:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  target = torch.abs(target_raw + offset)
  if method == "instant":
    _, dy = _feet_local_xy_separation_from_predictor(env, cmd)
  else:
    raise ValueError(f"method must be 'instant'; got {method!r}.")
  err = dy - target
  base = _apply_tracking_kernel(err**2, torch.abs(err), kernel=kernel, std=std) * mask
  if command_name is None:
    out = base
  else:
    cmd_vel = env.command_manager.get_command(command_name)
    assert cmd_vel is not None, f"Command '{command_name}' not found."
    gate = (torch.abs(cmd_vel[:, 2]) <= ang_vel_threshold).float()
    out = base * gate
  return out


def step_length_matching(
  env: "ManagerBasedRlEnv",
  std: float = 0.25,
  min_target_length: float = 0.36,
  max_target_length: float = 0.45,
  command_name: str = "twist",
  vel_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str = "feet_ground_contact",
  last_step_length: list[torch.Tensor] = [],
) -> torch.Tensor:
  """Reward maintaining an adaptive step length between feet based on command speed."""
  
  
  mask = _predictor_ready_mask(env)
  cmd = _get_predictor_command(env)
  target_raw = cmd.get_step_length_target()
  if target_raw is None or mask.sum() == 0:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  target = target_raw.clamp(min=min_target_length, max=max_target_length)

  return step_length(
    env=env,
    std=std,
    target_length=target,
    command_name=command_name,
    vel_threshold=vel_threshold,
    asset_cfg=asset_cfg,
    sensor_name=sensor_name,
    last_step_length=last_step_length,
  )


def root_height_matching(
  env: "ManagerBasedRlEnv",
  std: float = 0.1,
  kernel: str = "exp",
) -> torch.Tensor:
  """Track root world-z against predictor target `root_height`."""
  mask = _predictor_ready_mask(env)
  cmd = _get_predictor_command(env)
  pred = cmd.get_root_height_target()
  if pred is None or mask.sum() == 0:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  robot = env.scene["robot"]
  actual = robot.data.root_link_pos_w[:, 2]
  err = actual - pred
  return _apply_tracking_kernel(err**2, torch.abs(err), kernel=kernel, std=std) * mask


def _roll_pitch_from_quat(quat_w: torch.Tensor) -> torch.Tensor:
  roll, pitch, _ = euler_xyz_from_quat(quat_w)
  return torch.stack([roll, pitch], dim=1)


def roll_pitch_matching(
  env: "ManagerBasedRlEnv",
  predicted_rp: torch.Tensor | None,
  actual_rp: torch.Tensor,
  std: float = 0.15,
  kernel: str = "exp",
) -> torch.Tensor:
  """Shared roll/pitch tracking term to avoid duplicated logic."""
  mask = _predictor_ready_mask(env)
  if predicted_rp is None or mask.sum() == 0:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  err = actual_rp - predicted_rp
  err_sq = torch.sum(err**2, dim=1)
  err_abs = torch.mean(torch.abs(err), dim=1)
  return _apply_tracking_kernel(err_sq, err_abs, kernel=kernel, std=std) * mask


def pelvis_roll_pitch_matching(
  env: "ManagerBasedRlEnv",
  std: float = 0.15,
  kernel: str = "exp",
  link_name: str = "pelvis",
) -> torch.Tensor:
  cmd = _get_predictor_command(env)
  pred = cmd.get_pelvis_roll_pitch_target()
  robot = env.scene["robot"]
  if link_name not in robot.body_names:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  pelvis_idx = robot.body_names.index(link_name)
  actual = _roll_pitch_from_quat(robot.data.body_link_quat_w[:, pelvis_idx, :])
  return roll_pitch_matching(env, predicted_rp=pred, actual_rp=actual, std=std, kernel=kernel)


def torso_roll_pitch_matching(
  env: "ManagerBasedRlEnv",
  std: float = 0.15,
  kernel: str = "exp",
  link_name: str = "torso_link",
) -> torch.Tensor:
  cmd = _get_predictor_command(env)
  pred = cmd.get_torso_roll_pitch_target()
  robot = env.scene["robot"]
  if link_name not in robot.body_names:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  torso_idx = robot.body_names.index(link_name)
  actual = _roll_pitch_from_quat(robot.data.body_link_quat_w[:, torso_idx, :])
  return roll_pitch_matching(env, predicted_rp=pred, actual_rp=actual, std=std, kernel=kernel)


def feet_pitch_matching(
  env: "ManagerBasedRlEnv",
  std: float = 0.25,
  sensor_name: str | None = None,
  contact_force_threshold: float = 50.0,
  kernel: str = "exp",
) -> torch.Tensor:
  mask = _predictor_ready_mask(env)
  cmd = _get_predictor_command(env)
  pitch_targets = cmd.get_feet_pitch_targets()
  if pitch_targets is None or mask.sum() == 0:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  robot = env.scene["robot"]
  root_quat_w = robot.data.root_link_quat_w
  body_quat_w = robot.data.body_link_quat_w
  root_inv = quat_inv(root_quat_w)
  left_rel = quat_mul(root_inv, body_quat_w[:, cmd.left_foot_body_idx, :])
  _, left_pitch, _ = euler_xyz_from_quat(left_rel)
  right_rel = quat_mul(root_inv, body_quat_w[:, cmd.right_foot_body_idx, :])
  _, right_pitch, _ = euler_xyz_from_quat(right_rel)
  actual_pitch = torch.stack([left_pitch, right_pitch], dim=1)
  desired_pitch = pitch_targets.clone()
  if sensor_name is not None:
    sensor = env.scene[sensor_name]
    if sensor.data.force is not None:
      fnorm = torch.norm(sensor.data.force[:, :2, :], dim=-1)
      in_contact = fnorm > contact_force_threshold
      desired_pitch = torch.where(in_contact, torch.zeros_like(desired_pitch), desired_pitch)
  pitch_err = actual_pitch - desired_pitch
  err_sq = torch.sum(pitch_err**2, dim=1)
  err_abs = torch.mean(torch.abs(pitch_err), dim=1)
  return _apply_tracking_kernel(err_sq, err_abs, kernel=kernel, std=std) * mask


def feet_yaw_matching(
  env: "ManagerBasedRlEnv",
  std: float = 0.25,
  kernel: str = "exp",
  command_name: str | None = None,
  ang_vel_threshold: float = 0.8,
) -> torch.Tensor:
  mask = _predictor_ready_mask(env)
  cmd = _get_predictor_command(env)
  yaw_targets = cmd.get_feet_relative_yaw_targets()
  if yaw_targets is None or mask.sum() == 0:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
  robot = env.scene["robot"]
  root_quat_w = robot.data.root_link_quat_w
  body_quat_w = robot.data.body_link_quat_w
  root_inv = quat_inv(root_quat_w)
  left_rel = quat_mul(root_inv, body_quat_w[:, cmd.left_foot_body_idx, :])
  right_rel = quat_mul(root_inv, body_quat_w[:, cmd.right_foot_body_idx, :])

  # Heading-only manifold error from root-relative foot quaternions.
  # This avoids Euler yaw instability when roll/pitch vary.
  fwd = torch.tensor([1.0, 0.0, 0.0], device=env.device, dtype=torch.float32).view(1, 3)
  left_fwd = quat_apply(left_rel, fwd.expand(env.num_envs, -1))
  right_fwd = quat_apply(right_rel, fwd.expand(env.num_envs, -1))
  actual_fwd_xy = torch.stack([left_fwd[:, :2], right_fwd[:, :2]], dim=1)
  actual_fwd_xy = torch.nn.functional.normalize(actual_fwd_xy, p=2, dim=-1, eps=1e-6)

  yaw_targets = (yaw_targets + math.pi) % (2 * math.pi) - math.pi
  target_fwd_xy = torch.stack([torch.cos(yaw_targets), torch.sin(yaw_targets)], dim=-1)
  target_fwd_xy = torch.nn.functional.normalize(target_fwd_xy, p=2, dim=-1, eps=1e-6)

  cross_z = actual_fwd_xy[..., 0] * target_fwd_xy[..., 1] - actual_fwd_xy[..., 1] * target_fwd_xy[..., 0]
  dot = (actual_fwd_xy * target_fwd_xy).sum(dim=-1)
  yaw_err = torch.atan2(cross_z, dot)
  err_sq = torch.mean(yaw_err**2, dim=1)
  err_abs = torch.mean(torch.abs(yaw_err), dim=1)
  base = _apply_tracking_kernel(err_sq, err_abs, kernel=kernel, std=std) * mask
  if command_name is None:
    return base
  cmd_vel = env.command_manager.get_command(command_name)
  assert cmd_vel is not None, f"Command '{command_name}' not found."
  gate = (torch.abs(cmd_vel[:, 2]) <= ang_vel_threshold).float()
  return base * gate


