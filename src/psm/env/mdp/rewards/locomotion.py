"""Reward terms for the PSM G1 env (not provided by mjlab.tasks / mjlab.envs.mdp)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_apply_yaw,
  quat_inv,
  quat_mul,
)
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def lin_vel_z_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize vertical (z) linear velocity of the base (L2 norm)."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_w[:, 2])


def root_height_exp(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for maintaining target root height (Laplacian exponential kernel).

  Matches metaloco's ``base_height_cmd_exp``. Uses root link position directly —
  no body_ids required. If ``feet_asset_cfg`` is given, height is measured
  relative to the lowest foot (useful on rough terrain).
  """
  asset: Entity = env.scene[asset_cfg.name]
  desired_height: float | torch.Tensor = target_height
  error = torch.abs(asset.data.root_link_pos_w[:, 2] - desired_height)
  return torch.exp(-error / std)


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity in world frame.

  Matches AP: command (heading-frame XY) is rotated to world frame via yaw,
  then compared against world-frame velocity.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_world = torch.cat(
    [command[:, :2], torch.zeros(env.num_envs, 1, device=env.device)], dim=-1
  )
  cmd_world = quat_apply_yaw(asset.data.root_link_quat_w, cmd_world)
  lin_vel_error = torch.sum(
    torch.square(cmd_world[:, :2] - asset.data.root_link_lin_vel_w[:, :2]), dim=1
  )
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward commanded yaw-rate tracking in world frame.

  AP design keeps yaw tracking independent from roll/pitch stabilization.
  Roll/pitch angular velocity should be regularized by a separate penalty term.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_w
  z_error = torch.square(command[:, 2] - actual[:, 2])
  return torch.exp(-z_error / std**2)


def flat_orientation_exp(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-flat base orientation using exponential kernel.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)  # [B, 4]
    g = asset.data.gravity_vec_w.reshape(-1, 3)[0]
    g_expanded = g.unsqueeze(0).expand(body_quat_w.shape[0], -1)
    projected_gravity_b = quat_apply_inverse(body_quat_w, g_expanded)
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-xy_squared / std**2)


def body_orientation_exp(
  env: ManagerBasedRlEnv,
  std: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  mask: list[float] | None = None,
  sensor_name: str | None = None,
  contact_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize deviation from vertical using projected gravity in the body frame.

  For each body in ``asset_cfg.body_ids``, world up ``[0, 0, 1]`` is expressed in the
  body frame via ``quat_apply_inverse``. The horizontal part ``[..., :2]`` is
  ``(body X, body Y)`` of that vector. ``mask`` scales those two entries — index 0
  multiplies ``projected_gravity_b[..., 0]``, index 1 multiplies ``[..., 1]``. This is
  **not** an Euler-angle decomposition; informal "roll/pitch" wording elsewhere refers
  to these two components.

  Reward: ``exp(-sum_i (m_i * g_i)^2 / std^2)`` over masked horizontal gravity, optionally
  gated by contact when ``sensor_name`` is set.

  Args:
    env: Environment instance.
    std: Standard deviation for exponential reward scaling [default: 0.25].
    asset_cfg: Asset config; must specify body_ids (e.g. ankle bodies).
    mask: Weights for ``[projected_gravity_body_x, projected_gravity_body_y]``
      [default: [1.0, 1.0]].
    sensor_name: Optional contact sensor name; reward only when in contact (use plain
      string, not SceneEntityCfg — sensors do not support entity config resolution).
    contact_threshold: Force threshold (N) to consider body in contact [default: 10.0].
  """
  if mask is None:
    mask = [1.0, 1.0]
  asset: Entity = env.scene[asset_cfg.name]
  assert asset_cfg.body_ids is not None, "body_orientation_exp requires body_ids in asset_cfg"

  body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
  B, N, _ = body_quat_w.shape
  # World up [0, 0, 1] in body frame (match metaloco gravity convention)
  gravity = torch.tensor([0.0, 0.0, 1.0], device=env.device).view(1, 1, 3).expand(B, N, 3)
  quat_flat = body_quat_w.reshape(B * N, 4)
  gravity_flat = gravity.reshape(B * N, 3)
  projected_gravity_b = quat_apply_inverse(quat_flat, gravity_flat).view(B, N, 3)

  mask_tensor = torch.tensor(mask, device=env.device).view(1, 1, 2)
  masked_gravity_xy = projected_gravity_b[..., :2] * mask_tensor
  orientation_error = torch.sum(masked_gravity_xy**2, dim=-1)  # [B, N]
  reward = torch.exp(-orientation_error / std**2).mean(dim=-1)  # [B]

  if sensor_name is not None:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    data = contact_sensor.data
    # Prefer force magnitude for contact mask; fall back to contact time (mjlab).
    # Sensor data is indexed by sensor primary bodies (same order as asset_cfg when matching).
    if hasattr(data, "force") and data.force is not None:
      contact_forces = torch.norm(data.force, dim=-1)  # [B, N]
      bodies_in_contact = contact_forces > contact_threshold
    else:
      bodies_in_contact = data.current_contact_time > 0.0
    contact_mask = bodies_in_contact.float()
    contact_sum = contact_mask.sum(dim=1)
    valid_envs = contact_sum > 0
    contact_reward = torch.zeros_like(reward)
    if valid_envs.any():
      contact_reward[valid_envs] = (
        torch.exp(-orientation_error / std**2) * contact_mask
      ).sum(dim=1)[valid_envs] / contact_sum[valid_envs]
    return contact_reward

  return reward


def action_rate_l1(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize the rate of change of actions using L1 (sum of absolute differences).

  Matches metaloco/vel_vanilla action_rate_l1: penalty = sum_i |action_i - prev_action_i|.
  Use with negative weight (e.g. -0.1) for action smoothness.
  """
  return torch.sum(
    torch.abs(env.action_manager.action - env.action_manager.prev_action),
    dim=1,
  )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      nonzero_command = total_command > command_threshold
      # Matches AP: reward when moving, penalise airtime when standing still.
      reward = (
        reward * nonzero_command.float()
        - (current_air_time > 0).float().mean(dim=1) * (~nonzero_command).float()
      )
  return reward


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def body_yaw_alignment(
  env: "ManagerBasedRlEnv",
  std: float = 0.25,
  threshold: float = 0.1,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  offsets: list[float] | None = None,
) -> torch.Tensor:
  """Reward keeping specified bodies aligned with the base heading in yaw.

  The alignment is evaluated in the horizontal plane using forward direction
  vectors in world coordinates. Optional per-body yaw offsets can be used to
  encode slight out-toeing/in-toeing.
  """
  asset: Entity = env.scene[asset_cfg.name]
  device = env.device

  body_quats = asset.data.body_link_quat_w[:, asset_cfg.body_ids]  # [B, N, 4]
  root_quat = asset.data.root_link_quat_w  # [B, 4]

  num_bodies = body_quats.shape[1]
  if offsets is None:
    offsets = [0.0] * num_bodies
  if len(offsets) != num_bodies:
    raise ValueError(
      f"offsets must have length {num_bodies} to match the number of bodies"
    )

  # Forward vector in local coordinates (per env).
  forward_vec = torch.tensor([1.0, 0.0, 0.0], device=device).view(1, 1, 3)
  forward_vec = forward_vec.expand(env.num_envs, 1, 3)  # [B, 1, 3]

  # Root forward direction in world frame.
  root_quat_exp = root_quat.unsqueeze(1)  # [B, 1, 4]
  quat_flat = root_quat_exp.reshape(-1, 4)
  vec_flat = (-forward_vec).reshape(-1, 3)
  root_forward_flat = quat_apply_inverse(quat_flat, vec_flat)
  root_forward = root_forward_flat.view_as(forward_vec)  # [B, 1, 3]

  # Target forward directions in the XY-plane with per-body yaw offsets.
  offsets_tensor = torch.tensor(offsets, device=device)
  cos_o = torch.cos(offsets_tensor)
  sin_o = torch.sin(offsets_tensor)
  target_forwards = torch.stack([cos_o, sin_o, torch.zeros_like(cos_o)], dim=-1)  # [N,3]
  target_forwards = target_forwards.unsqueeze(0).expand(env.num_envs, -1, -1)  # [B,N,3]

  # Rotate target directions by body orientation to world frame.
  quat_flat_b = body_quats.reshape(-1, 4)
  vec_flat_b = (-target_forwards).reshape(-1, 3)
  body_forward_flat = quat_apply_inverse(quat_flat_b, vec_flat_b)
  body_forward = body_forward_flat.view_as(target_forwards)  # [B, N, 3]

  root_forward_xy = root_forward.expand(-1, num_bodies, -1)[..., :2]
  body_forward_xy = body_forward[..., :2]

  # Sine of angle between vectors (2D cross product magnitude).
  sin_angle = torch.abs(
    root_forward_xy[..., 0] * body_forward_xy[..., 1]
    - root_forward_xy[..., 1] * body_forward_xy[..., 0]
  )

  rewards = torch.exp(-(sin_angle**2) / (std**2))

  # Disable reward when commanded yaw rate is large.
  command = env.command_manager.get_command(command_name)
  assert command is not None
  ang_vel_cmd = torch.abs(command[:, 2])
  rewards *= (ang_vel_cmd <= threshold).unsqueeze(-1)

  return rewards.mean(dim=1)


def stand_still(
  env: "ManagerBasedRlEnv",
  target_height: float = 0.035,
  std: float = 0.2,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  vel_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward stable standing when commanded velocity is small.

  Matches PSM env: height / velocity / orientation / root-over-feet are
  computed on the configured bodies (typically feet or ankles). Orientation uses
  projected gravity in each body's frame, not the root alone.
  """
  asset: Entity = env.scene[asset_cfg.name]

  command = env.command_manager.get_command(command_name)
  assert command is not None
  vel_magnitude = torch.norm(command[:, :3], dim=1)
  reward_mask = vel_magnitude < vel_threshold

  # 1) Keep feet / ankle bodies at desired standing height.
  body_heights = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  height_error = torch.abs(body_heights - target_height)
  height_reward = torch.exp(-torch.mean(height_error, dim=1) / (std**2))

  # 2) Minimize residual linear + angular motion on those bodies.
  body_lin_vel = torch.norm(
    asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1
  )
  body_ang_vel = torch.norm(
    asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :], dim=-1
  )
  vel_reward = torch.exp(-torch.mean(body_lin_vel + 0.5 * body_ang_vel, dim=1) / std)

  # 3) Keep selected bodies flat (project gravity in each body frame).
  body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
  gravity_w = asset.data.gravity_vec_w
  gravity_w_exp = gravity_w.unsqueeze(1).expand(-1, body_quat_w.shape[1], -1)
  projected_gravity_b = quat_apply_inverse(
    body_quat_w.reshape(-1, 4), gravity_w_exp.reshape(-1, 3)
  ).reshape(body_quat_w.shape[0], body_quat_w.shape[1], 3)
  xy_squared = torch.sum(torch.square(projected_gravity_b[:, :, :2]), dim=-1)
  orientation_reward = torch.exp(-torch.mean(xy_squared, dim=1) / (std**2))

  # 4) Keep root position centered between the two feet / ankles.
  feet_xy = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :2]
  feet_midpoint = torch.mean(feet_xy, dim=1)
  root_xy = asset.data.root_link_pos_w[:, :2]
  root_error = torch.norm(root_xy - feet_midpoint, dim=1)
  root_reward = torch.exp(-root_error / std)

  combined = (height_reward + vel_reward + orientation_reward + root_reward) / 4.0
  return combined * reward_mask


def no_jump(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Cost that returns the number of self-collisions detected by a sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return -((sensor.data.found).sum(-1) == 0).float()


def coordinated_duty_cycle(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  slow_stance_ratio: float = 0.6,
  fast_stance_ratio: float = 0.5,
  min_cycle_time: float = 0.4,
  vel_transition: float = 0.3,
  vel_max: float = 1.0,
  std: float = 0.25,
  coordination_weight: float = 0.5,
  vel_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward coordinated duty cycle between two feet with adaptive stance ratio.

  When command magnitude is at or below *vel_threshold* (stand still / near zero),
  the reward is zero so gait shaping does not fight ``stand_still``-style terms.

  Expects *sensor_name* to refer to a :class:`ContactSensor` with exactly two feet.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  air_time = data.current_air_time  # [B, N]
  contact_time = data.current_contact_time  # [B, N]
  assert air_time is not None and contact_time is not None

  # Require exactly two feet for biped coordination.
  if air_time.shape[1] != 2:
    raise ValueError("coordinated_duty_cycle expects a contact sensor with 2 feet.")

  command = env.command_manager.get_command(command_name)
  assert command is not None

  lin_vel_magnitude = torch.norm(command[:, :2], dim=1)
  ang_vel_magnitude = torch.abs(command[:, 2]) * 0.5
  vel_magnitude = torch.max(lin_vel_magnitude, ang_vel_magnitude)

  # Adaptive target stance ratio.
  normalized_vel = torch.clamp(
    (vel_magnitude - vel_transition) / (vel_max - vel_transition), 0.0, 1.0
  )
  target_stance_ratio = (
    slow_stance_ratio * (1.0 - normalized_vel) + fast_stance_ratio * normalized_vel
  )  # [B]

  total_cycle_time = contact_time + air_time  # [B, 2]
  valid_cycle_mask = total_cycle_time > min_cycle_time

  actual_stance_ratio = torch.where(
    valid_cycle_mask,
    contact_time / (total_cycle_time + 1e-6),
    target_stance_ratio.unsqueeze(-1),
  )

  stance_ratio_error = torch.abs(
    actual_stance_ratio - target_stance_ratio.unsqueeze(-1)
  )
  individual_rewards = torch.exp(-stance_ratio_error / std)
  individual_rewards = torch.where(
    valid_cycle_mask, individual_rewards, torch.ones_like(individual_rewards)
  )
  individual_reward = torch.mean(individual_rewards, dim=1)

  # Coordination term: encourage alternating pattern.
  in_contact = contact_time > 0.1
  both_in_contact = torch.all(in_contact, dim=1)
  both_in_swing = torch.all(~in_contact, dim=1)
  single_support = torch.sum(in_contact.int(), dim=1) == 1

  coordination_reward = (
    single_support.float()
    + 1.0 * both_in_contact.float()
    - 0.5 * both_in_swing.float()
  )

  combined = (1.0 - coordination_weight) * individual_reward + coordination_weight * coordination_reward
  return combined * (vel_magnitude > vel_threshold).float()


def step_length(
  env: "ManagerBasedRlEnv",
  std: float = 0.25,
  target_length: float = 0.5,
  command_name: str = "twist",
  vel_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str = "feet_ground_contact",
  last_step_length: list[torch.Tensor] = [],
) -> torch.Tensor:
  """Reward maintaining a specific step length between feet in the base frame."""
  asset: Entity = env.scene[asset_cfg.name]

  command = env.command_manager.get_command(command_name)
  assert command is not None
  vel_magnitude = torch.norm(command[:, :2], dim=1)

  # Only apply reward when both commanded and actual forward speed exceed threshold.
  reward_mask_cmd_fwd = torch.abs(command[:, 0]) > vel_threshold
  reward_mask_actual_fwd = torch.abs(asset.data.root_link_lin_vel_b[:, 0]) > vel_threshold
  reward_mask = reward_mask_cmd_fwd & reward_mask_actual_fwd

  # Foot positions in base frame.
  root_pos = asset.data.root_link_pos_w.unsqueeze(1)  # [B, 1, 3]
  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # [B, 2, 3]
  rel_pos = foot_pos_w - root_pos  # [B, 2, 3]

  root_quat = asset.data.root_link_quat_w  # [B, 4]
  quat = root_quat.unsqueeze(1).expand(-1, rel_pos.shape[1], -1)  # [B, 2, 4]
  rel_flat = rel_pos.reshape(-1, 3)
  quat_flat = quat.reshape(-1, 4)
  rel_b_flat = quat_apply_inverse(quat_flat, rel_flat)
  rel_pos_b = rel_b_flat.view_as(rel_pos)  # [B, 2, 3]

  # Forward distance in local X between the two feet.
  x_distance = torch.abs(rel_pos_b[:, 0, 0] - rel_pos_b[:, 1, 0])

  # Contact sensor for determining when both feet are on the ground.
  contact_sensor: ContactSensor = env.scene[sensor_name]
  currently_in_contact = contact_sensor.data.current_contact_time > 0.0
  # Expect exactly two feet; use all slots.
  currently_in_contact = currently_in_contact  # [B, 2]
  both_in_contact = currently_in_contact[:, 0] & currently_in_contact[:, 1]

  if len(last_step_length) == 0:
    last_step_length.append(torch.zeros_like(x_distance))
    # No reward for the first step.
    reward_mask = reward_mask & torch.zeros_like(reward_mask, dtype=torch.bool)
  else:
    last_step_length[0] = torch.where(both_in_contact, x_distance, last_step_length[0])

  step_length_error = last_step_length[0] - target_length
  base_reward = torch.exp(-(step_length_error**2) / (std**2))
  fallback_reward = torch.exp(-(last_step_length[0] ** 2) / (std**2))

  return reward_mask * base_reward + (~reward_mask) * 0.1 * fallback_reward


def feet_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  contact_threshold: float = 100.0,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  command_idx: tuple[int, ...] = (0, 1, 2),
) -> torch.Tensor:
  """Reward 1.0 when all tracked feet are in contact, gated by low command."""
  sensor: ContactSensor = env.scene[sensor_name]

  if sensor.data.force is not None:
    contact_forces = torch.norm(sensor.data.force, dim=-1)
    in_contact = contact_forces > contact_threshold
  else:
    assert sensor.data.found is not None
    in_contact = sensor.data.found > 0

  reward = in_contact.all(dim=1).float()

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    assert command is not None
    cmd_mag = torch.norm(command[:, list(command_idx)], dim=1)
    reward = reward * (cmd_mag < command_threshold).float()

  return reward



# Alias used by reward cfg.
contact = feet_contact
