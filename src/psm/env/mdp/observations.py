from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _body_index(asset: Entity, body_name: str) -> int:
  return asset.body_names.index(body_name)


def torso_ang_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  body_name: str = "torso_link",
) -> torch.Tensor:
  """Torso angular velocity in the torso frame."""
  asset: Entity = env.scene[asset_cfg.name]
  idx = _body_index(asset, body_name)
  quat_w = asset.data.body_link_quat_w[:, idx]
  ang_vel_w = asset.data.body_link_ang_vel_w[:, idx]
  return quat_apply_inverse(quat_w, ang_vel_w)


def torso_projected_gravity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  body_name: str = "torso_link",
) -> torch.Tensor:
  """Gravity direction in the torso frame."""
  asset: Entity = env.scene[asset_cfg.name]
  idx = _body_index(asset, body_name)
  quat_w = asset.data.body_link_quat_w[:, idx]
  return quat_apply_inverse(quat_w, asset.data.gravity_vec_w)


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  phase_obs = torch.zeros(env.num_envs, 2, device=env.device)
  phase_obs[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
  phase_obs[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
  stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
  phase_obs = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase_obs), phase_obs)
  return phase_obs
