from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .commands import PsmVelocityCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_predictor_command(env: "ManagerBasedRlEnv") -> PsmVelocityCommand:
  cmd = env.command_manager.get_term("twist")
  if cmd is None:
    raise KeyError("PSM velocity command 'twist' not found.")
  return cast(PsmVelocityCommand, cmd)


def predictor_upper_targets(env: "ManagerBasedRlEnv") -> torch.Tensor:
  cmd = _get_predictor_command(env)
  return cmd.get_upper_joint_target()


def predictor_body_features(env: "ManagerBasedRlEnv") -> torch.Tensor:
  cmd = _get_predictor_command(env)
  return cmd.get_body_feature_targets()


def predictor_step_length_target(env: "ManagerBasedRlEnv") -> torch.Tensor:
  cmd = _get_predictor_command(env)
  target = cmd.get_step_length_target()
  return target


def predictor_step_width_target(env: "ManagerBasedRlEnv") -> torch.Tensor:
  cmd = _get_predictor_command(env)
  target = cmd.get_step_width_target()
  return target
