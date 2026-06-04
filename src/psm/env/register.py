"""Register the Psm-G1 task with mjlab (import this module from train/play)."""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from psm.env.cfg import psm_env_cfg
from psm.env.rl_cfg import g1_psm_ppo_runner_cfg
from psm.env.runner import PsmG1OnPolicyRunner

register_mjlab_task(
  task_id="Psm-G1",
  env_cfg=psm_env_cfg(),
  play_env_cfg=psm_env_cfg(play=True),
  rl_cfg=g1_psm_ppo_runner_cfg(),
  runner_cls=PsmG1OnPolicyRunner,
)
