"""Log key Psm-G1 settings after tyro resolves the train/play config."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from psm.env.mdp.commands import PsmVelocityCommandCfg


def log_psm_train_config(task_id: str, env_cfg: ManagerBasedRlEnvCfg) -> None:
  if task_id != "Psm-G1":
    return

  twist = env_cfg.commands.get("twist")
  if not isinstance(twist, PsmVelocityCommandCfg):
    print(f"[WARN] Psm-G1: expected PsmVelocityCommandCfg, got {type(twist)!r}")
    return

  rw = env_cfg.rewards
  print(
    "[INFO] Psm-G1 config:"
    f"\n      predictor_path={twist.predictor_path}"
    f"\n      num_envs={env_cfg.scene.num_envs}"
    f"\n      reward weights: track_lin={rw['track_linear_velocity'].weight},"
    f" track_ang={rw['track_angular_velocity'].weight},"
    f" upper_joints={rw['upper_joints'].weight}"
    f"\n      curriculum terms: {len(env_cfg.curriculum)}"
  )
