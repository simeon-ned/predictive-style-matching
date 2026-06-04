"""PSM G1 RL environment package.

Import ``psm.env.register`` (or use ``psm-env-train``) to register the mjlab task.
Config factories live in ``psm.env.cfg`` — not re-exported here to avoid import cycles.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
  if name in ("make_psm_env_cfg", "psm_env_cfg"):
    from psm.env.cfg.env_cfg import make_psm_env_cfg, psm_env_cfg

    return {"make_psm_env_cfg": make_psm_env_cfg, "psm_env_cfg": psm_env_cfg}[name]
  if name == "g1_psm_ppo_runner_cfg":
    from psm.env.cfg.rsl_rl_cfg import g1_psm_ppo_runner_cfg

    return g1_psm_ppo_runner_cfg
  if name == "PsmG1OnPolicyRunner":
    from psm.env.runner import PsmG1OnPolicyRunner

    return PsmG1OnPolicyRunner
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  "make_psm_env_cfg",
  "psm_env_cfg",
  "g1_psm_ppo_runner_cfg",
  "PsmG1OnPolicyRunner",
]
