from __future__ import annotations

from typing import Any

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner

from psm.env.weights.snapshot import snapshot_to_log_dir


class PsmG1OnPolicyRunner(VelocityOnPolicyRunner):
  """G1 PSM runner using the standard ManagerBasedRlEnv.

  The PSM predictor is integrated via the ``twist`` command term (``PsmVelocityCommand``),
  so no custom environment subclass is required.
  """

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict[str, Any],
    log_dir: str | None = None,
    device: str = "cpu",
    **kwargs: Any,
  ):
    super().__init__(env, train_cfg, log_dir, device, **kwargs)
    snapshot_to_log_dir(env, log_dir)

