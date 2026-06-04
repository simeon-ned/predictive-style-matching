from __future__ import annotations

from typing import Any

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner

from psm.env.utils.predictor_snapshot import snapshot_predictor_to_log_dir


class PsmG1OnPolicyRunner(VelocityOnPolicyRunner):
  """G1 PSM runner; snapshots packaged predictor weights into the log on train."""

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict[str, Any],
    log_dir: str | None = None,
    device: str = "cpu",
    **kwargs: Any,
  ):
    super().__init__(env, train_cfg, log_dir, device, **kwargs)
    snapshot_predictor_to_log_dir(env, log_dir)
