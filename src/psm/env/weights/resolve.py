"""Resolve ``params/predictor`` next to a policy checkpoint from ``sys.argv``.

Play builds the env before the checkpoint is applied to the policy; train does
the same (env first, then ``runner.load``).  The registered env cfg still has
the package-default ``predictor_path``.  By scanning ``sys.argv`` when
``PsmVelocityCommand`` inits, we mirror mjlab's checkpoint resolution:

* **Play:** ``--checkpoint-file`` or ``--wandb-run-path`` (no ``--agent.resume``).
* **Train resume:** ``--agent.resume True`` with local ``--agent.load-run`` /
  ``--agent.load-checkpoint`` or with top-level ``--wandb-run-path`` (same as
  ``mjlab.scripts.train.run_train``).

Fresh training (resume false) keeps the packaged ``data/`` dir unless you
override ``predictor_path`` in config.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Must match ``cfg.py`` default: ``weights/data`` next to this package.
_DEFAULT_PACKAGE_DATA_DIR = Path(__file__).resolve().parent / "data"


def package_data_dir() -> Path:
  return _DEFAULT_PACKAGE_DATA_DIR.resolve()


def _argv_flag_value(argv: list[str], *flags: str) -> str | None:
  for i, a in enumerate(argv):
    for fl in flags:
      if a == fl and i + 1 < len(argv):
        return argv[i + 1]
      eq = fl + "="
      if a.startswith(eq):
        return a[len(eq) :]
  return None


def _log_bundle_dir_if_valid(run_dir: Path) -> Path | None:
  snap = (run_dir / "params" / "predictor").resolve()
  if (snap / "metadata.pkl").is_file() and (snap / "predictor.pth").is_file():
    return snap
  return None


def _argv_agent_resume_true(argv: list[str]) -> bool:
  """Match tyro + ``FlagConversionOff``: ``--agent.resume True`` / ``=True``."""
  for a in argv:
    if a.startswith("--agent.resume="):
      return a.split("=", 1)[1].strip().lower() in ("true", "1", "yes", "on")
  for i, a in enumerate(argv):
    if a == "--agent.resume" and i + 1 < len(argv):
      return argv[i + 1].strip().lower() in ("true", "1", "yes", "on")
  return False


def _train_resume_checkpoint_file(argv: list[str]) -> Path | None:
  """Resolve policy checkpoint path the same way as ``mjlab.scripts.train.run_train``."""
  if not _argv_agent_resume_true(argv):
    return None

  from mjlab.utils.os import get_checkpoint_path, get_wandb_checkpoint_path

  from psm.env.rl_cfg import g1_psm_ppo_runner_cfg

  experiment_name = g1_psm_ppo_runner_cfg().experiment_name
  log_root_path = (Path("logs") / "rsl_rl" / experiment_name).resolve()

  wandb_run = _argv_flag_value(argv, "--wandb-run-path", "--wandb_run_path")
  if wandb_run is not None:
    ckpt_name = _argv_flag_value(argv, "--wandb-checkpoint-name", "--wandb_checkpoint_name")
    resume, _ = get_wandb_checkpoint_path(log_root_path, Path(wandb_run), ckpt_name)
    return resume if resume.is_file() else None

  load_run = _argv_flag_value(argv, "--agent.load-run", "--agent.load_run") or ".*"
  load_ckpt = (
    _argv_flag_value(argv, "--agent.load-checkpoint", "--agent.load_checkpoint")
    or "model_.*.pt"
  )
  resume = get_checkpoint_path(log_root_path, load_run, load_ckpt)
  return resume if resume.is_file() else None


def infer_log_snapshot_dir_from_argv() -> Path | None:
  """Return ``.../params/predictor`` next to the checkpoint implied by argv, if valid."""
  argv = sys.argv[1:]

  ckpt_s = _argv_flag_value(argv, "--checkpoint-file", "--checkpoint_file")
  if ckpt_s is not None:
    resume = Path(ckpt_s).expanduser().resolve()
    if resume.is_file():
      return _log_bundle_dir_if_valid(resume.parent)
    return None

  train_resume_ckpt = _train_resume_checkpoint_file(argv)
  if train_resume_ckpt is not None:
    return _log_bundle_dir_if_valid(train_resume_ckpt.parent)

  wandb_run = _argv_flag_value(argv, "--wandb-run-path", "--wandb_run_path")
  if wandb_run is None:
    return None

  from mjlab.utils.os import get_wandb_checkpoint_path

  from psm.env.rl_cfg import g1_psm_ppo_runner_cfg

  experiment_name = g1_psm_ppo_runner_cfg().experiment_name
  log_root_path = (Path("logs") / "rsl_rl" / experiment_name).resolve()
  ckpt_name = _argv_flag_value(argv, "--wandb-checkpoint-name", "--wandb_checkpoint_name")
  resume, _ = get_wandb_checkpoint_path(log_root_path, Path(wandb_run), ckpt_name)
  if resume.is_file():
    return _log_bundle_dir_if_valid(resume.parent)
  return None


def effective_data_path(cfg_predictor_path: str) -> tuple[str, bool]:
  """Return (path, used_log_snapshot).

  If the config still points at the packaged default ``data/`` and argv
  references a checkpoint run (play, or train with ``--agent.resume True``) that
  contains ``params/predictor``, use that log bundle.
  """
  cfg_resolved = Path(cfg_predictor_path).expanduser().resolve()
  default_resolved = package_data_dir()
  if cfg_resolved != default_resolved:
    return cfg_predictor_path, False

  snap = infer_log_snapshot_dir_from_argv()
  if snap is None:
    return cfg_predictor_path, False

  print(
    "[INFO] PSM: using weights from log snapshot next to policy checkpoint:\n"
    f"      {snap}"
  )
  return str(snap), True
