"""Resolve which PsmPredictor bundle directory the env should load."""

from __future__ import annotations

import sys
from pathlib import Path

from psm.predictor.bundle import is_auto_predictor_path


def _argv_flag_value(argv: list[str], *flags: str) -> str | None:
  for i, a in enumerate(argv):
    for fl in flags:
      if a == fl and i + 1 < len(argv):
        return argv[i + 1]
      eq = fl + "="
      if a.startswith(eq):
        return a[len(eq) :]
  return None


def _log_predictor_bundle(run_dir: Path) -> Path | None:
  bundle = (run_dir / "params" / "predictor").resolve()
  if (bundle / "metadata.pkl").is_file() and (bundle / "predictor.pth").is_file():
    return bundle
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

  from psm.env.cfg.rsl_rl_cfg import g1_psm_ppo_runner_cfg

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


def infer_log_predictor_dir_from_argv() -> Path | None:
  """Return ``<run>/params/predictor`` implied by CLI checkpoint flags, if valid."""
  argv = sys.argv[1:]

  ckpt_s = _argv_flag_value(argv, "--checkpoint-file", "--checkpoint_file")
  if ckpt_s is not None:
    resume = Path(ckpt_s).expanduser().resolve()
    if resume.is_file():
      return _log_predictor_bundle(resume.parent)
    return None

  train_resume_ckpt = _train_resume_checkpoint_file(argv)
  if train_resume_ckpt is not None:
    return _log_predictor_bundle(train_resume_ckpt.parent)

  wandb_run = _argv_flag_value(argv, "--wandb-run-path", "--wandb_run_path")
  if wandb_run is None:
    return None

  from mjlab.utils.os import get_wandb_checkpoint_path

  from psm.env.cfg.rsl_rl_cfg import g1_psm_ppo_runner_cfg

  experiment_name = g1_psm_ppo_runner_cfg().experiment_name
  log_root_path = (Path("logs") / "rsl_rl" / experiment_name).resolve()
  ckpt_name = _argv_flag_value(argv, "--wandb-checkpoint-name", "--wandb_checkpoint_name")
  resume, _ = get_wandb_checkpoint_path(log_root_path, Path(wandb_run), ckpt_name)
  if resume.is_file():
    return _log_predictor_bundle(resume.parent)
  return None


def effective_predictor_path(cfg_predictor_path: str) -> tuple[str, bool]:
  """Return ``(path, used_rl_log_bundle)`` for ``PsmVelocityCommand`` init.

  Custom ``predictor_path`` in config is left unchanged. Auto paths (latest
  ``logs/predictor`` run or packaged weights) may be replaced by
  ``<policy_run>/params/predictor`` when play/resume flags imply a checkpoint.
  """
  if not is_auto_predictor_path(cfg_predictor_path):
    return cfg_predictor_path, False

  bundle = infer_log_predictor_dir_from_argv()
  if bundle is None:
    return cfg_predictor_path, False

  print(
    "[INFO] PSM: using predictor bundle next to policy checkpoint:\n"
    f"      {bundle}"
  )
  return str(bundle), True


effective_data_path = effective_predictor_path
infer_log_snapshot_dir_from_argv = infer_log_predictor_dir_from_argv
