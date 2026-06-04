"""Wrapper for mjlab's train script that auto-registers PSM tasks.

Usage:
  psm-env-train Psm-G1
  psm-env-train Psm-G1 --predictor-path logs/predictor/2026-01-01_12-00-00
  psm-env-train Psm-G1 --predictor-bundled

By default this wrapper sets ``--agent.logger tensorboard`` so training does not
initialize Weights & Biases (no login prompts). Pass ``--use_wandb`` to use the
task's normal W&B logging (default ``logger`` from ``RslRlBaseRunnerCfg``).

You can still force W&B without the flag via ``--agent.logger wandb``; you can
force TensorBoard with ``--use_wandb`` by overriding, e.g.
``--use_wandb --agent.logger tensorboard`` (last flag wins per tyro).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import mjlab.utils.os as _mjlab_os

from psm.scripts._predictor_cli import apply_predictor_cli, log_default_predictor_if_unset

log = logging.getLogger(__name__)


def _has_explicit_agent_logger(argv: list[str]) -> bool:
  """True if argv already constrains ``agent.logger`` (tyro ``--agent.logger`` forms)."""
  for a in argv:
    if a == "--agent.logger" or a.startswith("--agent.logger="):
      return True
  return False


def _filter_argv_for_wandb_flag(argv: list[str]) -> tuple[list[str], bool]:
  """Remove ``--use_wandb`` from argv and return whether it was present."""
  use = False
  out: list[str] = []
  i = 0
  while i < len(argv):
    if argv[i] == "--use_wandb":
      use = True
      i += 1
      continue
    out.append(argv[i])
    i += 1
  return out, use


def _apply_motion_file_shortcut(argv: list[str]) -> list[str]:
  """Map ``--motion-file`` to mjlab's nested env override argument.

  This keeps tracking training fully offline without relying on W&B registry
  arguments, while still using mjlab's original train entrypoint.
  """
  out: list[str] = []
  i = 0
  while i < len(argv):
    arg = argv[i]
    if arg.startswith("--motion-file="):
      motion_file = arg.split("=", 1)[1]
      out.extend(["--env.commands.motion.motion-file", motion_file])
      i += 1
      continue
    if arg == "--motion-file":
      if i + 1 >= len(argv):
        raise ValueError("--motion-file requires a value")
      out.extend(["--env.commands.motion.motion-file", argv[i + 1]])
      i += 2
      continue
    out.append(arg)
    i += 1
  return out


def _apply_default_logger(argv: list[str], *, use_wandb: bool) -> list[str]:
  """If not using W&B and user did not set logger, default to tensorboard."""
  if use_wandb or _has_explicit_agent_logger(argv):
    return argv
  return [*argv, "--agent.logger", "tensorboard"]


def _safe_dump_yaml(filename: Path, data, sort_keys: bool = False) -> None:
  """Drop-in replacement for mjlab's dump_yaml that silently skips non-serializable fields.

  This is needed because rsl_rl's resolve_symmetry_config injects the live
  environment object (which contains mujoco._specs.MjSpec) into agent_cfg
  before dump_yaml is called.  Rather than removing the symmetry feature,
  we simply make the YAML dump fault-tolerant.
  """
  import yaml

  def _sanitize(obj):
    if isinstance(obj, dict):
      return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
      return type(obj)(_sanitize(v) for v in obj)
    if isinstance(obj, (str, int, float, bool, type(None))):
      return obj
    if callable(obj):
      return f"{obj.__module__}.{obj.__qualname__}"
    try:
      yaml.dump(obj)
      return obj
    except Exception:
      return f"<non-serializable: {type(obj).__name__}>"

  filename = Path(filename)
  if not filename.suffix:
    filename = filename.with_suffix(".yaml")
  filename.parent.mkdir(parents=True, exist_ok=True)
  with open(filename, "w") as f:
    yaml.dump(_sanitize(data), f, sort_keys=sort_keys)


# Patch before mjlab's train script runs so dump_yaml calls use our version.
_mjlab_os.dump_yaml = _safe_dump_yaml


def _patch_mjlab_train_logging() -> None:
  import mjlab.scripts.train as mjlab_train

  _run_train = mjlab_train.run_train

  def _run_train_with_psm_log(task_id: str, cfg, log_dir):
    from psm.scripts._task_log import log_psm_train_config

    log_psm_train_config(task_id, cfg.env)
    return _run_train(task_id, cfg, log_dir)

  mjlab_train.run_train = _run_train_with_psm_log


def main() -> None:
  # Import tasks so register_mjlab_task runs before mjlab reads the registry.
  import psm.env.register  # noqa: F401

  prog = sys.argv[0]
  rest, use_wandb = _filter_argv_for_wandb_flag(sys.argv[1:])
  rest = _apply_motion_file_shortcut(rest)
  rest = apply_predictor_cli(rest)
  log_default_predictor_if_unset(rest)
  rest = _apply_default_logger(rest, use_wandb=use_wandb)
  sys.argv = [prog, *rest]

  _patch_mjlab_train_logging()

  from mjlab.scripts.train import main as _mjlab_train_main

  _mjlab_train_main()


if __name__ == "__main__":
  main()
