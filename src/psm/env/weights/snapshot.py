"""Copy bundled weights into the RL log dir and point ``env.yaml`` at them.

Training writes ``params/env.yaml`` before the runner starts, so it initially
records whatever ``predictor_path`` was used to construct the env (often a path
inside the package).  We copy ``metadata.pkl``, ``predictor.pth``, and any
other files from that directory into ``params/predictor/`` and rewrite
``env.yaml`` so an old run folder alone is enough to rebuild the same env.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper

from psm.env.mdp.commands import PsmVelocityCommandCfg


def snapshot_to_log_dir(
  env: RslRlVecEnvWrapper,
  log_dir: str | None,
) -> None:
  """Rank-0 only: copy artifacts next to ``params/env.yaml`` and fix path."""
  if not log_dir:
    return
  if int(os.environ.get("RANK", "0")) != 0:
    return

  raw = env.unwrapped
  if not isinstance(raw, ManagerBasedRlEnv):
    return

  twist_cfg = raw.cfg.commands.get("twist")
  if not isinstance(twist_cfg, PsmVelocityCommandCfg):
    return

  src = Path(twist_cfg.predictor_path).expanduser().resolve()
  if not src.is_dir():
    print(f"[WARN] psm predictor_path is not a directory: {src}")
    return

  log_root = Path(log_dir)
  dst = log_root / "params" / "predictor"
  dst.mkdir(parents=True, exist_ok=True)

  copied: list[str] = []
  for item in sorted(src.iterdir()):
    if item.is_file():
      shutil.copy2(item, dst / item.name)
      copied.append(item.name)

  meta: dict[str, Any] = {
    "source_path_at_train_time": str(src),
    "log_weights_path": str(dst.resolve()),
    "copied_files": copied,
  }
  with open(dst / "manifest.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(meta, f, sort_keys=False)

  env_yaml = log_root / "params" / "env.yaml"
  if env_yaml.is_file():
    try:
      with open(env_yaml, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
      if isinstance(doc, dict):
        cmds = doc.get("commands")
        if isinstance(cmds, dict):
          twist = cmds.get("twist")
          if isinstance(twist, dict):
            twist["predictor_path"] = str(dst.resolve())
            with open(env_yaml, "w", encoding="utf-8") as f:
              yaml.safe_dump(doc, f, sort_keys=False)
    except Exception as e:
      print(f"[WARN] Could not update env.yaml predictor_path: {e}")
  else:
    print(f"[WARN] env.yaml not found at {env_yaml}; files copied but YAML not patched")

  print(
    f"[INFO] PSM weights snapshot: {len(copied)} file(s) -> {dst} "
    "(see params/predictor/manifest.yaml)"
  )
