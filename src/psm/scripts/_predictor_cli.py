"""Map ``--predictor-path`` / ``--predictor-bundled`` to mjlab env overrides."""

from __future__ import annotations

from psm.predictor.bundle import bundled_weights_dir, default_predictor_path


def _argv_has_predictor_override(argv: list[str]) -> bool:
  for a in argv:
    if a == "--env.commands.twist.predictor-path" or a.startswith(
      "--env.commands.twist.predictor-path="
    ):
      return True
    if a == "--predictor-path" or a.startswith("--predictor-path="):
      return True
    if a == "--predictor-bundled":
      return True
  return False


def apply_predictor_cli(argv: list[str]) -> list[str]:
  """Expand PSM predictor shortcuts for ``psm-env-train`` / ``psm-env-play``."""
  out: list[str] = []
  i = 0
  while i < len(argv):
    arg = argv[i]
    if arg == "--predictor-bundled":
      out.extend(
        [
          "--env.commands.twist.predictor-path",
          str(bundled_weights_dir().resolve()),
        ]
      )
      i += 1
      continue
    if arg.startswith("--predictor-bundled="):
      raise ValueError("--predictor-bundled does not take a value")
    if arg.startswith("--predictor-path="):
      out.extend(["--env.commands.twist.predictor-path", arg.split("=", 1)[1]])
      i += 1
      continue
    if arg == "--predictor-path":
      if i + 1 >= len(argv):
        raise ValueError("--predictor-path requires a directory")
      out.extend(["--env.commands.twist.predictor-path", argv[i + 1]])
      i += 2
      continue
    out.append(arg)
    i += 1
  return out


def log_default_predictor_if_unset(argv: list[str]) -> None:
  if _argv_has_predictor_override(argv):
    return
  path = default_predictor_path()
  print(f"[INFO] PSM predictor bundle (default): {path}")
