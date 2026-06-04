"""Predictor bundle paths: train logs, packaged default, and RL auto-selection."""

from __future__ import annotations

from pathlib import Path

PREDICTOR_LOGS_DIR = Path("logs") / "predictor"


def bundled_weights_dir() -> Path:
  """Packaged tuned checkpoint under ``psm/predictor/weights/``."""
  return Path(__file__).resolve().parent / "weights"


def is_predictor_bundle(directory: Path) -> bool:
  directory = directory.expanduser().resolve()
  return (directory / "predictor.pth").is_file() and (directory / "metadata.pkl").is_file()


def latest_predictor_train_dir() -> Path | None:
  """Newest complete bundle under ``logs/predictor/<timestamp>/``."""
  root = PREDICTOR_LOGS_DIR.resolve()
  if not root.is_dir():
    return None
  candidates = [d for d in root.iterdir() if d.is_dir() and is_predictor_bundle(d)]
  if not candidates:
    return None
  return sorted(candidates, key=lambda p: p.name)[-1]


def default_predictor_path(*, use_bundled: bool = False) -> str:
  """Path for RL env cfg: latest train log, else packaged weights.

  Pass ``use_bundled=True`` to force the repo-shipped bundle (``--predictor-bundled``).
  """
  if use_bundled:
    return str(bundled_weights_dir().resolve())
  latest = latest_predictor_train_dir()
  if latest is not None:
    return str(latest.resolve())
  return str(bundled_weights_dir().resolve())


def is_auto_predictor_path(path: str) -> bool:
  """True when the user did not set a custom bundle (RL may override from policy log)."""
  resolved = Path(path).expanduser().resolve()
  if resolved == bundled_weights_dir().resolve():
    return True
  latest = latest_predictor_train_dir()
  if latest is not None and resolved == latest.resolve():
    return True
  return False
