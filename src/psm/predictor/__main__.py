"""Allow ``python -m psm.predictor`` (prints how to run play/train)."""

from __future__ import annotations

import sys


def main() -> None:
  print(
    "Use: python -m psm.predictor.play  or  python -m psm.predictor.train\n"
    "Console scripts: psm-predictor-play, psm-predictor-train",
    file=sys.stderr,
  )
  raise SystemExit(2)


if __name__ == "__main__":
  main()
