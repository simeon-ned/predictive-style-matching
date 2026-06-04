"""Backward-compatible wrapper. Prefer ``psm-env-play`` after ``pip install -e .``."""

from psm._scripts.play import main

if __name__ == "__main__":
  main()
