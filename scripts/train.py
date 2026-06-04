"""Backward-compatible wrapper. Prefer ``psm-env-train`` after ``pip install -e .``."""

from psm._scripts.train import main

if __name__ == "__main__":
  main()
