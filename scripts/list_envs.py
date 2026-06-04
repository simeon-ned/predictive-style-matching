"""Backward-compatible wrapper. Prefer ``psm-list-envs`` after ``pip install -e .``."""

from psm._scripts.list_envs import main

if __name__ == "__main__":
  main()
