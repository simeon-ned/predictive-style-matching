"""Wrapper for mjlab's list_envs script that also registers PSM tasks.

Usage:
  python scripts/list_envs.py
  psm-list-envs
"""

from mjlab.scripts.list_envs import main as _mjlab_list_envs_main


def main() -> None:
  import psm.env.register  # noqa: F401

  _mjlab_list_envs_main()


if __name__ == "__main__":
  main()
