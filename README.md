# Predictive Style Matching (PSM)

[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f?style=for-the-badge)](https://simeon-ned.github.io/predictive-style-matching/)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b?style=for-the-badge)](https://simeon-ned.github.io/predictive-style-matching/)
<!-- [![Cite](https://img.shields.io/badge/Cite-BibTeX-555?style=for-the-badge)](https://github.com/simeon-ned/predictive-style-matching#citation) — enable after arXiv -->

Code for *Predictive Style Matching: Natural and Robust Humanoid Locomotion* on Unitree G1 (first-draft implementation, **work in progress**).

The project page lives in [`docs/`](docs/) (GitHub Pages). The arXiv badge will point to the preprint once posted; until then it opens the site.

## Dependencies

RL training and simulation build on **[mjlab](https://github.com/mujocolab/mjlab)**. This repo registers the `Psm-G1` task via mjlab’s task entry point. See the [mjlab docs](https://mujocolab.github.io/mjlab/) for install notes, multi-GPU training, and play/export.

Hardware deployment is described for [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) (also mjlab-based).

## Install

```bash
git clone https://github.com/simeon-ned/predictive-style-matching.git
cd predictive-style-matching
pip install -e .   # installs psm + mjlab
```

## Quick start

**Predictor (offline)**

```bash
psm-predictor-train
psm-predictor-play --npz data/motions/your_clip.npz
```

Logs go to `logs/predictor/<timestamp>/` (`predictor.pth`, `metadata.pkl`, `config.yaml`).

**RL ([mjlab](https://github.com/mujocolab/mjlab))**

```bash
psm-env-train Psm-G1
psm-env-play Psm-G1
psm-list-envs   # optional: list registered tasks
```

Defaults: **4096** envs for train, **1** for play (override with `--env.scene.num-envs`).

By default, RL uses the **latest** bundle under `logs/predictor/`, then falls back to `src/psm/predictor/weights/` if none exist. Override:

```bash
psm-env-train Psm-G1 --predictor-path /path/to/bundle
psm-env-train Psm-G1 --predictor-bundled
# or: --env.commands.twist.predictor-path /path/to/bundle
```

Policy logs: `logs/rsl_rl/g1_psm/` (each run snapshots the active predictor under `params/predictor/`). ONNX: `<run>/params/latest.onnx` on play.

**Deploy:** [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) on the robot; copy the exported policy from your play run.

## Repository layout

```text
├── README.md           # this file — code & usage
├── pyproject.toml
├── data/
├── src/psm/
│   ├── predictor/      # training; logs → logs/predictor/; weights/ = packaged fallback
│   ├── scripts/        # psm-env-train, psm-env-play, utilities
│   └── env/            # RL (Psm-G1): cfg/, mdp/, runner.py, utils/ (deploy, symmetry, predictor path/log)
└── docs/               # GitHub Pages project site (not Python)
    ├── index.html
    └── static/
```

Paper TeX may live in a separate private repo during review; figures for the site are under `docs/static/images/`.

<!-- ## Citation — enable after arXiv posting

If you use this work, please cite:

```bibtex
@misc{nedelchev2026psm,
  title         = {Predictive Style Matching: Natural and Robust Humanoid Locomotion},
  author        = {Nedelchev, Simeon and Zaliaev, Eduard and Chaikovskaia, Ekaterina and Davydenko, Egor and Gorbachev, Roman},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/XXXX.XXXXX}
}
```
-->

## Acknowledgments

- [mjlab](https://github.com/mujocolab/mjlab) — RL environment and training stack
- [MuJoCo](https://github.com/google-deepmind/mujoco) — physics simulation
- [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) — G1 sim2real deploy reference
