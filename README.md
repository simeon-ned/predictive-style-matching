# Predictive Style Matching (PSM)

Code for *Predictive Style Matching: Natural and Robust Humanoid Locomotion* on Unitree G1.

**Project website:** [https://simeon-ned.github.io/predictive-style-matching/](https://simeon-ned.github.io/predictive-style-matching/) (sources in [`docs/`](docs/))

## Install

```bash
git clone https://github.com/simeon-ned/predictive-style-matching.git
cd predictive-style-matching
pip install -e .
```

## Quick start

**Predictor (offline)**

```bash
psm-predictor-train
psm-predictor-play --npz data/motions/your_clip.npz
```

**RL (mjlab)**

```bash
python scripts/train.py Psm-G1 --env.scene.num-envs=4096
python scripts/play.py Psm-G1
```

Deploy sim2real via [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab); export ONNX from play (`logs/.../params/latest.onnx`).

## Repository layout

```text
├── README.md           # this file — code & usage
├── pyproject.toml
├── scripts/
├── data/
├── src/psm/
│   ├── predictor/      # PsmPredictor training
│   └── env/            # RL environment (Psm-G1)
└── docs/               # GitHub Pages project site (not Python)
    ├── index.html
    └── static/
```

Paper TeX may live in a separate private repo during review; figures for the site are under `docs/static/images/`.
