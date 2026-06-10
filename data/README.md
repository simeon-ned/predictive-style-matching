# Motion data

## Quick start (recommended)

```bash
# CSV or GMR PKL → extended NPZ (FK + PSM features, GPU)
psm-data-to-npz --input-path data/seed --output-dir data/motions

# Or augment existing compact NPZ clips in place:
psm-augment-npz --input-path data/motions

# Train (auto-builds motions.npz from clips if needed)
psm-predictor-train

# Visualize a converted clip (Viser browser viewer)
psm-vis-npz --npz data/motions/your_clip.npz
psm-vis-npz --motion-dir data/motions
```

## Input formats (`psm-data-to-npz`)

| Format | Layout |
|--------|--------|
| **CSV** | `root_pos(3) + root_rot_xyzw(4) + joint_pos(n_dof)` per row, no header. Default input rate 30 Hz. |
| **PKL (GMR)** | Dict with `fps`, `root_pos (T×3)`, `root_rot (T×4, xyzw)`, `dof_pos (T×n_dof)`. Optional: `joint_names`, `dof_joint_names`, or `joint_order`. |

Both are resampled to 50 Hz and run through G1 FK before export.

## Layout

| Path | Role |
|------|------|
| `data/motions/raw/` | Source CSV/PKL clips (optional) |
| `data/motions/*.npz` | Per-clip extended NPZ (`psm_*` keys + kinematics) |
| `data/motions/motions.npz` | Stacked training bundle (`segment_start_idx`, …) |

## Extended NPZ schema

Each clip NPZ includes kinematics plus precomputed predictor arrays:

- **Kinematics:** `joint_pos`, `qpos`, `body_pos_w`, `body_pos_r`, …
- **PSM training:** `psm_lower_joints`, `psm_upper_joints`, `psm_foot_pos_hist`, `psm_body_features`, `psm_cmd_features`, …

Feature config is baked in at conversion time (`psm.predictor.config` joint lists and `CMD_TRAJ_HORIZONS`). Re-run conversion if you change those.

## Legacy compact NPZ

Clips without `psm_*` keys are upgraded automatically when you run `psm-predictor-train`, or manually via `psm-augment-npz`.

## Git

Motion files under `data/` are **gitignored** (not pushed). Only `data/README.md` and `.gitkeep` placeholders are tracked. Add your own CSV/PKL clips locally.
