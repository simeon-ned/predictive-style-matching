# Motion data

## Quick start (recommended)

```bash
# 1. CSV → extended NPZ (FK + PSM features, GPU)
psm-csv-to-npz --dataset motions --robot g1

# Or augment existing compact NPZ clips in place:
psm-augment-npz --input-path data/motions

# 2. Stack clips into one training bundle
psm-stack-motions --dataset-path data/motions

# 3. Train (loads data/motions/motions.npz in seconds)
psm-predictor-train
```

## Layout

| Path | Role |
|------|------|
| `data/motions/raw/` | Source CSV clips (optional) |
| `data/motions/*.npz` | Per-clip extended NPZ (`psm_*` keys + kinematics) |
| `data/motions/motions.npz` | Stacked training bundle (`segment_start_idx`, …) |

## Extended NPZ schema

Each clip NPZ includes kinematics plus precomputed predictor arrays:

- **Kinematics:** `joint_pos`, `qpos`, `body_pos_w`, `body_pos_r`, …
- **PSM training:** `psm_lower_joints`, `psm_upper_joints`, `psm_foot_pos_hist`, `psm_body_features`, `psm_cmd_features`, …

Feature config is baked in at conversion time (`psm.predictor.config` joint lists and `CMD_TRAJ_HORIZONS`). Re-run conversion if you change those.

## Legacy compact NPZ

Clips without `psm_*` keys still load via the slow path (`npz_schema` expansion + runtime feature compute). Use `psm-augment-npz` to upgrade them.

Fetch LFS data:

```bash
bash src/psm/scripts/lfs_pull_data.sh
```
