# Motion data

Place motion NPZ clips for PSM predictor training under `motions/`.

Training data may be tracked with Git LFS. To fetch:

```bash
bash src/psm/scripts/lfs_pull_data.sh
```

## NPZ format

Predictor training accepts **compact** per-clip NPZ files with at least:

- `joint_names`, `joint_pos`, `joint_vel` (optional)
- `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`
- `fps`, `robot` (optional)

**Full** NPZ files (with `qpos`, `body_pos_r`, …) are also supported. The loader derives missing fields automatically (`psm.predictor.npz_schema`).

Default training glob: `data/motions/*.npz` (see `psm.predictor.config.MOTION_FILES_PATTERN`).
