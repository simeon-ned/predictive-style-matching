# Motion data

Place LAFAN-style NPZ clips under `motions/`.

Training data may be tracked with Git LFS. To fetch:

```bash
bash src/psm/scripts/lfs_pull_data.sh
```

Or copy your own `*.npz` files here (see `psm.predictor.utils.load_motion_data_npz` for required keys).
