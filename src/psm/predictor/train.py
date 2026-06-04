import csv
import os
import torch
from datetime import datetime

from .psm_predictor import PsmPredictor
from .utils import (
    load_motion_data_npz,
    prepare_training_data,
    save_metadata,
    save_config_yaml,
    create_loss_weights,
    create_feature_regularization_weights,
    get_mirror_indices_scales,
    mirror_lower_sequence_batch,
    make_cmd_flip_signs,
    make_body_vel_flip_signs,
)
from .config import (
    HISTORY_HORIZON,
    PREDICTION_HORIZON,
    N_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    HIDDEN_SIZE,
    DEVICE,
    MOTION_FILES_PATTERN,
    LOGS_DIR,
    CHECKPOINT_SAVE_INTERVAL,
    CHECKPOINT_LOG_INTERVAL,
    AMSGRAD,
    MIRROR_DATA,
    SYMMETRY_SPEC,
    NOISE_STD,
    FEET_BODIES,
    LOSS_WEIGHTS,
    USE_WEIGHTED_LOSS,
    LOG_PER_FEATURE_LOSS_INTERVAL,
    ENABLE_SYMMETRY_LOSS,
    SYMMETRY_LOSS_WEIGHT,
    SYMMETRY_LOSS_RAMP_EPOCHS,
    ENABLE_FEATURE_REGULARIZATION,
    FEATURE_REGULARIZATION_CONFIG,
    FEATURE_REGULARIZATION_WEIGHT,
    WEIGHT_DECAY,
    GRAD_CLIP_NORM,
    ENCODER_HIDDEN_SIZE,
    GRU_NUM_LAYERS,
    HEAD_HIDDEN_DEPTH,
    HEAD_DROPOUT,
    ACTIVATION,
    UPPER_JOINT_NAMES,
    LOWER_JOINT_NAMES,
    CMD_TRAJ_HORIZONS,
    CMD_TRAJ_YAW_FRAME_DELTAS,
    HISTORY_INPUT_MODE,
    JOINTS_HISTORY_WEIGHT,
    FEET_HISTORY_WEIGHT,
    USE_LOWER_JOINT_VELOCITY,
    USE_FOOT_VELOCITY,
)


def create_log_dir():
    """Create log directory with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(LOGS_DIR, timestamp)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")

    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Created log directory: {log_dir}")

    return log_dir, checkpoint_dir


_TRAINING_HISTORY_FIELDS = (
    "epoch",
    "lr",
    "total_loss",
    "recon_loss",
    "sym_loss",
    "sym_weight",
    "sym_mae",
    "reg_loss",
    "vel_smooth_loss",
    "acc_smooth_loss",
)


def _append_training_history_csv(
    path: str,
    row: dict[str, float | int | str],
) -> None:
    """Append one row to checkpoints/training_history.csv (creates file + header if needed)."""
    new_file = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_TRAINING_HISTORY_FIELDS), extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow({k: row[k] for k in _TRAINING_HISTORY_FIELDS})


def train_model():
    """Train the arm matching model (weighted lower history + GRU/Conv/MLP encoder)."""
    if str(DEVICE).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    print(f"Loading motion data (npz, mirroring: {MIRROR_DATA})")
    use_lower_joint_velocity = bool(USE_LOWER_JOINT_VELOCITY)
    use_foot_velocity = bool(USE_FOOT_VELOCITY)
    history_input_mode = str(HISTORY_INPUT_MODE).lower()
    if history_input_mode not in ("joints", "feet", "both"):
        raise ValueError(
            f"HISTORY_INPUT_MODE must be 'joints', 'feet', or 'both', got {history_input_mode!r}"
        )
    use_lower_enc = history_input_mode in ("joints", "both")
    use_foot_enc = history_input_mode in ("feet", "both")
    if use_lower_joint_velocity and not use_lower_enc:
        raise ValueError(
            "USE_LOWER_JOINT_VELOCITY requires HISTORY_INPUT_MODE 'joints' or 'both' "
            f"(got {history_input_mode!r})"
        )
    if use_foot_velocity and not use_foot_enc:
        raise ValueError(
            "USE_FOOT_VELOCITY requires HISTORY_INPUT_MODE 'feet' or 'both' "
            f"(got {history_input_mode!r})"
        )
    joints_history_weight = float(JOINTS_HISTORY_WEIGHT)
    feet_history_weight = float(FEET_HISTORY_WEIGHT)
    history_recency_decay = 0.0
    encoder_type = "gru"
    data = load_motion_data_npz(
        MOTION_FILES_PATTERN,
        upper_joint_names=UPPER_JOINT_NAMES,
        lower_joint_names=LOWER_JOINT_NAMES,
        feet_bodies=(FEET_BODIES[0], FEET_BODIES[1]),
        cmd_traj_horizons=CMD_TRAJ_HORIZONS,
        cmd_traj_yaw_frame_deltas=CMD_TRAJ_YAW_FRAME_DELTAS,
    )

    print("Preparing training data...")
    training_data = prepare_training_data(
        data,
        HISTORY_HORIZON,
        PREDICTION_HORIZON,
        DEVICE,
        use_lower_joint_velocity=use_lower_joint_velocity,
        use_foot_velocity=use_foot_velocity,
        symmetry_spec=SYMMETRY_SPEC,
    )

    num_lower = data["lower_joints"].shape[1]
    num_upper = data["upper_joints"].shape[1]
    num_body_features = training_data["num_body_features"]

    out_dim = (num_upper + num_body_features) * PREDICTION_HORIZON

    print(f"Output dimension: {out_dim}")
    print(
        f"Encoder: {encoder_type}, lower_joint_vel={use_lower_joint_velocity}, "
        f"foot_vel={use_foot_velocity}, recency_decay={history_recency_decay}"
    )
    print(f"History input mode: {history_input_mode} (joint_w={joints_history_weight}, feet_w={feet_history_weight})")
    print(f"Lower joints used ({len(data['lower_joint_names'])}): {data['lower_joint_names']}")
    print(f"Body features: {training_data['body_feature_names']}")
    print(f"Command features: {training_data['cmd_feature_names']}")

    log_dir, checkpoint_dir = create_log_dir()

    predictor = PsmPredictor(
        output_size=out_dim,
        y_mean=training_data["y_mean"],
        y_std=training_data["y_std"],
        leg_pos_mean=training_data["leg_pos_mean"],
        leg_pos_std=training_data["leg_pos_std"],
        foot_pos_mean=training_data["foot_pos_mean"],
        foot_pos_std=training_data["foot_pos_std"],
        body_vel_mean=training_data["body_vel_mean"],
        body_vel_std=training_data["body_vel_std"],
        cmd_mean=training_data["cmd_mean"],
        cmd_std=training_data["cmd_std"],
        num_lower=num_lower,
        history_horizon=HISTORY_HORIZON,
        prediction_horizon=PREDICTION_HORIZON,
        cmd_feature_dim=int(training_data["cmd_features"].shape[1]),
        history_input_mode=history_input_mode,
        joints_history_weight=joints_history_weight,
        feet_history_weight=feet_history_weight,
        use_lower_joint_velocity=use_lower_joint_velocity,
        use_foot_velocity=use_foot_velocity,
        leg_vel_mean=training_data["leg_vel_mean"]
        if use_lower_joint_velocity
        else None,
        leg_vel_std=training_data["leg_vel_std"]
        if use_lower_joint_velocity
        else None,
        foot_vel_mean=training_data["foot_vel_mean"]
        if use_foot_velocity
        else None,
        foot_vel_std=training_data["foot_vel_std"]
        if use_foot_velocity
        else None,
        history_recency_decay=history_recency_decay,
        encoder_type=encoder_type,
        encoder_hidden_size=ENCODER_HIDDEN_SIZE,
        hidden_size=HIDDEN_SIZE,
        gru_num_layers=GRU_NUM_LAYERS,
        head_hidden_depth=HEAD_HIDDEN_DEPTH,
        head_dropout=HEAD_DROPOUT,
        activation=ACTIVATION,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=LEARNING_RATE,
        amsgrad=AMSGRAD,
        weight_decay=WEIGHT_DECAY,
    )

    if USE_WEIGHTED_LOSS:
        loss_weights = create_loss_weights(
            upper_joint_names=data["upper_joint_names"],
            body_feature_names=training_data["body_feature_names"],
            prediction_horizon=PREDICTION_HORIZON,
            loss_weight_config=LOSS_WEIGHTS,
            device=DEVICE,
        )
    else:
        loss_weights = None

    if ENABLE_FEATURE_REGULARIZATION and FEATURE_REGULARIZATION_CONFIG:
        feature_reg_weights = (
            create_feature_regularization_weights(
                upper_joint_names=data["upper_joint_names"],
                body_feature_names=training_data["body_feature_names"],
                prediction_horizon=PREDICTION_HORIZON,
                regularization_config=FEATURE_REGULARIZATION_CONFIG,
                device=DEVICE,
            )
            * FEATURE_REGULARIZATION_WEIGHT
        )
    else:
        feature_reg_weights = None

    ctor = {
        "output_size": out_dim,
        "y_mean": training_data["y_mean"],
        "y_std": training_data["y_std"],
        "leg_pos_mean": training_data["leg_pos_mean"],
        "leg_pos_std": training_data["leg_pos_std"],
        "foot_pos_mean": training_data["foot_pos_mean"],
        "foot_pos_std": training_data["foot_pos_std"],
        "body_vel_mean": training_data["body_vel_mean"],
        "body_vel_std": training_data["body_vel_std"],
        "cmd_mean": training_data["cmd_mean"],
        "cmd_std": training_data["cmd_std"],
        "num_lower": num_lower,
        "history_horizon": HISTORY_HORIZON,
        "prediction_horizon": PREDICTION_HORIZON,
        "cmd_feature_dim": int(training_data["cmd_features"].shape[1]),
        "history_input_mode": history_input_mode,
        "joints_history_weight": joints_history_weight,
        "feet_history_weight": feet_history_weight,
        # Alias read by ``psm`` env (same as prediction_horizon).
        "prediction_h": PREDICTION_HORIZON,
        "use_lower_joint_velocity": use_lower_joint_velocity,
        "use_foot_velocity": use_foot_velocity,
        "use_lower_velocity": use_lower_joint_velocity,
        "leg_vel_mean": training_data["leg_vel_mean"]
        if use_lower_joint_velocity
        else None,
        "leg_vel_std": training_data["leg_vel_std"]
        if use_lower_joint_velocity
        else None,
        "foot_vel_mean": training_data["foot_vel_mean"]
        if use_foot_velocity
        else None,
        "foot_vel_std": training_data["foot_vel_std"]
        if use_foot_velocity
        else None,
        "history_recency_decay": history_recency_decay,
        "encoder_type": encoder_type,
        "encoder_hidden_size": ENCODER_HIDDEN_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "gru_num_layers": GRU_NUM_LAYERS,
        "head_hidden_depth": HEAD_HIDDEN_DEPTH,
        "head_dropout": HEAD_DROPOUT,
        "activation": ACTIVATION,
    }

    metadata = {
        "constructor_params": ctor,
        "horizon": HISTORY_HORIZON,
        "use_lower_joint_velocity": use_lower_joint_velocity,
        "use_foot_velocity": use_foot_velocity,
        "use_lower_velocity": use_lower_joint_velocity,
        "encoder_type": encoder_type,
        "history_recency_decay": history_recency_decay,
        "history_input_mode": history_input_mode,
        "joints_history_weight": joints_history_weight,
        "feet_history_weight": feet_history_weight,
        "lower_order": data["lower_joint_names"],
        "upper_order": data["upper_joint_names"],
        "body_feature_names": training_data["body_feature_names"],
        "body_vel_names": training_data["body_vel_names"],
        "cmd_feature_names": training_data["cmd_feature_names"],
        "cmd_traj_horizons": list(training_data["cmd_traj_horizons"]),
        "cmd_traj_yaw_frame_deltas": bool(training_data["cmd_traj_yaw_frame_deltas"]),
        "body_vel_future_names": training_data["cmd_feature_names"],
        "feet_bodies": list(FEET_BODIES),
        "root_height_mean": float(data.get("root_height_mean", 0.0)),
    }

    config_yaml = {
        "log_info": {"timestamp": datetime.now().isoformat(), "log_dir": log_dir},
        "model_info": {
            "horizon": HISTORY_HORIZON,
            "output_dim": out_dim,
            "encoder_type": encoder_type,
            "use_lower_joint_velocity": use_lower_joint_velocity,
            "use_foot_velocity": use_foot_velocity,
            "use_lower_velocity": use_lower_joint_velocity,
            "history_recency_decay": history_recency_decay,
        },
        "training_params": {
            "HISTORY_HORIZON": HISTORY_HORIZON,
            "PREDICTION_HORIZON": PREDICTION_HORIZON,
            "N_EPOCHS": N_EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "LEARNING_RATE": LEARNING_RATE,
            "HIDDEN_SIZE": HIDDEN_SIZE,
            "ENCODER_HIDDEN_SIZE": ENCODER_HIDDEN_SIZE,
            "ENCODER_TYPE": encoder_type,
            "GRU_NUM_LAYERS": GRU_NUM_LAYERS,
            "HEAD_HIDDEN_DEPTH": HEAD_HIDDEN_DEPTH,
            "HEAD_DROPOUT": HEAD_DROPOUT,
            "ACTIVATION": ACTIVATION,
            "LOWER_JOINT_NAMES": LOWER_JOINT_NAMES,
            "HISTORY_RECENCY_DECAY": history_recency_decay,
            "USE_LOWER_JOINT_VELOCITY": use_lower_joint_velocity,
            "USE_FOOT_VELOCITY": use_foot_velocity,
            "HISTORY_INPUT_MODE": history_input_mode,
            "HISTORY_USE_LOWER_ENC": use_lower_enc,
            "HISTORY_USE_FOOT_ENC": use_foot_enc,
            "JOINT_HISTORY_WEIGHT": joints_history_weight,
            "FEET_HISTORY_WEIGHT": feet_history_weight,
            "AMSGRAD": AMSGRAD,
            "MIRROR_DATA": MIRROR_DATA,
            "NOISE_STD": NOISE_STD,
            "USE_WEIGHTED_LOSS": USE_WEIGHTED_LOSS,
            "LOSS_WEIGHTS": LOSS_WEIGHTS if USE_WEIGHTED_LOSS else {},
            "ENABLE_SYMMETRY_LOSS": ENABLE_SYMMETRY_LOSS,
            "SYMMETRY_LOSS_WEIGHT": SYMMETRY_LOSS_WEIGHT,
            "SYMMETRY_LOSS_RAMP_EPOCHS": SYMMETRY_LOSS_RAMP_EPOCHS,
            "ENABLE_FEATURE_REGULARIZATION": ENABLE_FEATURE_REGULARIZATION,
            "FEATURE_REGULARIZATION_CONFIG": FEATURE_REGULARIZATION_CONFIG
            if ENABLE_FEATURE_REGULARIZATION
            else {},
            "FEATURE_REGULARIZATION_WEIGHT": FEATURE_REGULARIZATION_WEIGHT,
        },
        "data_info": {
            "training_data_files": [os.path.basename(f) for f in data["motion_files"]],
            "lower_joint_names": data["lower_joint_names"],
            "upper_joint_names": data["upper_joint_names"],
            "body_feature_names": training_data["body_feature_names"],
            "body_vel_names": training_data["body_vel_names"],
            "cmd_feature_names": training_data["cmd_feature_names"],
            "feet_bodies": list(FEET_BODIES),
            "root_height_mean": float(data.get("root_height_mean", 0.0)),
        },
    }

    metadata_path = os.path.join(log_dir, "metadata.pkl")
    config_yaml_path = os.path.join(log_dir, "config.yaml")
    history_csv_path = os.path.join(checkpoint_dir, "training_history.csv")

    (
        output_indices,
        output_signs,
        lower_mirror_indices,
        lower_mirror_signs,
    ) = get_mirror_indices_scales(
        data["lower_joint_names"],
        data["upper_joint_names"],
        training_data["body_feature_names"],
        HISTORY_HORIZON,
        PREDICTION_HORIZON,
        DEVICE,
        SYMMETRY_SPEC,
    )

    bv_flip = make_body_vel_flip_signs(training_data["body_vel_names"], device=DEVICE).view(
        1, 1, -1
    )
    cmd_flip = make_cmd_flip_signs(training_data["cmd_feature_names"], device=DEVICE).view(1, -1)

    def mirror_body_vel_hist(v: torch.Tensor) -> torch.Tensor:
        return v * bv_flip

    def mirror_body_vel_future(v: torch.Tensor) -> torch.Tensor:
        return v * cmd_flip

    def mirror_foot_hist(v: torch.Tensor) -> torch.Tensor:
        # [Lx,Ly,Lz,Rx,Ry,Rz] -> [Rx,-Ry,Rz,Lx,-Ly,Lz]
        o = v.clone()
        o[..., 0] = v[..., 3]
        o[..., 1] = -v[..., 4]
        o[..., 2] = v[..., 5]
        o[..., 3] = v[..., 0]
        o[..., 4] = -v[..., 1]
        o[..., 5] = v[..., 2]
        return o

    def symmetry_error_fn(
        lp,
        fp,
        lv,
        fv,
        bvh,
        bvf,
    ):
        pred_original = predictor(lp, fp, bvh, bvf, lv, fv)
        lp_m = mirror_lower_sequence_batch(lp, lower_mirror_indices, lower_mirror_signs)
        fp_m = mirror_foot_hist(fp) if fp is not None else None
        lv_m = (
            mirror_lower_sequence_batch(lv, lower_mirror_indices, lower_mirror_signs)
            if lv is not None
            else None
        )
        fv_m = mirror_foot_hist(fv) if fv is not None else None
        bvh_m = mirror_body_vel_hist(bvh)
        bvf_m = mirror_body_vel_future(bvf)
        pred_mirrored = predictor(lp_m, fp_m, bvh_m, bvf_m, lv_m, fv_m)
        pred_original_mirrored = pred_original[:, output_indices] * output_signs
        return pred_original_mirrored - pred_mirrored

    leg_pos_std = training_data["leg_pos_std"].view(1, 1, -1)
    bv_std = training_data["body_vel_std"].view(1, 1, -1)
    foot_vel_std_view = (
        training_data["foot_vel_std"].view(1, 1, -1)
        if use_foot_velocity and training_data["foot_vel_std"] is not None
        else None
    )

    print("Starting training...")
    for epoch in range(N_EPOCHS):
        original_batch_size = BATCH_SIZE // 2 if MIRROR_DATA else BATCH_SIZE
        motion_idx = (
            torch.randint(0, 2**63 - 1, size=(original_batch_size,), device=DEVICE)
            % training_data["motion_offsets"].shape[0]
        )
        span = (
            training_data["motion_lengths"][motion_idx]
            - HISTORY_HORIZON
            - PREDICTION_HORIZON
        )
        span = torch.clamp(span, min=1)
        start_indices = (
            (
                training_data["motion_offsets"][motion_idx]
                + torch.randint(0, 2**63 - 1, size=(original_batch_size,), device=DEVICE)
                % span
            ).unsqueeze(1)
            + torch.arange(0, HISTORY_HORIZON, step=1, device=DEVICE)
        ).int()

        lower_pos = training_data["lower_joints"][start_indices]
        lower_vel = (
            training_data["lower_joint_vel"][start_indices]
            if use_lower_joint_velocity and use_lower_enc
            else None
        )
        foot_vel = (
            training_data["foot_vel_hist"][start_indices]
            if use_foot_velocity and use_foot_enc
            else None
        )
        foot_pos_hist = training_data["foot_pos_hist"][start_indices]
        base_timestep = start_indices[:, -1]

        future_steps = torch.stack(
            [base_timestep + t + 1 for t in range(PREDICTION_HORIZON)], dim=1
        )

        body_vel_hist = training_data["body_vel"][start_indices]

        body_vel_future = training_data["cmd_features"][base_timestep]

        upper_future = training_data["upper_joints"][future_steps].reshape(
            original_batch_size, -1
        )
        body_future = training_data["body_features"][future_steps].reshape(
            original_batch_size, -1
        )
        output_batch = torch.cat([upper_future, body_future], dim=1).clone()

        if MIRROR_DATA:
            bvh_m = mirror_body_vel_hist(body_vel_hist)
            bvf_m = mirror_body_vel_future(body_vel_future)
            if use_lower_enc:
                lp_m = mirror_lower_sequence_batch(
                    lower_pos, lower_mirror_indices, lower_mirror_signs
                )
                lower_pos = torch.cat([lower_pos, lp_m], dim=0)
            if use_foot_enc:
                fp_m = mirror_foot_hist(foot_pos_hist)
                foot_pos_hist = torch.cat([foot_pos_hist, fp_m], dim=0)
            if use_lower_joint_velocity and lower_vel is not None:
                lv_m = mirror_lower_sequence_batch(
                    lower_vel, lower_mirror_indices, lower_mirror_signs
                )
                lower_vel = torch.cat([lower_vel, lv_m], dim=0)
            if use_foot_velocity and foot_vel is not None:
                fv_m = mirror_foot_hist(foot_vel)
                foot_vel = torch.cat([foot_vel, fv_m], dim=0)
            body_vel_hist = torch.cat([body_vel_hist, bvh_m], dim=0)
            body_vel_future = torch.cat([body_vel_future, bvf_m], dim=0)
            output_batch = torch.cat(
                [output_batch, output_batch[:, output_indices] * output_signs[None, :]],
                dim=0,
            )

        clean_lp = clean_fp = clean_bvh = clean_bvf = None
        clean_lv = clean_fv = None
        if ENABLE_SYMMETRY_LOSS:
            if use_lower_enc:
                clean_lp = lower_pos[:original_batch_size].clone()
            if use_foot_enc:
                clean_fp = foot_pos_hist[:original_batch_size].clone()
            clean_bvh = body_vel_hist[:original_batch_size].clone()
            clean_bvf = body_vel_future[:original_batch_size].clone()
            if lower_vel is not None:
                clean_lv = lower_vel[:original_batch_size].clone()
            if foot_vel is not None:
                clean_fv = foot_vel[:original_batch_size].clone()

        if NOISE_STD > 0:
            if use_lower_enc:
                lower_pos = lower_pos + torch.randn_like(lower_pos) * NOISE_STD * leg_pos_std
            if use_foot_enc:
                foot_pos_hist = foot_pos_hist + torch.randn_like(foot_pos_hist) * NOISE_STD * training_data["foot_pos_std"].view(1, 1, -1)
            if use_foot_velocity and foot_vel is not None and foot_vel_std_view is not None:
                foot_vel = foot_vel + torch.randn_like(foot_vel) * NOISE_STD * foot_vel_std_view
            body_vel_hist = body_vel_hist + torch.randn_like(body_vel_hist) * NOISE_STD * bv_std
            body_vel_future = body_vel_future + torch.randn_like(body_vel_future) * NOISE_STD * training_data["cmd_std"].view(1, -1)

        lower_in = lower_pos if use_lower_enc else None
        foot_in = foot_pos_hist if use_foot_enc else None
        predictions = predictor(lower_in, foot_in, body_vel_hist, body_vel_future, lower_vel, foot_vel)

        squared_errors = (predictions - output_batch).square()
        if USE_WEIGHTED_LOSS and loss_weights is not None:
            reconstruction_loss = (squared_errors * loss_weights).mean()
        else:
            reconstruction_loss = squared_errors.mean()

        if ENABLE_SYMMETRY_LOSS:
            sym_err = symmetry_error_fn(clean_lp, clean_fp, clean_lv, clean_fv, clean_bvh, clean_bvf)
            sym_loss = torch.mean(sym_err.square())
        else:
            sym_loss = torch.tensor(0.0, device=DEVICE)

        if ENABLE_FEATURE_REGULARIZATION and feature_reg_weights is not None:
            reg_loss = (predictions.square() * feature_reg_weights).mean()
        else:
            reg_loss = torch.tensor(0.0, device=DEVICE)

        vel_smooth_loss = torch.tensor(0.0, device=DEVICE)
        acc_smooth_loss = torch.tensor(0.0, device=DEVICE)

        total_loss = reconstruction_loss
        if ENABLE_SYMMETRY_LOSS:
            ramp = (
                1.0
                if SYMMETRY_LOSS_RAMP_EPOCHS <= 0
                else min(1.0, float(epoch) / float(SYMMETRY_LOSS_RAMP_EPOCHS))
            )
            sym_weight = SYMMETRY_LOSS_WEIGHT * ramp
            total_loss = total_loss + sym_weight * sym_loss
        else:
            sym_weight = 0.0
        if ENABLE_FEATURE_REGULARIZATION and feature_reg_weights is not None:
            total_loss = total_loss + reg_loss
        # Temporal losses are intentionally disabled.

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        should_log = (epoch % CHECKPOINT_LOG_INTERVAL == 0) or (epoch == N_EPOCHS - 1)

        if should_log:
            parts = [
                f"Total: {total_loss.item():.6f}",
                f"Recon: {reconstruction_loss.item():.6f}",
            ]
            sym_mae_val = 0.0
            if ENABLE_SYMMETRY_LOSS:
                sym_mae_val = sym_err.abs().mean().item()
                parts.append(f"SymMSE: {sym_loss.item():.6f}")
                parts.append(f"SymMAE: {sym_mae_val:.6f}")
                parts.append(f"SymW: {sym_weight:.4f}")
            if ENABLE_FEATURE_REGULARIZATION and feature_reg_weights is not None:
                parts.append(f"Reg: {reg_loss.item():.6f}")
            print(f"Epoch {epoch}, " + ", ".join(parts))

            history_row: dict[str, float | int | str] = {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "total_loss": float(total_loss.item()),
                "recon_loss": float(reconstruction_loss.item()),
                "sym_loss": float(sym_loss.item()),
                "sym_weight": float(sym_weight),
                "sym_mae": float(sym_mae_val),
                "reg_loss": float(reg_loss.item()),
                "vel_smooth_loss": float(vel_smooth_loss.item()),
                "acc_smooth_loss": float(acc_smooth_loss.item()),
            }
            _append_training_history_csv(history_csv_path, history_row)

            if (
                LOG_PER_FEATURE_LOSS_INTERVAL > 0
                and epoch % LOG_PER_FEATURE_LOSS_INTERVAL == 0
                and epoch > 0
            ):
                with torch.no_grad():
                    per_feature = squared_errors.mean(dim=0)
                    print(f"\n=== Per-Feature Loss (Epoch {epoch}) ===")
                    print("Upper joints at t=0:")
                    for i, name in enumerate(data["upper_joint_names"]):
                        print(f"  {name:30s}  MSE: {per_feature[i].item():.6f}")
                    offset = len(data["upper_joint_names"]) * PREDICTION_HORIZON
                    print("Body features at t=0:")
                    for i, name in enumerate(training_data["body_feature_names"]):
                        print(f"  {name:30s}  MSE: {per_feature[offset + i].item():.6f}")
                    print()

        if epoch % CHECKPOINT_SAVE_INTERVAL == 0:
            state = predictor.state_dict()
            checkpoint_path = os.path.join(checkpoint_dir, f"predictor_{epoch}.pth")
            torch.save(state, checkpoint_path)
            torch.save(state, os.path.join(log_dir, "predictor.pth"))
            save_metadata(metadata, metadata_path)
            save_config_yaml(config_yaml, config_yaml_path)
            print(f"Saved checkpoint: {checkpoint_path} (also {os.path.join(log_dir, 'predictor.pth')})")

    final_predictor_path = os.path.join(log_dir, "predictor.pth")
    torch.save(predictor.state_dict(), final_predictor_path)
    save_metadata(metadata, metadata_path)
    save_config_yaml(config_yaml, config_yaml_path)
    print("Training completed! Final model saved:", final_predictor_path)
    print(f"Log directory: {log_dir}")


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
