"""Configuration for PSM predictor training and inference."""

# Training hyperparameters
# --- Small / medium dataset preset ---
# Shorter horizons shrink the output head (fewer parameters). Smaller hidden
# sizes + dropout + noise + symmetry reduce overfitting. Lower LR and batch
# give stabler updates when each epoch sees the same clips often.
# If you later get *lots* of motion data, raise HIDDEN_SIZE / horizons toward 256 / 40.
HISTORY_HORIZON = 25
PREDICTION_HORIZON = 10
N_EPOCHS = 45000
BATCH_SIZE = 512
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128

# Device configuration
DEVICE = "cuda"

# Data paths (NPZ motion clips for predictor training).
MOTION_FILES_PATTERN = "data/motions/*.npz"

# NPZ schema: compact per-clip export and optional full fields (``qpos``, ``body_pos_r``, …).
# Minimum keys: joint_names, joint_pos, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w.
# ``load_motion_data_npz`` derives qpos/qvel and root-frame body arrays when absent (see npz_schema.py).
ROOT_BODY_NAME = "pelvis"
# Link body order when ``body_names`` is omitted from NPZ (None = infer from ``robot`` / G1 asset).
MOTION_BODY_NAMES = None

# Logging configuration
# Offline predictor runs: logs/predictor/<YYYY-MM-DD_HH-MM-SS>/
LOGS_DIR = "logs/predictor"

# Data augmentation
MIRROR_DATA = True  # Enable left/right mirroring to improve symmetry
SYMMETRY_SPEC = [
    # Mirroring across sagittal plane for G1 joints (see ``psm.assets.unitree_g1``).
    #
    # Signs are derived from joint axes in ``psm/assets/unitree_g1/xmls/g1.xml``:
    # - axis along X or Z flips sign under mirroring; axis along Y keeps sign.
    ("left_hip_pitch_joint", "right_hip_pitch_joint", 1),   # axis (0,1,0)
    ("left_hip_roll_joint", "right_hip_roll_joint", -1),    # axis (1,0,0)
    ("left_hip_yaw_joint", "right_hip_yaw_joint", -1),      # axis (0,0,1)
    ("left_knee_joint", "right_knee_joint", 1),             # axis (0,1,0)
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint", 1),  # axis (0,1,0)
    ("left_ankle_roll_joint", "right_ankle_roll_joint", -1),   # axis (1,0,0)

    ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint", 1),  # axis (0,1,0)
    ("left_shoulder_roll_joint", "right_shoulder_roll_joint", -1),   # axis (1,0,0)
    ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint", -1),     # axis (0,0,1)
    ("left_elbow_joint", "right_elbow_joint", 1),                    # axis (0,1,0)

    ("left_wrist_roll_joint", "right_wrist_roll_joint", -1),         # axis (1,0,0)
    ("left_wrist_pitch_joint", "right_wrist_pitch_joint", 1),        # axis (0,1,0)
    ("left_wrist_yaw_joint", "right_wrist_yaw_joint", -1),           # axis (0,0,1)

    # Center joints (no swapping, just sign if needed)
    ("waist_yaw_joint", "waist_yaw_joint", -1),    # axis (0,0,1)
    ("waist_roll_joint", "waist_roll_joint", -1),  # axis (1,0,0)
    ("waist_pitch_joint", "waist_pitch_joint", 1), # axis (0,1,0)

    # Body features
    ("left_foot_pitch", "right_foot_pitch", 1),
    ("left_foot_rel_yaw", "right_foot_rel_yaw", -1),
    ("step_length", "step_length", 1),
    ("step_width", "step_width", 1),
    ("cadence_hz", "cadence_hz", 1),
    ("double_support_factor", "double_support_factor", 1),
    ("root_height", "root_height", 1),
    ("pelvis_roll", "pelvis_roll", -1),
    ("pelvis_pitch", "pelvis_pitch", 1),
    ("torso_roll", "torso_roll", -1),
    ("torso_pitch", "torso_pitch", 1),
]

# Checkpoint configuration
CHECKPOINT_SAVE_INTERVAL = 1000
CHECKPOINT_LOG_INTERVAL = 1000

# Model configuration
AMSGRAD = False

# GRU hidden size (also used as fused embedding dim before the output MLP).
ENCODER_HIDDEN_SIZE = 128

# Stacked GRU layers (1 = original single-layer GRU). Deeper stacks increase
# capacity but can overfit small datasets.
GRU_NUM_LAYERS = 1

# Fusion MLP after concatenating encoder + body-vel history + future block.
# Depth = number of hidden Linear+activation blocks before the final projection.
HEAD_HIDDEN_DEPTH = 2
HEAD_DROPOUT = 0

# Nonlinearity for encoder conv/MLP tails and fusion head: relu | gelu | silu
ACTIVATION = "gelu"

# Command conditioning features for the predictor input:
# [root_vx, root_vy, root_wz] + trajectory features in current root yaw frame.
# Horizons are in frames (50 Hz default data => 8/16/24 ~ 0.16/0.32/0.48s).
CMD_TRAJ_HORIZONS = (15, 30, 45)
CMD_TRAJ_YAW_FRAME_DELTAS = False

# Data augmentation (stronger input noise helps small datasets; reduce to ~0.01 if training gets unstable).
NOISE_STD = 0.01

# Bodies used for computing "foot" features in NPZ logs (MuJoCo body names).
FEET_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]

# What goes into the GRU encoder history: lower-body joint angles, root-relative foot
# positions (6 = Lxyz+Rxyz), or both. Must match training metadata at inference / RL.
HISTORY_INPUT_MODE = "both"  # "joints" | "feet" | "both"
# Relative scaling of each history branch before GRU encoding.
JOINTS_HISTORY_WEIGHT = 1.0
FEET_HISTORY_WEIGHT = 1.0

# Concatenate normalized velocities into the GRU input (in addition to positions).
# Joints: lower-body joint rates (modes ``joints`` / ``both``). Feet: root-frame foot
# velocities from finite differences of foot positions (modes ``feet`` / ``both``).
USE_LOWER_JOINT_VELOCITY = False
USE_FOOT_VELOCITY = False

# --- Robot / NPZ joint layout ---
# Names must match `joint_names` in your NPZ (e.g. LAFAN export). Upper = predicted targets;
# lower = conditioning. Change these when switching to another robot or naming scheme.
UPPER_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Lower joints fed to the predictor (history / conditioning). List every joint you want here
# (e.g. add ankles by including those names in order).
LOWER_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    # "left_ankle_pitch_joint",
    # "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    # "right_ankle_pitch_joint",
    # "right_ankle_roll_joint",
]

# Loss weighting configuration (optional)
LOSS_WEIGHTS = {
    "*_foot_pitch": 1.0,
    "*_foot_rel_yaw": 1.0,
    "step_length": 1.0,
    "step_width": 1.0,
    "double_support_factor": 1.0,
    "cadence_hz": 1.0,
    "root_height": 1.0,
    "pelvis_roll": 1.0,
    "pelvis_pitch": 1.0,
    "torso_roll": 1.0,
    "torso_pitch": 1.0,
}
USE_WEIGHTED_LOSS = False
LOG_PER_FEATURE_LOSS_INTERVAL = 0  # 0 to disable

# Symmetry regularization loss: mirror(model(x)) ~= model(mirror(x))
ENABLE_SYMMETRY_LOSS = True
# Slightly stronger than before: foot/std symmetry + clean-input sym loss justify it.
SYMMETRY_LOSS_WEIGHT = 0.1
# Linearly ramp symmetry weight from 0 to SYMMETRY_LOSS_WEIGHT over this many epochs.
# 0 disables ramping and applies full weight from epoch 0.
SYMMETRY_LOSS_RAMP_EPOCHS = 2500

# Feature regularization: penalize selected predicted outputs towards 0
ENABLE_FEATURE_REGULARIZATION = True
FEATURE_REGULARIZATION_CONFIG = {
    # Format: {"pattern": weight}
    # Patterns match feature/joint names (supports wildcards)
    # Higher weight = stronger penalty for non-zero values
    "waist_yaw_*": 0.1,
    "waist_roll_*": 0.1,
    "waist_pitch_*": 0.15,
    "*_foot_pitch": 0.1,        # Penalize foot pitch deviations from zero
    "*_foot_rel_yaw": 0.05,      # Penalize foot relative yaw deviations from zero
    # New features regularization (small weights)
    "future_wz_local": 0.1,    # Penalize angular velocity (encourage straight walking)
    # Note: future_vx_local is NOT regularized (walking speed should not be penalized)
    "cadence_hz": 0.005,
    "double_support_factor": 0.01,
    "root_height": 0.0,
    # Weaker than before so symmetry + reconstruction are not fought as hard.
    "pelvis_pitch": 0.06,
    "pelvis_roll": 0.02,
    "torso_roll": 0.02,
    "torso_pitch": 0.06,
}
FEATURE_REGULARIZATION_WEIGHT = 1.0

# Optimizer/stability knobs for small-data generalization.
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP_NORM = 1.0
