"""G1 PSM env: ``make_psm_env_cfg`` (MDP terms) and ``psm_env_cfg`` (training weights + play)."""

from __future__ import annotations

import math

from psm.assets.unitree_g1.g1_constants import G1_ACTION_SCALE, get_g1_robot_cfg
from psm.predictor.bundle import default_predictor_path
import psm.env.mdp as mdp
from psm.env.mdp.commands import PsmVelocityCommandCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import curriculums as mdp_curriculums
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

_G1_FOOT_GEOM_NAMES = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)
# Match Servant ``SERVANT_BODY_NAME`` / ``SERVANT_STAND_STILL_BODY_NAMES`` roles.
_G1_TORSO_BODY = ("torso_link",)
_G1_STAND_STILL_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
_G1_FLAT_CONTACT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
_G1_FOOT_SITES = ("left_foot", "right_foot")

TRAIN_NUM_ENVS = 4096
PLAY_NUM_ENVS = 1


def make_psm_env_cfg() -> ManagerBasedRlEnvCfg:
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.6, "command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=_G1_FOOT_SITES)},
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "pred_upper": ObservationTermCfg(func=mdp.predictor_upper_targets),
    "pred_body_features": ObservationTermCfg(func=mdp.predictor_body_features),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=G1_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "twist": PsmVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.2,
      # rel_heading_envs=0.25,
      lin_vel_deadband=0.1,
      ang_vel_deadband=0.1,
      # heading_command=True,
      # heading_control_stiffness=0.5,
      debug_vis=True,
      viz=PsmVelocityCommandCfg.VizCfg(z_offset=1.15),
      ranges=PsmVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 2.0),  
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-1.0, 1.0),
        # heading=(-math.pi, math.pi),
      ),
      predictor_path=default_predictor_path(),
      command_name="twist",
      feet_contact_sensor_name="feet_ground_contact",
      warmup_history_steps=None,
      ghost_upper_prediction=True,
      ghost_alpha=0.4,
    ),
  }

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0), "yaw": (-3.14, 3.14)},
        "velocity_range": {},
      },
    ),
    # "randomize_torso_mass": EventTermCfg(
    #   mode="startup",
    #   func=dr.body_mass,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    #     "ranges": (0.0, 1.0),
    #     "operation": "add",
    #   },
    # ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.075, 0.075),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "pull_robot": EventTermCfg(
      func=mdp.apply_external_force_torque,
      mode="interval",
      interval_range_s=(1.0, 4.0),
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "force_range": (-6.0, 6.0),
        "torque_range": (-1.5, 1.5),
      },
    ),
    # "randomize_actuator_gains": EventTermCfg(
    #   mode="startup",
    #   func=dr.pd_gains,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    #     "kp_range": (0.9, 1.1),
    #     "kd_range": (0.8, 1.2),
    #     "operation": "scale",
    #   },
    # ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=_G1_FOOT_GEOM_NAMES),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)},
      },
    ),
  }

  # Rewards aligned with ``make_servant_velocity_loco_env_cfg``; upper body via predictor.
  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=3.0,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      # func=mdp.track_angular_velocity,
      func=mdp.track_angular_velocity,
      weight=4.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "upright": RewardTermCfg(
      func=mdp.flat_orientation_exp,
      weight=2.5,
      params={
        "std": 0.25,
        "asset_cfg": SceneEntityCfg("robot", body_names=_G1_TORSO_BODY),
      },
    ),
    "root_height": RewardTermCfg(
      func=mdp.root_height_exp,
      weight=2.0,
      params={
        "target_height": 0.79,
        "std": 0.1,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.5,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=_G1_TORSO_BODY)},

    ),
    "lin_vel_z_l2": RewardTermCfg(
      func=mdp.lin_vel_z_l2,
      weight=0.0,
    ),
    "dof_acc_l2": RewardTermCfg(
      func=envs_mdp.joint_acc_l2,
      weight=0.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",))},
    ),
    "mechanical_power": RewardTermCfg(
      func=envs_mdp.electrical_power_cost,
      weight=0.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",))},
    ),
    "dof_pos_limits": RewardTermCfg(
      func=vel_mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",))},
    ),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l1, weight=-0.1),
    "air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.8,
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=0.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=_G1_FOOT_SITES),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
      },
    ),
    # "step_width": RewardTermCfg(
    #   func=mdp.adaptive_leg_width,
    #   weight=3.0,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", site_names=_G1_FOOT_SITES),
    #     "max_target_width": 0.24,
    #     "min_target_width": 0.2,
    #     "vel_transition": 0.4,
    #     "vel_max": 1.0,
    #     "std": 0.25,
    #     "command_name": "twist",
    #   },
    # ),
    "step_width": RewardTermCfg(
      func=mdp.step_width_matching,
      weight=0.0,
      params={
        "command_name": "twist",
        # "vel_threshold": 0.1,
        # "max_target_width": 0.24,
      },
    ),
    "feet_yaw": RewardTermCfg(
      func=mdp.body_yaw_alignment,
      weight=1.5,
      params={
        "command_name": "twist",
        "asset_cfg": SceneEntityCfg("robot", body_names=_G1_STAND_STILL_BODIES),
        "std": 0.25,
        "offsets": [-0.05, 0.05],
      },
    ),
    # "feet_yaw": RewardTermCfg(
    #   func=mdp.feet_yaw_matching,
    #   weight=1.5,
    #   params={
    #     "std": 0.25,
    #     # "command_name": "twist",
    #     "ang_vel_threshold": 0.1,
    #   },
    # ),
    "stand_still": RewardTermCfg(
      func=mdp.stand_still,
      weight=4.0,
      params={
        "target_height": 0.034,
        "std": 0.25,
        "command_name": "twist",
        "asset_cfg": SceneEntityCfg("robot", body_names=_G1_STAND_STILL_BODIES),
        "vel_threshold": 0.1,
      },
    ),
    "feet_contact": RewardTermCfg(
      func=mdp.contact,
      weight=1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "contact_threshold": 40.0,
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "no_jump": RewardTermCfg(
      func=mdp.no_jump,
      weight=5.0,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "duty_cycle": RewardTermCfg(
      func=mdp.coordinated_duty_cycle,
      weight=0.0,
      params={
        "command_name": "twist",
        "sensor_name": "feet_ground_contact",
        "slow_stance_ratio": 0.75,
        "fast_stance_ratio": 0.5,
        "vel_transition": 0.4,
        "vel_max": 0.6,
        "std": 0.25,
        "coordination_weight": 0.5,
        "vel_threshold": 0.1,
      },
    ),
    "feet_pitch": RewardTermCfg(
      func=mdp.feet_pitch_matching,
      weight=0.0,
      params={
        "std": 0.2,
        "sensor_name": "feet_ground_contact",
        "contact_force_threshold": 25.0,
      },
    ),

    "feet_roll": RewardTermCfg(
      func=mdp.body_orientation_exp,
      weight=0.0,
      params={
        "std": 0.25,
        "asset_cfg": SceneEntityCfg("robot", body_names=_G1_FLAT_CONTACT_BODIES),
        "mask": [0.0, 1.0],
      },
    ),
    # "step_length": RewardTermCfg(
    #   func=mdp.adaptive_step_length,
    #   weight=0.0,
    #   params={
    #     "command_name": "twist",
    #     "vel_threshold": 0.1,
    #     "min_target_length": 0.35,
    #     "max_target_length": 0.65,
    #     "vel_transition": 0.15,
    #     "vel_max": 1.0,
    #     "std": 0.25,
    #     "asset_cfg": SceneEntityCfg("robot", site_names=_G1_FOOT_SITES),
    #     "sensor_name": "feet_ground_contact",
    #   },
    # ),
    "step_length": RewardTermCfg(
      func=mdp.step_length_matching,
      weight=0.0,
      params={
        "command_name": "twist",
        "vel_threshold": 0.1,
        "min_target_length": 0.35,
        "max_target_length": 0.65,
        # "vel_transition": 0.15,
        # "vel_max": 1.0,
        "std": 0.25,
        "asset_cfg": SceneEntityCfg("robot", site_names=_G1_FOOT_SITES),
        "sensor_name": "feet_ground_contact",
      },
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
    ),
    "flat_contact": RewardTermCfg(
      func=mdp.body_orientation_exp,
      weight=0.3,
      params={
        "std": 0.25,
        "asset_cfg": SceneEntityCfg("robot", body_names=_G1_FLAT_CONTACT_BODIES),
        "mask": [0.0, 1.0],
      },
    ),
    "upper_joints": RewardTermCfg(
      func=mdp.predictor_upper_joint_tracking,
      weight=2.0,
      params={
        "std": 0.2,
        "kernel": "exp",
        "scales": {
          r".*waist_yaw.*": 0.5,
          r".*waist_roll.*": 0.5,
          r".*waist_pitch.*": 0.3,
        },
      },
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}),
  }

  curriculum = {
    "command_vel": CurriculumTermCfg(
      func=mdp_curriculums.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-0.4, 0.8), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.5, 1.5)},
          {"step": 2000 * 24, "lin_vel_x": (-0.6, 1.5), "lin_vel_y": (-0.8, 0.8), "ang_vel_z": (-2.0, 2.0)},
          {"step": 4000 * 24, "lin_vel_x": (-0.8, 2.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-2.5, 2.5)},
        ],
      },
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_g1_robot_cfg()},
      sensors=(feet_ground_cfg, self_collision_cfg),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=None,
      njmax=300,
      contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=50,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )


def psm_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Training/play config with ``servant_loco_env_cfg``-style weight ramps (flat terrain)."""
  cfg = make_psm_env_cfg()
  cfg.scene.num_envs = PLAY_NUM_ENVS if play else TRAIN_NUM_ENVS

  cfg.rewards["track_linear_velocity"].weight = 4.0
  cfg.rewards["track_angular_velocity"].weight = 5.0
  cfg.rewards["root_height"].weight = 2.0
  cfg.rewards["body_ang_vel"].weight = -1.0
  cfg.rewards["lin_vel_z_l2"].weight = -0.5
  cfg.rewards["dof_pos_limits"].weight = -3.0
  cfg.rewards["action_rate"].weight = -0.1
  cfg.rewards["soft_landing"].weight = -1e-5
  cfg.rewards["dof_acc_l2"].weight = -2e-6
  cfg.rewards["stand_still"].weight = 4.0
  cfg.rewards["feet_contact"].weight = 1.0
  cfg.rewards["upright"].weight = 2.5

  cfg.rewards["foot_slip"].weight = -0.2
  cfg.rewards["duty_cycle"].weight = 4.0
  cfg.rewards["mechanical_power"].weight = -0.0002
  cfg.rewards["upper_joints"].weight =3.
  cfg.rewards["step_width"].weight = 1.0
  cfg.rewards["step_length"].weight = 0.0
  cfg.rewards["flat_contact"].weight = 0.3

  cfg.curriculum["step_width"] = CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={
      "reward_name": "step_width",
      "stages": [{"step": 0*1500 * 24, "weight": 2.0}, 
                        {"step": 2000 * 24, "weight": 3.0}],
    },
  )
  cfg.curriculum["step_length"] = CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={
      "reward_name": "step_length",
      "stages": [{"step": 0*1000 * 24, "weight": 1.0},
                        {"step": 2500 * 24, "weight": 2.5}],
    },
  )
  cfg.curriculum["feet_yaw"] = CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={
      "reward_name": "feet_yaw",
      "stages": [{"step": 0*1500 * 24, "weight": 3.0}],
    },
  )
  cfg.curriculum["dof_acc_l2_ramp"] = CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={
      "reward_name": "dof_acc_l2",
      "stages": [
        {"step": 0*1000 * 24, "weight": -4e-6},
        {"step": 3000 * 24, "weight": -6e-6},
      ],
    },
  )
  
  cfg.curriculum["upper_joints"] = CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={
      "reward_name": "upper_joints",
      "stages": [{"step": 0*1500 * 24, "weight": 3.0},
                        {"step": 3000 * 24, "weight": 6.0},
                        {"step": 4000 * 24, "weight": 9.0}],
    },
  )
  
  cfg.curriculum["foot_slip"] = CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={
      "reward_name": "foot_slip",
      "stages": [{"step": 0*1000 * 24, "weight": -0.1}],
    },
  )
  cfg.curriculum["mechanical_power"] = CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={
      "reward_name": "mechanical_power",
      "stages": [{"step": 2000 * 24, "weight": -0.001},
                        {"step": 4000 * 24, "weight": -0.0014}],
    },
  )
  cfg.curriculum["action_rate"] = CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={
      "reward_name": "action_rate",
      "stages": [{"step": 0*2000 * 24, "weight": -0.15}],
    },
  )
  cfg.curriculum["feet_pitch"] = CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={
      "reward_name": "feet_pitch",
      "stages": [{"step": 0*1500 * 24, "weight": 2.0}],
    },
  )
  cfg.curriculum["feet_roll"] = CurriculumTermCfg(
    func=envs_mdp.reward_curriculum,
    params={
      "reward_name": "feet_roll",
      "stages": [{"step": 0*1500 * 24, "weight": 1.5}],
    },
  )

  cfg.observations["actor"].history_length = 1

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("pull_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, PsmVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


__all__ = [
  "make_psm_env_cfg",
  "psm_env_cfg",
]
