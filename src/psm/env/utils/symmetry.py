from __future__ import annotations

"""Left-right symmetry for the Unitree G1 robot (PSM task).

Mirrors both ``actor`` and ``critic`` observation groups so rsl_rl mirror loss /
data augmentation (which passes the full ``TensorDict`` of observations) stays
consistent with asymmetric critic terms (feet, predictor targets).

Comparison to Isaac Lab (e.g. ANYmal ``velocity.mdp.symmetry``):
  That stack uses ``use_data_augmentation=True`` and stacks **four** views per
  sample (identity + LR + front-back + diagonal) via ``obs.repeat(4)``. We apply
  the **same training idea**—extra symmetric samples in each PPO mini-batch—
  with ``repeat(SYMMETRY_AUGMENT_FACTOR)`` where ``SYMMETRY_AUGMENT_FACTOR=2``
  (identity + LR). Front-back / diagonal are **not** implemented for G1: a biped
  has no front/hind leg decomposition like a quadruped, and forward/backward
  task semantics differ from symmetric ANYmal locomotion.

Optional kwargs (e.g. ``obs_type``) match Isaac Lab / isaaclab_rl style: ``policy``
or ``actor`` mirrors only the actor group; ``critic`` only the critic; omitted
mirrors every present group (standard leggedrobotics/rsl_rl behaviour).

rsl_rl PPO: ``use_data_augmentation`` calls this function with batched
``observations`` / ``actions`` and infers ``num_aug`` from the returned batch
size. ``use_mirror_loss`` uses the same function to build mirrored means; when
``use_data_augmentation`` is True, the mini-batch is augmented before that step
(see leggedrobotics/rsl_rl PPO implementation).
"""

# One mirrored copy + original → factor 2 (Isaac ANYmal-style 4-fold would use 4).
SYMMETRY_AUGMENT_FACTOR: int = 2

import numpy as np
import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv


# Must match `SYMMETRY_SPEC` body rows in `psm.predictor.config` for predictor body targets.
_BODY_FEATURE_SYMMETRY_SPEC = [
  ("left_foot_pitch", "right_foot_pitch", 1),
  ("left_foot_rel_yaw", "right_foot_rel_yaw", -1),
  ("step_length", "step_length", 1),
  ("step_width", "step_width", 1),
  ("cadence_hz", "cadence_hz", 1),
  ("double_support_factor", "double_support_factor", 1),
  ("swing_peak_height", "swing_peak_height", 1),
  ("pelvis_roll", "pelvis_roll", -1),
  ("pelvis_pitch", "pelvis_pitch", 1),
  ("torso_roll", "torso_roll", -1),
  ("torso_pitch", "torso_pitch", 1),
]

_SYMMETRY_SPEC = [
  # (left_name, right_name, scale)   scale=-1 for roll/yaw axes, +1 for pitch
  # Legs
  ("left_hip_pitch_joint", "right_hip_pitch_joint", 1),
  ("left_hip_roll_joint", "right_hip_roll_joint", -1),
  ("left_hip_yaw_joint", "right_hip_yaw_joint", -1),
  ("left_knee_joint", "right_knee_joint", 1),
  ("left_ankle_pitch_joint", "right_ankle_pitch_joint", 1),
  ("left_ankle_roll_joint", "right_ankle_roll_joint", -1),
  # Arms
  ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint", 1),
  ("left_shoulder_roll_joint", "right_shoulder_roll_joint", -1),
  ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint", -1),
  ("left_elbow_joint", "right_elbow_joint", 1),
  ("left_wrist_roll_joint", "right_wrist_roll_joint", -1),
  ("left_wrist_pitch_joint", "right_wrist_pitch_joint", 1),
  ("left_wrist_yaw_joint", "right_wrist_yaw_joint", -1),
  ("waist_yaw_joint", "waist_yaw_joint", -1),
  ("waist_roll_joint", "waist_roll_joint", -1),
  ("waist_pitch_joint", "waist_pitch_joint", 1),
]


def _policy_joint_names(unwrapped_env: ManagerBasedRlEnv) -> list[str]:
  """Joint names in policy action / joint_pos / joint_vel tensor order."""
  return list(unwrapped_env.action_manager.get_term("joint_pos").target_names)


def _build_symmetry_map(
  ordered_names: list[str],
  device: torch.device,
  symmetry_spec: list[tuple[str, str, int]] = _SYMMETRY_SPEC,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Build an index/scale mirror map for a specific ordered name list."""
  n = len(ordered_names)
  indices = torch.arange(n, device=device)
  scales = torch.ones(n, device=device)
  name_to_idx = {name: idx for idx, name in enumerate(ordered_names)}

  for name_a, name_b, scale in symmetry_spec:
    ia = name_to_idx.get(name_a)
    ib = name_to_idx.get(name_b)
    if ia is None or ib is None:
      continue
    indices[[ia, ib]] = indices[[ib, ia]]
    scales[ia] = scale
    scales[ib] = scale

  if not torch.equal(indices[indices], torch.arange(n, device=device)):
    raise RuntimeError(
      "Symmetry index map is not an involution for names: "
      f"{ordered_names}"
    )
  return indices, scales


def _apply_symmetry(
  names: list[str],
  n: int,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return (indices, scales) that implement the left-right reflection."""
  return _build_symmetry_map(names[:n], device)


def _apply_body_feature_symmetry(
  names: list[str],
  n: int,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Left-right reflection for `body_feature_names`-ordered predictor outputs."""
  indices = torch.arange(n, device=device)
  scales = torch.ones(n, device=device)
  for name_a, name_b, scale in _BODY_FEATURE_SYMMETRY_SPEC:
    if name_a in names and name_b in names:
      ia, ib = names.index(name_a), names.index(name_b)
      indices[[ia, ib]] = indices[[ib, ia]]
      scales[ia] = scale
      scales[ib] = scale
  return indices, scales


def _pairwise_swap_indices(flat_dim: int, pair_size: int, device: torch.device) -> torch.Tensor:
  """Swap consecutive blocks of ``pair_size`` (e.g. L/R feet per timestep)."""
  assert flat_dim % (2 * pair_size) == 0, (
    f"Expected flat_dim divisible by {2 * pair_size}, got {flat_dim}"
  )
  indices = torch.arange(flat_dim, device=device)
  out = indices.clone()
  n_pairs = flat_dim // (2 * pair_size)
  for p in range(n_pairs):
    a = p * 2 * pair_size
    out[a : a + pair_size] = indices[a + pair_size : a + 2 * pair_size]
    out[a + pair_size : a + 2 * pair_size] = indices[a : a + pair_size]
  return out


def _foot_contact_force_indices(flat_dim: int, device: torch.device) -> torch.Tensor:
  """Swap left/right foot force blocks (3D per foot)."""
  assert flat_dim % 6 == 0, f"foot_contact_forces: expected multiple of 6, got {flat_dim}"
  return _pairwise_swap_indices(flat_dim, 3, device)


def _identity_indices(flat_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
  idx = torch.arange(flat_dim, device=device)
  return idx, torch.ones(flat_dim, device=device)


def _append_term_maps(
  term_name: str,
  offset_idx: int,
  unwrapped_env: ManagerBasedRlEnv,
  device: torch.device,
  term_flat_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
  """Return concatenated (indices+offset, scales) for one observation term."""
  if term_name in ("base_lin_vel", "projected_gravity", "torso_gravity"):
    assert term_flat_dim == 3, f"{term_name}: expected flat dim 3, got {term_flat_dim}"
    idx = torch.arange(offset_idx, offset_idx + 3, device=device)
    scale = torch.tensor([1.0, -1.0, 1.0], device=device)
    return idx, scale, offset_idx + 3

  if term_name in ("base_ang_vel", "torso_ang_vel"):
    assert term_flat_dim == 3, f"{term_name}: expected flat dim 3, got {term_flat_dim}"
    idx = torch.arange(offset_idx, offset_idx + 3, device=device)
    scale = torch.tensor([-1.0, 1.0, -1.0], device=device)
    return idx, scale, offset_idx + 3

  if term_name == "command":
    assert term_flat_dim == 3, f"{term_name}: expected flat dim 3, got {term_flat_dim}"
    idx = torch.arange(offset_idx, offset_idx + 3, device=device)
    scale = torch.tensor([1.0, -1.0, -1.0], device=device)
    return idx, scale, offset_idx + 3

  if term_name == "phase":
    # phase=[sin(phi), cos(phi)] from a global scalar phase.
    # Left-right mirror does not change global time phase in this setup.
    assert term_flat_dim == 2, f"{term_name}: expected flat dim 2, got {term_flat_dim}"
    idx = torch.arange(offset_idx, offset_idx + 2, device=device)
    scale = torch.ones(2, device=device)
    return idx, scale, offset_idx + 2

  if term_name == "actions":
    act_indices, act_scales = _action_reflection(unwrapped_env, device)
    assert act_indices.shape[0] == term_flat_dim, (
      f"actions: dim mismatch {act_indices.shape[0]} vs {term_flat_dim}"
    )
    return act_indices + offset_idx, act_scales, offset_idx + term_flat_dim

  if term_name in ("joint_pos", "joint_vel"):
    joint_names = _policy_joint_names(unwrapped_env)
    assert len(joint_names) == term_flat_dim, (
      f"{term_name}: joint count {len(joint_names)} vs flat dim {term_flat_dim}"
    )
    j_idx, j_scale = _build_symmetry_map(joint_names, device)
    return j_idx + offset_idx, j_scale, offset_idx + term_flat_dim

  if term_name in ("foot_height", "foot_air_time", "foot_contact"):
    # Two feet per step in site order (left_foot, right_foot); flatten is pairs.
    assert term_flat_dim % 2 == 0, f"{term_name}: expected even flat dim, got {term_flat_dim}"
    local = _pairwise_swap_indices(term_flat_dim, 1, device)
    return local + offset_idx, torch.ones(term_flat_dim, device=device), offset_idx + term_flat_dim

  if term_name == "foot_contact_forces":
    local_idx = _foot_contact_force_indices(term_flat_dim, device)
    return local_idx + offset_idx, torch.ones(term_flat_dim, device=device), offset_idx + term_flat_dim

  if term_name == "pred_upper":
    cmd = unwrapped_env.command_manager.get_term("twist")
    upper_order: list[str] = list(getattr(cmd, "upper_order", ()))
    if len(upper_order) != term_flat_dim:
      raise ValueError(
        f"pred_upper: upper_order length {len(upper_order)} != obs dim {term_flat_dim}"
      )
    pu_idx, pu_scale = _apply_symmetry(upper_order, len(upper_order), device)
    return pu_idx + offset_idx, pu_scale, offset_idx + term_flat_dim

  if term_name == "pred_body_features":
    cmd = unwrapped_env.command_manager.get_term("twist")
    body_names = list(getattr(cmd, "body_feature_names", ()))
    if len(body_names) != term_flat_dim:
      raise ValueError(
        f"pred_body_features: body_feature_names length {len(body_names)} != obs dim {term_flat_dim}"
      )
    bf_idx, bf_scale = _apply_body_feature_symmetry(body_names, len(body_names), device)
    return bf_idx + offset_idx, bf_scale, offset_idx + term_flat_dim

  raise NotImplementedError(
    f"symmetry_cfg: add handling for observation term {term_name!r} "
    f"(flat_dim={term_flat_dim})"
  )


def _action_reflection(
  unwrapped_env: ManagerBasedRlEnv,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return (action_indices, action_scales) in policy joint order."""
  return _build_symmetry_map(_policy_joint_names(unwrapped_env), device)


def _build_indices_and_scales_for_group(
  unwrapped_env: ManagerBasedRlEnv,
  group_name: str,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
  term_names = unwrapped_env.observation_manager.active_terms[group_name]
  term_dims = unwrapped_env.observation_manager.group_obs_term_dim[group_name]
  obs_indices: list[torch.Tensor] = []
  obs_scales: list[torch.Tensor] = []
  offset_idx = 0

  for term_name, dim_tuple in zip(term_names, term_dims, strict=True):
    term_flat_dim = int(np.prod(dim_tuple))
    idx, scale, offset_idx = _append_term_maps(
      term_name, offset_idx, unwrapped_env, device, term_flat_dim
    )
    obs_indices.append(idx)
    obs_scales.append(scale)

  return torch.cat(obs_indices), torch.cat(obs_scales)


def _mirror_group_tensor(
  flat: torch.Tensor,
  obs_indices_t: torch.Tensor,
  obs_scales_t: torch.Tensor,
) -> torch.Tensor:
  """Apply left-right map to a [B, D] observation slice."""
  if flat.shape[1] > obs_scales_t.shape[0]:
    aug = flat.detach().clone().reshape(flat.shape[0], -1, obs_scales_t.shape[0])
    aug = (aug * obs_scales_t[None, None])[:, :, obs_indices_t].reshape(flat.shape[0], -1)
  else:
    aug = (flat.detach().clone() * obs_scales_t[None])[:, obs_indices_t]
  return aug


def _obs_groups_for_kwargs(kwargs: dict) -> list[str] | None:
  """Return which TensorDict keys to mirror; None means all of actor/critic present."""
  ot = kwargs.get("obs_type")
  if ot is None:
    return None
  if ot in ("policy", "actor"):
    return ["actor"]
  if ot == "critic":
    return ["critic"]
  raise ValueError(
    f"symmetry_cfg: obs_type must be 'policy', 'actor', 'critic', or omitted; got {ot!r}"
  )


@torch.no_grad()
def compute_symmetric_states(
  env: ManagerBasedRlEnv,
  obs: TensorDict | None = None,
  actions: torch.Tensor | None = None,
  **kwargs,
):
  """Mirror observations and actions for left-right symmetry.

  Augments every observation group listed in ``active_terms`` (typically
  ``actor`` and ``critic``) so mirror loss sees physically consistent mirrored
  states for both policy and value networks.
  """
  unwrapped_env = env.unwrapped
  groups_filter = _obs_groups_for_kwargs(kwargs)

  if obs is not None:
    obs_aug = obs.repeat(SYMMETRY_AUGMENT_FACTOR)
    half = obs.batch_size[0]

    for group_name in unwrapped_env.observation_manager.active_terms:
      if groups_filter is not None and group_name not in groups_filter:
        continue
      if group_name not in obs.keys():
        continue
      group_flat = obs[group_name]
      if not isinstance(group_flat, torch.Tensor):
        continue

      obs_indices_t, obs_scales_t = _build_indices_and_scales_for_group(
        unwrapped_env, group_name, group_flat.device
      )
      if group_flat.shape[1] != obs_scales_t.shape[0]:
        raise ValueError(
          f"symmetry_cfg: {group_name} dim {group_flat.shape[1]} does not match "
          f"expected {obs_scales_t.shape[0]} from observation_manager term layout"
        )
      aug = _mirror_group_tensor(group_flat, obs_indices_t, obs_scales_t)
      obs_aug[group_name][half:] = aug
  else:
    obs_aug = None

  if actions is not None:
    act_indices, act_scales = _action_reflection(unwrapped_env, actions.device)
    aug_act = (actions.detach().clone() * act_scales[None])[:, act_indices]
    actions_aug = actions.repeat(SYMMETRY_AUGMENT_FACTOR, 1)
    actions_aug[actions.shape[0] :] = aug_act
  else:
    actions_aug = None

  return obs_aug, actions_aug
