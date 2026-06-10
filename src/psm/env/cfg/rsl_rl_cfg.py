from __future__ import annotations

"""RL configuration for the G1 PSM task.

Symmetry (Isaac Lab idea, adapted to G1)
----------------------------------------
Isaac Lab ANYmal uses ``symmetry_cfg`` with ``use_data_augmentation=True`` and a
``compute_symmetric_states`` that **stacks** original + transformed samples so
PPO’s policy and value losses see more diverse data per mini-batch (rsl_rl
repeats advantages / returns by ``num_aug``).

We use the same **data-augmentation** switch. Our transform is **left–right
bilateral** only (**2×** batch), not ANYmal’s **4×** (LR + front–back +
diagonal): a biped has no front/hind leg swap; FB/diagonal are not the same
task symmetries as on a quadruped.

**Should ``use_data_augmentation`` be True?**
  **Yes**, if you want the Isaac-style recipe: train on both original and
  mirrored transitions each update (often paired with ``use_mirror_loss=False``,
  like ANYmal’s symmetry configs that only enable augmentation).

  Use ``use_data_augmentation=False`` and ``use_mirror_loss=True`` if you prefer
  rsl_rl’s **mirror MSE** on the Gaussian mean only (no extra batch size in the
  surrogate/value path).

  You can enable both; rsl_rl supports it, but the symmetry signal is partially
  redundant—consider lowering ``mirror_loss_coeff``.
"""

from dataclasses import dataclass, field
from typing import Callable

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from psm.env.utils.symmetry import compute_symmetric_states


@dataclass
class SymmetryCfg:
  """Passed through to rsl_rl (``algorithm.symmetry_cfg`` in the resolved runner dict)."""

  use_data_augmentation: bool = True
  """Disabled by default to match Unitree velocity baseline behaviour."""

  use_mirror_loss: bool = True
  """If True, adds symmetry MSE on mean actions vs mirrored actions (rsl_rl mirror path)."""

  data_augmentation_func: Callable = compute_symmetric_states
  """Must return augmented obs/actions; batch size sets ``num_aug`` for rsl_rl."""

  mirror_loss_coeff: float = 1.0
  """Weight on the symmetry loss term when ``use_mirror_loss`` is True."""


@dataclass
class G1PsmPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  symmetry_cfg: SymmetryCfg = field(default_factory=SymmetryCfg)


def g1_psm_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """PPO runner config for the G1 PSM task.

  Symmetry is disabled by default for parity with Unitree velocity training.
  You can re-enable later by toggling ``algorithm.symmetry_cfg``.

  ``obs_groups`` maps rsl_rl actor/critic networks to TensorDict keys (Isaac Lab
  often names the policy group ``policy``; mjlab uses ``actor`` / ``critic``).
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=G1PsmPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    experiment_name="g1_psm",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=6_000,
  )

