from __future__ import annotations

import torch
from torch import nn


def _activation_class(name: str) -> type[nn.Module]:
  n = str(name).lower()
  if n == "relu":
    return nn.ReLU
  if n == "gelu":
    return nn.GELU
  if n in ("silu", "swish"):
    return nn.SiLU
  raise ValueError(f"Unknown ACTIVATION {name!r}; use relu|gelu|silu")


class PsmPredictor(nn.Module):
  """Upper-body + body-feature predictor from lower-body history (weighted) and root velocities.

  Encodes a recency-weighted sequence of lower joint positions [+ velocities] with an MLP, GRU,
  or 1D conv stack, then fuses with body-frame root-velocity history and command features.
  """

  def __init__(
    self,
    output_size: int,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    leg_pos_mean: torch.Tensor,
    leg_pos_std: torch.Tensor,
    body_vel_mean: torch.Tensor,
    body_vel_std: torch.Tensor,
    foot_pos_mean: torch.Tensor | None = None,
    foot_pos_std: torch.Tensor | None = None,
    cmd_mean: torch.Tensor | None = None,
    cmd_std: torch.Tensor | None = None,
    *,
    num_lower: int,
    history_horizon: int,
    prediction_horizon: int,
    cmd_feature_dim: int | None = None,
    history_input_mode: str = "joints",
    joints_history_weight: float = 1.0,
    feet_history_weight: float = 1.0,
    use_lower_joint_velocity: bool = False,
    use_foot_velocity: bool = False,
    leg_vel_mean: torch.Tensor | None = None,
    leg_vel_std: torch.Tensor | None = None,
    foot_vel_mean: torch.Tensor | None = None,
    foot_vel_std: torch.Tensor | None = None,
    history_recency_decay: float = 2.5,
    encoder_type: str = "gru",
    encoder_hidden_size: int = 256,
    conv1d_channels: int = 96,
    hidden_size: int = 256,
    gru_num_layers: int = 1,
    conv1d_kernel_size: int = 5,
    head_hidden_depth: int = 3,
    head_dropout: float = 0.0,
    activation: str = "relu",
  ):
    super().__init__()
    self.output_size = int(output_size)
    self.num_lower = int(num_lower)
    self.history_horizon = int(history_horizon)
    self.prediction_horizon = int(prediction_horizon)
    self.use_lower_joint_velocity = bool(use_lower_joint_velocity)
    self.use_foot_velocity = bool(use_foot_velocity)
    self.history_recency_decay = float(history_recency_decay)
    self.history_input_mode = str(history_input_mode).lower()
    if self.history_input_mode not in ("joints", "feet", "both"):
      raise ValueError("history_input_mode must be one of: joints|feet|both")
    self.joints_history_weight = float(joints_history_weight)
    self.feet_history_weight = float(feet_history_weight)
    self.encoder_type = "gru"
    self.encoder_hidden_size = int(encoder_hidden_size)
    self.gru_num_layers = int(gru_num_layers)
    self.conv1d_kernel_size = int(conv1d_kernel_size)
    self.head_hidden_depth = int(head_hidden_depth)
    self.head_dropout = float(head_dropout)
    self.activation_name = str(activation)
    self._act = _activation_class(activation)

    self.register_buffer("y_mean", y_mean.clone())
    self.register_buffer("y_std", y_std.clone())
    self.register_buffer("leg_pos_mean", leg_pos_mean.clone())
    self.register_buffer("leg_pos_std", leg_pos_std.clone())
    if foot_pos_mean is None:
      foot_pos_mean = torch.zeros(6, dtype=leg_pos_mean.dtype, device=leg_pos_mean.device)
    if foot_pos_std is None:
      foot_pos_std = torch.ones(6, dtype=leg_pos_std.dtype, device=leg_pos_std.device)
    self.register_buffer("foot_pos_mean", foot_pos_mean.clone())
    self.register_buffer("foot_pos_std", foot_pos_std.clone())
    self.register_buffer("body_vel_mean", body_vel_mean.clone())
    self.register_buffer("body_vel_std", body_vel_std.clone())
    cmd_d = int(cmd_feature_dim) if cmd_feature_dim is not None else int(2 * prediction_horizon)
    if cmd_mean is None:
      cmd_mean = torch.zeros(cmd_d, dtype=body_vel_mean.dtype, device=body_vel_mean.device)
    if cmd_std is None:
      cmd_std = torch.ones(cmd_d, dtype=body_vel_std.dtype, device=body_vel_std.device)
    self.register_buffer("cmd_mean", cmd_mean.clone())
    self.register_buffer("cmd_std", cmd_std.clone())

    if self.use_lower_joint_velocity:
      if leg_vel_mean is None or leg_vel_std is None:
        raise ValueError("leg_vel_mean/std required when use_lower_joint_velocity is True")
      self.register_buffer("leg_vel_mean", leg_vel_mean.clone())
      self.register_buffer("leg_vel_std", leg_vel_std.clone())
    else:
      self.register_buffer("leg_vel_mean", torch.zeros(num_lower))
      self.register_buffer("leg_vel_std", torch.ones(num_lower))

    n_foot = int(self.foot_pos_mean.numel())
    if self.use_foot_velocity:
      if foot_vel_mean is None or foot_vel_std is None:
        raise ValueError("foot_vel_mean/std required when use_foot_velocity is True")
      self.register_buffer("foot_vel_mean", foot_vel_mean.clone())
      self.register_buffer("foot_vel_std", foot_vel_std.clone())
    else:
      self.register_buffer("foot_vel_mean", torch.zeros(n_foot))
      self.register_buffer("foot_vel_std", torch.ones(n_foot))

    leg_in = 0
    if self.history_input_mode in ("joints", "both"):
      leg_in += num_lower
      if self.use_lower_joint_velocity:
        leg_in += num_lower
    if self.history_input_mode in ("feet", "both"):
      leg_in += n_foot
      if self.use_foot_velocity:
        leg_in += n_foot
    nl = max(1, self.gru_num_layers)
    self._gru = nn.GRU(
      leg_in,
      encoder_hidden_size,
      num_layers=nl,
      batch_first=True,
    )
    enc_dim = encoder_hidden_size

    self.cmd_feature_dim = int(self.cmd_mean.numel())
    self.body_vel_dim = int(self.body_vel_mean.numel())
    # Lightweight summary of velocity history: mean + latest, instead of flatten(H * D).
    self.body_vel_summary_dim = 2 * self.body_vel_dim
    fuse_in = enc_dim + self.body_vel_summary_dim + self.cmd_feature_dim
    self._head = self._build_fusion_head(
      fuse_in=fuse_in,
      hidden_size=hidden_size,
      output_size=output_size,
      depth=self.head_hidden_depth,
      dropout=self.head_dropout,
    )

  def _build_fusion_head(
    self,
    *,
    fuse_in: int,
    hidden_size: int,
    output_size: int,
    depth: int,
    dropout: float,
  ) -> nn.Sequential:
    if depth < 1:
      raise ValueError("head_hidden_depth must be >= 1")
    layers: list[nn.Module] = []
    d_in = fuse_in
    for _ in range(depth):
      layers.append(nn.Linear(d_in, hidden_size))
      layers.append(self._act())
      if dropout > 0:
        layers.append(nn.Dropout(dropout))
      d_in = hidden_size
    layers.append(nn.Linear(d_in, output_size))
    return nn.Sequential(*layers)

  def _encode_lower_sequence(self, leg_seq: torch.Tensor) -> torch.Tensor:
    """leg_seq: (B, H, C) normalized."""
    _out, h_n = self._gru(leg_seq)
    return h_n[-1]

  def _normalize_body_vel(self, v: torch.Tensor) -> torch.Tensor:
    m = self.body_vel_mean.view(1, 1, -1)
    s = self.body_vel_std.view(1, 1, -1)
    return (v - m) / s

  def _leg_sequence_from_state(
    self,
    lower_pos_hist: torch.Tensor | None,
    lower_vel_hist: torch.Tensor | None,
    foot_pos_hist: torch.Tensor | None,
    foot_vel_hist: torch.Tensor | None,
  ) -> torch.Tensor:
    """Build normalized GRU input (B, H, C) from ``history_input_mode``."""
    mode = self.history_input_mode
    if mode == "joints":
      if lower_pos_hist is None:
        raise ValueError("lower_pos_hist is required when history_input_mode='joints'")
      if foot_pos_hist is not None:
        raise ValueError("foot_pos_hist must be None when history_input_mode='joints'")
      pos = (lower_pos_hist - self.leg_pos_mean.view(1, 1, -1)) / self.leg_pos_std.view(
        1, 1, -1
      )
      pos = pos * self.joints_history_weight
      if self.use_lower_joint_velocity:
        if lower_vel_hist is None:
          raise ValueError("lower_vel_hist required when use_lower_joint_velocity is True")
        lv = (lower_vel_hist - self.leg_vel_mean.view(1, 1, -1)) / self.leg_vel_std.view(
          1, 1, -1
        )
        return torch.cat([pos, lv * self.joints_history_weight], dim=-1)
      return pos
    if mode == "feet":
      if foot_pos_hist is None:
        raise ValueError("foot_pos_hist is required when history_input_mode='feet'")
      if lower_pos_hist is not None:
        raise ValueError("lower_pos_hist must be None when history_input_mode='feet'")
      feet = (foot_pos_hist - self.foot_pos_mean.view(1, 1, -1)) / self.foot_pos_std.view(
        1, 1, -1
      )
      feet = feet * self.feet_history_weight
      if self.use_foot_velocity:
        if foot_vel_hist is None:
          raise ValueError("foot_vel_hist required when use_foot_velocity is True")
        fv = (foot_vel_hist - self.foot_vel_mean.view(1, 1, -1)) / self.foot_vel_std.view(
          1, 1, -1
        )
        return torch.cat([feet, fv * self.feet_history_weight], dim=-1)
      return feet
    # both
    if lower_pos_hist is None or foot_pos_hist is None:
      raise ValueError(
        "lower_pos_hist and foot_pos_hist are both required when history_input_mode='both'"
      )
    pos = (lower_pos_hist - self.leg_pos_mean.view(1, 1, -1)) / self.leg_pos_std.view(
      1, 1, -1
    )
    f = (foot_pos_hist - self.foot_pos_mean.view(1, 1, -1)) / self.foot_pos_std.view(1, 1, -1)
    parts: list[torch.Tensor] = [
      pos * self.joints_history_weight,
      f * self.feet_history_weight,
    ]
    if self.use_lower_joint_velocity:
      if lower_vel_hist is None:
        raise ValueError("lower_vel_hist required when use_lower_joint_velocity is True")
      lv = (lower_vel_hist - self.leg_vel_mean.view(1, 1, -1)) / self.leg_vel_std.view(
        1, 1, -1
      )
      parts.append(lv * self.joints_history_weight)
    if self.use_foot_velocity:
      if foot_vel_hist is None:
        raise ValueError("foot_vel_hist required when use_foot_velocity is True")
      fv = (foot_vel_hist - self.foot_vel_mean.view(1, 1, -1)) / self.foot_vel_std.view(
        1, 1, -1
      )
      parts.append(fv * self.feet_history_weight)
    return torch.cat(parts, dim=-1)

  def forward(
    self,
    lower_pos_hist: torch.Tensor | None,
    foot_pos_hist: torch.Tensor | None,
    body_vel_hist: torch.Tensor,
    body_vel_future: torch.Tensor,
    lower_vel_hist: torch.Tensor | None = None,
    foot_vel_hist: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Args:
    lower_pos_hist: (B, H, N_lower) or None if ``history_input_mode`` is ``feet``
    foot_pos_hist: (B, H, 6) [left_xyz, right_xyz] in root frame, or None if ``joints``
    body_vel_hist: (B, H, D_vel)
    body_vel_future: (B, D_cmd) command features ([root_vx,root_vy,root_wz] + optional trajectory block)
    lower_vel_hist: (B, H, N_lower) if ``use_lower_joint_velocity``
    foot_vel_hist: (B, H, 6) if ``use_foot_velocity``
    """
    leg_seq = self._leg_sequence_from_state(
      lower_pos_hist, lower_vel_hist, foot_pos_hist, foot_vel_hist
    )
    z = self._encode_lower_sequence(leg_seq)

    bv_hist = self._normalize_body_vel(body_vel_hist)
    bv_mean = bv_hist.mean(dim=1)
    bv_last = bv_hist[:, -1, :]
    bv = torch.cat([bv_mean, bv_last], dim=1)
    bf = (body_vel_future - self.cmd_mean.view(1, -1)) / self.cmd_std.view(1, -1)

    fused = torch.cat([z, bv, bf], dim=1)
    return self._head(fused) * self.y_std + self.y_mean

  @torch.no_grad()
  def predict(
    self,
    lower_pos_hist: torch.Tensor | None,
    foot_pos_hist: torch.Tensor | None,
    body_vel_hist: torch.Tensor,
    body_vel_future: torch.Tensor,
    lower_vel_hist: torch.Tensor | None = None,
    foot_vel_hist: torch.Tensor | None = None,
  ) -> torch.Tensor:
    return self.forward(
      lower_pos_hist,
      foot_pos_hist,
      body_vel_hist,
      body_vel_future,
      lower_vel_hist,
      foot_vel_hist,
    )
