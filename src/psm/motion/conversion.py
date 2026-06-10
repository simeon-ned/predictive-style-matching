"""G1 CSV → NPZ forward kinematics (mjlab/mjwarp)."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from tqdm import tqdm

from psm.assets.unitree_g1.g1_constants import (
    MOTION_Z_DEBIAS_FOOT_BODY_NAMES,
    MOTION_Z_DEBIAS_FOOT_SOLE_Z,
    get_g1_robot_cfg,
)
from psm.predictor.npz_schema import body_pose_vel_in_root_frame


def tyro_cli(fn: Callable[..., None], /, *, bool_shorthand: tuple[str, ...] = ()) -> None:
    import tyro
    import mjlab

    argv = list(sys.argv[1:])
    for name in bool_shorthand:
        flag = f"--{name.replace('_', '-')}"
        i = 0
        while i < len(argv):
            if argv[i] == flag and (i + 1 >= len(argv) or argv[i + 1].startswith("-")):
                argv.insert(i + 1, "True")
                i += 2
            else:
                i += 1
    tyro.cli(fn, config=mjlab.TYRO_FLAGS, args=argv)


def g1_conversion_scene() -> SceneCfg:
    cfg = SceneCfg(entities={"robot": get_g1_robot_cfg()})
    cfg.num_envs = 1
    return cfg


def g1_joint_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if name.startswith("robot/"):
            name = name[len("robot/") :]
        names.append(name)
    if len(names) != model.nq - 7:
        raise RuntimeError(f"Expected {model.nq - 7} joint names, got {len(names)}")
    return names


def peek_csv_dof_width(csv_path: Path, line_range: tuple[int, int] | None = None) -> int:
    if line_range is None:
        row = np.loadtxt(csv_path, delimiter=",", maxrows=1)
    else:
        row = np.loadtxt(csv_path, delimiter=",", skiprows=line_range[0] - 1, maxrows=1)
    row = np.atleast_1d(np.asarray(row, dtype=np.float64))
    if row.size < 8:
        raise ValueError(f"CSV {csv_path} needs root(7)+joints, got {row.size}")
    return int(row.size - 7)


def debias_log_vertical(log: dict[str, Any], body_names: list[str]) -> float:
    idx = {n: i for i, n in enumerate(body_names)}
    foot_i = [idx[n] for n in MOTION_Z_DEBIAS_FOOT_BODY_NAMES]
    body_pos_w = np.asarray(log["body_pos_w"], dtype=np.float32).copy()
    z_shift = float(np.min(body_pos_w[:, foot_i, 2])) - MOTION_Z_DEBIAS_FOOT_SOLE_Z
    body_pos_w[:, :, 2] -= z_shift
    log["body_pos_w"] = body_pos_w
    body_pos_r = np.asarray(log["body_pos_r"], dtype=np.float64).copy()
    body_pos_r[:, :, 2] -= z_shift
    log["body_pos_r"] = body_pos_r
    qpos = np.asarray(log["qpos"], dtype=np.float64).copy()
    qpos[:, 2] -= z_shift
    log["qpos"] = qpos
    return z_shift


class _CsvMotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device | str,
        line_range: tuple[int, int] | None = None,
    ):
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        self.current_idx = 0
        self.device = device
        if line_range is None:
            motion = torch.from_numpy(np.loadtxt(motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(
                np.loadtxt(
                    motion_file,
                    delimiter=",",
                    skiprows=line_range[0] - 1,
                    max_rows=line_range[1] - line_range[0] + 1,
                )
            )
        motion = motion.to(torch.float32).to(device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]
        self.motion_dof_poss_input = motion[:, 7:]
        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        times = torch.arange(0, self.duration, self.output_dt, device=device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        phase = times / self.duration
        i0 = (phase * (self.input_frames - 1)).floor().long()
        i1 = torch.minimum(i0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - i0
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[i0], self.motion_base_poss_input[i1], blend.unsqueeze(1)
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[i0], self.motion_base_rots_input[i1], blend
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[i0], self.motion_dof_poss_input[i1], blend.unsqueeze(1)
        )
        self.motion_base_lin_vels = torch.gradient(
            self.motion_base_poss, spacing=self.output_dt, dim=0
        )[0]
        self.motion_dof_vels = torch.gradient(
            self.motion_dof_poss, spacing=self.output_dt, dim=0
        )[0]
        q_prev, q_next = self.motion_base_rots[:-2], self.motion_base_rots[2:]
        omega = axis_angle_from_quat(quat_mul(q_next, quat_conjugate(q_prev))) / (
            2.0 * self.output_dt
        )
        self.motion_base_ang_vels = torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    @staticmethod
    def _lerp(a, b, blend):
        return a * (1 - blend) + b * blend

    def _slerp(self, a, b, blend):
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], float(blend[i]))
        return out

    def next_state(self):
        s = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        done = self.current_idx >= self.output_frames
        if done:
            self.current_idx = 0
        return s, done


def _init_log(output_fps: float, joint_names: list[str], body_names: list[str]) -> dict[str, Any]:
    keys = (
        "qpos", "qvel", "joint_pos", "joint_vel",
        "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
        "body_pos_r", "body_quat_r", "body_lin_vel_r", "body_ang_vel_r",
    )
    log: dict[str, Any] = {"fps": [output_fps], "joint_names": joint_names, "body_names": body_names}
    for k in keys:
        log[k] = []
    return log


def _append_frame(log: dict[str, Any], robot: Entity) -> None:
    qpos = robot.data.data.qpos[0].detach().cpu().numpy().copy()
    qvel = robot.data.data.qvel[0].detach().cpu().numpy().copy()
    joint_pos = robot.data.joint_pos[0].detach().cpu().numpy().copy()
    joint_vel = robot.data.joint_vel[0].detach().cpu().numpy().copy()
    body_pos_w = robot.data.body_link_pos_w[0].detach().cpu().numpy().copy()
    body_quat_w = robot.data.body_link_quat_w[0].detach().cpu().numpy().copy()
    body_lin_vel_w = robot.data.body_link_lin_vel_w[0].detach().cpu().numpy().copy()
    body_ang_vel_w = robot.data.body_link_ang_vel_w[0].detach().cpu().numpy().copy()
    root_pos = robot.data.root_link_pos_w[0].detach().cpu().numpy()
    root_quat = robot.data.root_link_quat_w[0].detach().cpu().numpy()
    root_lin_vel = robot.data.root_link_lin_vel_w[0].detach().cpu().numpy()
    root_ang_vel = robot.data.root_link_ang_vel_w[0].detach().cpu().numpy()
    pos_r, quat_r, lin_r, ang_r = body_pose_vel_in_root_frame(
        root_pos, root_quat, root_lin_vel, root_ang_vel,
        body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w,
    )
    log["qpos"].append(qpos)
    log["qvel"].append(qvel)
    log["joint_pos"].append(joint_pos)
    log["joint_vel"].append(joint_vel)
    log["body_pos_w"].append(body_pos_w)
    log["body_quat_w"].append(body_quat_w)
    log["body_lin_vel_w"].append(body_lin_vel_w)
    log["body_ang_vel_w"].append(body_ang_vel_w)
    log["body_pos_r"].append(pos_r)
    log["body_quat_r"].append(quat_r)
    log["body_lin_vel_r"].append(lin_r)
    log["body_ang_vel_r"].append(ang_r)


def _finalize_log(log: dict[str, Any]) -> None:
    for key in (
        "qpos", "qvel", "joint_pos", "joint_vel",
        "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
        "body_pos_r", "body_quat_r", "body_lin_vel_r", "body_ang_vel_r",
    ):
        log[key] = np.stack(log[key], axis=0)


def run_csv_fk(
    sim: Simulation,
    scene: Scene,
    *,
    joint_names: list[str],
    body_names: list[str],
    csv_path: str,
    input_fps: float,
    output_fps: float,
    line_range: tuple[int, int] | None,
    renderer: OffscreenRenderer | None,
) -> dict[str, Any]:
    motion = _CsvMotionLoader(csv_path, int(input_fps), int(output_fps), sim.device, line_range)
    robot: Entity = scene["robot"]
    j_idx = robot.find_joints(joint_names, preserve_order=True)[0]
    log = _init_log(float(output_fps), joint_names, body_names)
    scene.reset()
    pbar = tqdm(total=motion.output_frames, desc=f"FK {Path(csv_path).name}", ncols=100)
    done = False
    while not done:
        (bp, br, blv, bav, jp, jv), done = motion.next_state()
        root = robot.data.default_root_state.clone()
        root[:, 0:3] = bp
        root[:, :2] += scene.env_origins[:, :2]
        root[:, 3:7] = br
        root[:, 7:10] = blv
        root[:, 10:] = bav
        robot.write_root_state_to_sim(root)
        qpos = robot.data.default_joint_pos.clone()
        qvel = robot.data.default_joint_vel.clone()
        qpos[:, j_idx] = jp
        qvel[:, j_idx] = jv
        robot.write_joint_state_to_sim(qpos, qvel)
        sim.forward()
        scene.update(sim.mj_model.opt.timestep)
        if renderer is not None:
            renderer.update(sim.data)
            renderer.render()
        _append_frame(log, robot)
        pbar.update(1)
    pbar.close()
    _finalize_log(log)
    return log
