#!/usr/bin/env python3
"""
Play back NPZ motions in the browser (Viser + mjviser).

Run::

  psm-vis-npz --npz data/motions/your_clip.npz
  psm-vis-npz --motion-dir data/motions
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh
import viser
from mjviser import ViserMujocoScene

from psm.predictor.npz_schema import list_clip_npz_files

def _repo_root() -> Path:
    import psm

    return Path(psm.__file__).resolve().parent.parent.parent


_DEFAULT_DATA_DIR = _repo_root() / "data" / "motions"


def _default_model_path() -> Path | None:
    from psm.assets.unitree_g1.g1_constants import G1_XML

    return G1_XML if G1_XML.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NPZ motion playback in Viser (mjviser ViserMujocoScene)."
    )
    parser.add_argument("--npz", type=str, default=None, help="Path to a single NPZ file")
    parser.add_argument(
        "--motion-dir",
        type=str,
        default=None,
        help=(
            "Directory with *.npz files; adds a GUI dropdown to switch clips. "
            f"Default: {_DEFAULT_DATA_DIR} if it exists."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="MuJoCo XML. Default: src/psm/assets/unitree_g1/xmls/g1.xml",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override trajectory FPS (else from NPZ or 50).",
    )
    args = parser.parse_args()

    motion_dir: Path | None = Path(args.motion_dir) if args.motion_dir else None
    if motion_dir is None and _DEFAULT_DATA_DIR.is_dir():
        motion_dir = _DEFAULT_DATA_DIR
    if motion_dir is not None and not motion_dir.is_dir():
        parser.error(f"--motion-dir is not a directory: {motion_dir}")

    npz_path: Path | None = Path(args.npz) if args.npz else None
    if npz_path is not None and not npz_path.is_file():
        parser.error(f"NPZ file not found: {npz_path}")

    if npz_path is None and motion_dir is None:
        parser.error(
            "Provide --npz, or use a motion directory (default: data/motions if present), "
            "or pass --motion-dir."
        )

    model_path = args.model_path
    if model_path is None:
        default = _default_model_path()
        if default is None:
            parser.error(
                "No default g1.xml under psm.assets.unitree_g1. "
                "Pass --model-path."
            )
        model_path = str(default)
        print(f"Using model: {model_path}")

    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)

    def load_motion(npz_file: Path) -> tuple[np.ndarray, np.ndarray, int, float, float]:
        data_npz = np.load(npz_file, allow_pickle=True)
        qpos_local = np.asarray(data_npz["qpos"])
        if qpos_local.ndim != 2:
            raise ValueError(f"qpos must be 2D, got shape {qpos_local.shape}")
        if model.nq != qpos_local.shape[1]:
            raise ValueError(
                f"NPZ nq={qpos_local.shape[1]} does not match model nq={model.nq}."
            )
        num_frames_local = int(qpos_local.shape[0])
        fps_local = args.fps
        if fps_local is None:
            fps_local = float(data_npz["fps"][0]) if "fps" in data_npz else 50.0
        dt_local = 1.0 / float(fps_local)

        if "qvel" in data_npz:
            qvel_local = np.asarray(data_npz["qvel"])
            if qvel_local.shape[0] != num_frames_local or qvel_local.shape[1] != model.nv:
                qvel_local = np.zeros((num_frames_local, model.nv), dtype=np.float64)
        else:
            qvel_local = np.zeros((num_frames_local, model.nv), dtype=np.float64)
        return qpos_local, qvel_local, num_frames_local, float(fps_local), float(dt_local)

    if motion_dir is not None and npz_path is None:
        candidates = list_clip_npz_files(motion_dir)
        if not candidates:
            parser.error(f"No clip NPZ files in --motion-dir: {motion_dir}")
        npz_path = candidates[0]
    assert npz_path is not None

    print(f"Loading NPZ: {npz_path}")
    if motion_dir is not None:
        print(f"Motion directory: {motion_dir}")

    traj: dict[str, Any] = {}
    qpos0, qvel0, nf0, fps0, dt0 = load_motion(npz_path)
    traj.update(qpos=qpos0, qvel=qvel0, n_frames=nf0, fps=fps0, dt=dt0)

    meansize = float(model.stat.meansize)

    # Same structure as mjviser/examples/motion_playback.py: server → ViserMujocoScene → GUI.
    # Camera tracking moves ``scene.fixed_bodies_frame`` by ``-tracked_body_xpos``; put any
    # world-anchored helpers under ``/fixed_bodies/...`` so they follow (root ``/floor`` would
    # stay fixed in world space and break the illusion of tracking).
    server = viser.ViserServer(label="Arm matching — NPZ (mjviser)")
    scene = ViserMujocoScene(server, model, num_envs=1)
    server.scene.add_grid(
        "/fixed_bodies/reference_grid",
        infinite_grid=True,
        fade_distance=50.0,
        shadow_opacity=0.2,
        plane_opacity=0.35,
    )
    tabs = scene.create_visualization_gui()

    replay_data = mujoco.MjData(model)

    class _ArrowOverlay:
        """Root velocity arrows (vx, vy, wz) in root frame."""

        def __init__(self) -> None:
            shaft = trimesh.creation.cylinder(radius=1.0, height=1.0)
            shaft.apply_translation([0.0, 0.0, 0.5])
            head = trimesh.creation.cone(radius=2.0, height=1.0, sections=12)
            self._shaft_v = shaft.vertices
            self._shaft_f = shaft.faces
            self._head_v = head.vertices
            self._head_f = head.faces

            self.shaft_handle = server.scene.add_batched_meshes_simple(
                "/fixed_bodies/debug_root_vel/shaft",
                self._shaft_v,
                self._shaft_f,
                batched_positions=np.zeros((3, 3), dtype=np.float32),
                batched_wxyzs=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (3, 1)),
                batched_scales=np.ones((3, 3), dtype=np.float32),
                batched_colors=np.array(
                    [[255, 60, 60], [60, 255, 60], [80, 120, 255]], dtype=np.uint8
                ),
                opacity=0.9,
                cast_shadow=False,
                receive_shadow=False,
            )
            self.head_handle = server.scene.add_batched_meshes_simple(
                "/fixed_bodies/debug_root_vel/head",
                self._head_v,
                self._head_f,
                batched_positions=np.zeros((3, 3), dtype=np.float32),
                batched_wxyzs=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (3, 1)),
                batched_scales=np.ones((3, 3), dtype=np.float32),
                batched_colors=np.array(
                    [[255, 60, 60], [60, 255, 60], [80, 120, 255]], dtype=np.uint8
                ),
                opacity=0.9,
                cast_shadow=False,
                receive_shadow=False,
            )

        def update(
            self,
            *,
            origin: np.ndarray,
            root_quat_wxyz: np.ndarray,
            v_local: np.ndarray,
            w_local: np.ndarray,
        ) -> None:
            basis = np.eye(3, dtype=np.float64)
            dir_w = np.zeros((3, 3), dtype=np.float64)
            for i in range(3):
                mujoco.mju_rotVecQuat(dir_w[i], basis[i], root_quat_wxyz)

            mags = np.array([float(v_local[0]), float(v_local[1]), float(w_local[2])], dtype=np.float64)
            lin_gain = 0.8 * meansize
            ang_gain = 0.4 * meansize
            lengths = np.array(
                [
                    abs(mags[0]) * lin_gain,
                    abs(mags[1]) * lin_gain,
                    abs(mags[2]) * ang_gain,
                ],
                dtype=np.float64,
            )

            z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            wxyzs = np.zeros((3, 4), dtype=np.float32)
            positions_shaft = np.zeros((3, 3), dtype=np.float32)
            positions_head = np.zeros((3, 3), dtype=np.float32)
            scales_shaft = np.zeros((3, 3), dtype=np.float32)
            scales_head = np.zeros((3, 3), dtype=np.float32)

            for i in range(3):
                if lengths[i] < 1e-6:
                    wxyzs[i] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    positions_shaft[i] = origin.astype(np.float32)
                    positions_head[i] = origin.astype(np.float32)
                    scales_shaft[i] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    scales_head[i] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    continue

                sign = 1.0 if mags[i] >= 0.0 else -1.0
                d = dir_w[i] * sign
                cross = np.cross(z, d)
                dot = float(np.dot(z, d))
                if dot < -0.999999:
                    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    q = np.array([0.0, axis[0], axis[1], axis[2]], dtype=np.float64)
                else:
                    q = np.array([1.0 + dot, cross[0], cross[1], cross[2]], dtype=np.float64)
                    q = q / np.linalg.norm(q)
                wxyzs[i] = q.astype(np.float32)

                positions_shaft[i] = origin.astype(np.float32)
                scales_shaft[i] = np.array(
                    [0.03 * meansize, 0.03 * meansize, lengths[i]], dtype=np.float32
                )

                end = origin + d * lengths[i]
                positions_head[i] = end.astype(np.float32)
                scales_head[i] = np.array(
                    [0.06 * meansize, 0.06 * meansize, 0.12 * meansize],
                    dtype=np.float32,
                )

            self.shaft_handle.batched_positions = positions_shaft
            self.shaft_handle.batched_wxyzs = wxyzs
            self.shaft_handle.batched_scales = scales_shaft
            self.head_handle.batched_positions = positions_head
            self.head_handle.batched_wxyzs = wxyzs
            self.head_handle.batched_scales = scales_head

    arrow_overlay = _ArrowOverlay()

    # --- Playback state (same pattern as mjviser/examples/motion_playback.py) ---
    frame_idx = [0]
    playing = [True]
    speed = [1.0]
    looping = [True]
    accumulator = [0.0]
    needs_render = [True]

    def render_frame(idx: int) -> None:
        qpos_traj = traj["qpos"]
        qvel_traj = traj["qvel"]
        nf = traj["n_frames"]
        i = max(0, min(int(idx), nf - 1))
        replay_data.qpos[:] = qpos_traj[i]
        replay_data.qvel[:] = qvel_traj[i]
        mujoco.mj_forward(model, replay_data)
        scene.update_from_mjdata(replay_data)

        root_pos = np.asarray(replay_data.qpos[0:3], dtype=np.float64)
        root_quat = np.asarray(replay_data.qpos[3:7], dtype=np.float64)
        v_world = np.asarray(replay_data.qvel[0:3], dtype=np.float64)
        conj = np.empty(4, dtype=np.float64)
        mujoco.mju_negQuat(conj, root_quat)
        v_local = np.empty(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(v_local, v_world, conj)
        w_local = np.asarray(replay_data.qvel[3:6], dtype=np.float64)
        arrow_overlay.update(
            origin=root_pos, root_quat_wxyz=root_quat, v_local=v_local, w_local=w_local
        )

        dt_traj = traj["dt"]
        t = i * dt_traj
        span = traj["n_frames"] * dt_traj
        time_label.content = (
            f'<span style="font-size:0.85em">'
            f"{t:.2f}s / {span:.2f}s (frame {i}/{traj['n_frames'] - 1})"
            f"</span>"
        )

    with tabs.add_tab("Playback", icon=viser.Icon.PLAYER_PLAY):
        motion_dropdown = None
        if motion_dir is not None:
            motions = list_clip_npz_files(motion_dir)
            motion_labels = [p.name for p in motions]
            if len(motion_labels) > 1:
                initial_label = npz_path.name
                if initial_label not in motion_labels:
                    initial_label = motion_labels[0]
                motion_dropdown = server.gui.add_dropdown(
                    "Motion",
                    options=motion_labels,
                    initial_value=initial_label,
                    hint=str(motion_dir),
                )

        timeline = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(0, traj["n_frames"] - 1),
            step=1,
            initial_value=0,
        )
        time_label = server.gui.add_html("")

        play_btn = server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)

        @play_btn.on_click
        def _(_) -> None:
            playing[0] = not playing[0]
            play_btn.label = "Pause" if playing[0] else "Play"
            play_btn.icon = (
                viser.Icon.PLAYER_PAUSE if playing[0] else viser.Icon.PLAYER_PLAY
            )

        speed_btns = server.gui.add_button_group(
            "Speed", options=["0.25x", "0.5x", "1x", "2x", "4x"]
        )

        @speed_btns.on_click
        def _(event) -> None:
            speed[0] = float(event.target.value.replace("x", ""))

        loop_cb = server.gui.add_checkbox("Loop", initial_value=True)

        @loop_cb.on_update
        def _(_) -> None:
            looping[0] = bool(loop_cb.value)

        @timeline.on_update
        def _(_) -> None:
            frame_idx[0] = int(timeline.value)
            needs_render[0] = True

        if motion_dropdown is not None:

            @motion_dropdown.on_update
            def _(event) -> None:
                selected = event.target.value
                selected_path = motion_dir / selected
                try:
                    qn, qvn, nn, fps_n, dt_n = load_motion(selected_path)
                except Exception as e:
                    print(f"Failed to load {selected_path}: {e}", file=sys.stderr)
                    return
                traj.clear()
                traj.update(qpos=qn, qvel=qvn, n_frames=nn, fps=fps_n, dt=dt_n)
                frame_idx[0] = 0
                playing[0] = False
                play_btn.label = "Play"
                play_btn.icon = viser.Icon.PLAYER_PLAY
                timeline.max = max(0, traj["n_frames"] - 1)
                timeline.value = 0
                needs_render[0] = True

    render_frame(0)

    print("\nViser server running. Open the URL printed above. Ctrl+C to exit.\n")

    last_time = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            wall_dt = now - last_time
            last_time = now

            nf = traj["n_frames"]
            dt = traj["dt"]

            if playing[0]:
                accumulator[0] += wall_dt * speed[0]
                frames_to_advance = int(accumulator[0] / dt)
                if frames_to_advance > 0:
                    accumulator[0] -= frames_to_advance * dt
                    new_idx = frame_idx[0] + frames_to_advance
                    if new_idx >= nf:
                        if looping[0]:
                            new_idx = new_idx % nf
                        else:
                            new_idx = nf - 1
                            playing[0] = False
                            play_btn.label = "Play"
                            play_btn.icon = viser.Icon.PLAYER_PLAY
                    frame_idx[0] = new_idx
                    timeline.value = new_idx
                    render_frame(new_idx)
            elif needs_render[0]:
                render_frame(frame_idx[0])
                needs_render[0] = False

            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        print("\nStopped.")
        server.stop()


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
