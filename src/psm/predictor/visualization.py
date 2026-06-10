"""
Viser + MuJoCo visualization for psm playback.

Solid robot = recorded (ground-truth) trajectory; optional ghost = predictor output.

Used by ``psm.predictor.play`` after receding-horizon prediction; keeps viewer code
separate from model loading and rollout.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np
import viser
from mjlab.viewer.viser import ViserMujocoScene
from mjlab.viewer.viser.term_plotter import ViserTermPlotter


def run_viser_visualization(
    model_xml: str,
    original_qpos: np.ndarray,
    predicted_qpos: np.ndarray,
    fps: float,
    body_vel_series: np.ndarray,
    body_features: dict[str, np.ndarray] | None = None,
) -> None:
    """Interactive browser viewer: solid robot = actual NPZ trajectory; ghost = prediction."""
    model = mujoco.MjModel.from_xml_path(model_xml)
    data_pred = mujoco.MjData(model)

    # Separate model for ghost so we can tint it light red without affecting main robot.
    ghost_model = mujoco.MjModel.from_xml_path(model_xml)
    ghost_rgba = ghost_model.geom_rgba.copy()
    # Light red tint: increase R, lower G/B, keep alpha 1.0
    ghost_rgba[:, 0] = 1.0
    ghost_rgba[:, 1] *= 0.6
    ghost_rgba[:, 2] *= 0.6
    ghost_model.geom_rgba[:] = ghost_rgba

    server = viser.ViserServer(label="Arm matching (mjlab viewer)")
    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.create_visualization_gui()
    scene.debug_visualization_enabled = True

    # Optional time-series plots for body features (e.g., step_length, step_width).
    feature_names = []
    if body_features is not None:
        feature_names = sorted(body_features.keys())
    plotter = None
    if feature_names:
        plotter = ViserTermPlotter(
            server,
            term_names=feature_names,
            name="Body features",
            history_length=200,
        )

    num_frames = original_qpos.shape[0]
    dt = 1.0 / fps
    current_frame = 0
    playing = False
    show_ghost = True
    ghost_qpos_cache = np.zeros(model.nq, dtype=original_qpos.dtype)
    vel_local = np.zeros(3, dtype=np.float64)

    def _render_frame(frame_idx: int) -> None:
        data_pred.qpos[:] = original_qpos[frame_idx]
        mujoco.mj_forward(model, data_pred)

        scene.clear()
        vx, vy, _wz = body_vel_series[frame_idx]
        vel_local[0] = vx
        vel_local[1] = vy
        vel_local[2] = 0.0
        root_body_id = scene._tracked_body_id or 1  # type: ignore[attr-defined]
        root_pos = np.asarray(data_pred.xpos[root_body_id], dtype=np.float64)
        root_mat = np.asarray(data_pred.xmat[root_body_id], dtype=np.float64).reshape(3, 3)
        vel_world = root_mat @ vel_local
        scene.add_arrow(
            start=root_pos,
            end=root_pos + vel_world * scene.meansize,
            color=(0.1, 0.8, 0.1, 1.0),
            width=0.02,
            label="cmd_vel",
        )
        if show_ghost:
            np.copyto(ghost_qpos_cache, predicted_qpos[frame_idx])
            scene.add_ghost_mesh(
                qpos=ghost_qpos_cache,
                model=ghost_model,
                alpha=0.4,
                label="predicted",
            )
        scene.update_from_mjdata(data_pred)

        if plotter is not None and body_features is not None:
            terms = []
            for name in feature_names:
                series = body_features[name]
                if frame_idx < len(series):
                    terms.append((name, np.array([series[frame_idx]], dtype=np.float64)))
            if terms:
                plotter.update(terms)

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(0, num_frames - 1),
            step=1,
            initial_value=0,
        )
        play_button = server.gui.add_button("Play/Pause")
        ghost_checkbox = server.gui.add_checkbox(
            "Show prediction (ghost)",
            initial_value=True,
            hint="Toggle the predictor trajectory as a transparent overlay.",
        )

    @play_button.on_click
    def _(_) -> None:
        nonlocal playing
        playing = not playing

    @ghost_checkbox.on_update
    def _(_) -> None:
        nonlocal show_ghost
        show_ghost = bool(ghost_checkbox.value)
        _render_frame(current_frame)

    @frame_slider.on_update
    def _(_) -> None:
        nonlocal current_frame
        current_frame = int(frame_slider.value)
        _render_frame(current_frame)

    _render_frame(0)
    current_frame = 0
    frame_slider.value = 0

    print("Viser server running; open the printed URL in your browser.")

    last_time = time.perf_counter()
    while True:
        now = time.perf_counter()
        if playing and now - last_time >= dt:
            current_frame = (current_frame + 1) % num_frames
            frame_slider.value = current_frame
            last_time = now
