"""Convert G1 motion data (CSV or GMR PKL) to extended NPZ (FK + PSM features)."""

from __future__ import annotations

from pathlib import Path

import torch
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

from psm.motion.conversion import (
    debias_log_vertical,
    g1_conversion_scene,
    g1_joint_names,
    load_gmr_pkl,
    peek_csv_dof_width,
    resolve_gmr_joint_names,
    resolve_input_motion_paths,
    run_motion_fk,
    tyro_cli,
)
from psm.predictor.npz_schema import resolve_conversion_paths, write_extended_clip_from_log


def _effective_input_fps(
    motion_path: Path,
    *,
    input_fps: float | None,
    default_csv_fps: float,
) -> float:
    if motion_path.suffix.lower() == ".pkl":
        if input_fps is not None:
            return float(input_fps)
        return float(load_gmr_pkl(motion_path)["fps"])
    return float(input_fps if input_fps is not None else default_csv_fps)


def main(
    input_path: str | None = None,
    output_dir: str | None = None,
    dataset: str | None = None,
    dataset_path: str | None = None,
    input_fps: float | None = None,
    output_fps: float = 50.0,
    device: str = "cuda:0",
    render: bool = False,
    line_range: tuple[int, int] | None = None,
    debias_z: bool = False,
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING] CUDA unavailable, using CPU.")
        device = "cpu"

    input_path, output_dir = resolve_conversion_paths(
        dataset=dataset,
        dataset_path=dataset_path,
        input_path=input_path,
        output_dir=output_dir,
    )
    print(f"[INFO] input={input_path} output={output_dir}")
    motion_files = resolve_input_motion_paths(input_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / output_fps
    scene = Scene(g1_conversion_scene(), device=device)
    model = scene.compile()
    joint_names = g1_joint_names(model)

    first = motion_files[0]
    if first.suffix.lower() == ".csv":
        if peek_csv_dof_width(first, line_range) != len(joint_names):
            raise ValueError("CSV joint count does not match G1 model")
    else:
        resolve_gmr_joint_names(load_gmr_pkl(first), model)

    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    body_names = [str(n) for n in scene["robot"].body_names]

    renderer = None
    if render:
        viewer_cfg = ViewerConfig(
            height=480, width=640,
            origin_type=ViewerConfig.OriginType.ASSET_ROOT,
            entity_name="robot", distance=2.0, elevation=-5.0, azimuth=20,
        )
        renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
        renderer.initialize()

    for motion_path in motion_files:
        pkl_joint_names: list[str] | None = None
        model_joint_names: list[str] | None = None
        if motion_path.suffix.lower() == ".pkl":
            model_joint_names, pkl_joint_names = resolve_gmr_joint_names(
                load_gmr_pkl(motion_path), model
            )
        clip_input_fps = _effective_input_fps(
            motion_path, input_fps=input_fps, default_csv_fps=30.0
        )
        print(f"[INFO] Converting {motion_path.name} (input_fps={clip_input_fps})")
        log = run_motion_fk(
            sim, scene,
            joint_names=joint_names,
            body_names=body_names,
            motion_path=motion_path,
            input_fps=clip_input_fps,
            output_fps=output_fps,
            line_range=line_range if motion_path.suffix.lower() == ".csv" else None,
            renderer=renderer,
            pkl_joint_names=pkl_joint_names,
            model_joint_names=model_joint_names,
        )
        if debias_z:
            print(f"[INFO] debias_z={debias_log_vertical(log, body_names):.6f} m")
        write_extended_clip_from_log(
            output_dir=output_dir,
            source_path=motion_path,
            log=log,
            joint_names=joint_names,
            body_names=body_names,
            fps=float(output_fps),
        )

    print(f"[INFO] Finished {len(motion_files)} clip(s)")


def cli() -> None:
    tyro_cli(main, bool_shorthand=("debias_z",))


if __name__ == "__main__":
    cli()
