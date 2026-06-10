"""Convert G1 CSV motions to extended NPZ (FK + PSM features)."""

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
    peek_csv_dof_width,
    run_csv_fk,
    tyro_cli,
)
from psm.predictor.npz_schema import resolve_conversion_paths, write_extended_clip_from_log


def _csv_paths(input_path: str) -> list[Path]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            raise ValueError(f"No .csv files in {path}")
        return files
    raise ValueError(f"Input path does not exist: {path}")


def main(
    input_path: str | None = None,
    output_dir: str | None = None,
    dataset: str | None = None,
    dataset_path: str | None = None,
    input_fps: float = 30.0,
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
    csv_files = _csv_paths(input_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / output_fps
    scene = Scene(g1_conversion_scene(), device=device)
    model = scene.compile()
    joint_names = g1_joint_names(model)
    if peek_csv_dof_width(csv_files[0], line_range) != len(joint_names):
        raise ValueError("CSV joint count does not match G1 model")

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

    for csv_path in csv_files:
        log = run_csv_fk(
            sim, scene,
            joint_names=joint_names,
            body_names=body_names,
            csv_path=str(csv_path),
            input_fps=input_fps,
            output_fps=output_fps,
            line_range=line_range,
            renderer=renderer,
        )
        if debias_z:
            print(f"[INFO] debias_z={debias_log_vertical(log, body_names):.6f} m")
        write_extended_clip_from_log(
            output_dir=output_dir,
            source_path=csv_path,
            log=log,
            joint_names=joint_names,
            body_names=body_names,
            fps=float(output_fps),
        )

    print(f"[INFO] Finished {len(csv_files)} clip(s)")


def cli() -> None:
    tyro_cli(main, bool_shorthand=("debias_z",))


if __name__ == "__main__":
    cli()
