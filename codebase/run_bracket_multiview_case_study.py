#!/usr/bin/env python3
"""Matched top-only vs consistent-side-view bracket sparse refinement study."""
from __future__ import annotations

import concurrent.futures
import json
import shutil
import subprocess
from pathlib import Path

import pyvista as pv
import trimesh

from run_connectivity_qd_sampling import GENERATOR, PYTHON, ROOT, generation_env


BASE = ROOT / "experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22"
EVAL = BASE / "shared_evaluations"
OUT = BASE / "multiview_case_study_2026-09-23"
CASES = {
    "swept": ("round_02/r2_outline_swept__medium/fea_on_bc6_framefix", "round_02/outline_probe/inputs/r2_outline_swept__medium"),
    "scalloped": ("round_02/r2_outline_scalloped__medium/fea_on_bc6_framefix", "round_02/outline_probe/inputs/r2_outline_scalloped__medium"),
    "spine": ("round_05/r5_longitudinal_spine__medium/fea_on_bc6_framefix", "round_05/novel_semantic/inputs/r5_longitudinal_spine__medium"),
}


def render_side(mesh_path: Path, path: Path, sign: int) -> None:
    mesh = pv.read(mesh_path)
    center = mesh.center
    bounds = mesh.bounds
    width_y = bounds[3] - bounds[2]
    plotter = pv.Plotter(off_screen=True, window_size=(512, 512))
    plotter.set_background("white")
    plotter.add_mesh(mesh, color="#747b80", smooth_shading=True, ambient=0.6, diffuse=0.4, specular=0.08)
    plotter.camera_position = [(center[0] + sign * 1.0, center[1], center[2]), center, (0, 0, 1)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = width_y / 1.8
    plotter.screenshot(str(path), transparent_background=True)
    plotter.close()


def prepare(name: str, source: Path, image_source: Path) -> dict:
    case = OUT / name
    input_dir = case / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image_source / "그림1.png", input_dir / "그림1.png")
    render_side(source / "dense_cache_mesh.obj", input_dir / "side_xminus.png", -1)
    render_side(source / "dense_cache_mesh.obj", input_dir / "side_xplus.png", +1)
    cfg = json.loads((source / "config.json").read_text())
    cfg["name"] = f"{name}_matched_multiview"
    cfg["views"] = "그림1,side_xminus,side_xplus"
    cfg["stages"]["mesh"].update({
        "n_views": 3,
        "load_dense_cache": str((source / "dense_cache.npz").resolve()),
        "save_dense_cache": None,
    })
    config = case / "config.json"
    config.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    return {"name": name, "source": str(source.resolve()), "input": str(input_dir.resolve()),
            "config": str(config.resolve()), "generation": str((case / "generation").resolve()),
            "baseline_mesh": str((source / "gen/mesh.obj").resolve())}


def generate(row: dict, gpu: int) -> dict:
    gen = Path(row["generation"])
    gen.mkdir(parents=True, exist_ok=True)
    log = gen.parent / "generation.log"
    cmd = [str(PYTHON), str(GENERATOR), "--config", row["config"],
           "--target-dir", row["input"], "--out", str(gen)]
    env = generation_env(gpu)
    env["FEA_MAX_NODE_MAP_DISTANCE"] = "0.02"
    with log.open("w") as stream:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    row.update(gpu=gpu, exit_code=result.returncode, log=str(log.resolve()),
               multiview_mesh=str((gen / "mesh.obj").resolve()))
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [prepare(name, EVAL / src, BASE / "shared_image_pool" / img)
            for name, (src, img) in CASES.items()]
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(generate, row, gpu) for row, gpu in zip(rows, (0, 5, 6))]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            print(row["name"], row["exit_code"], row["log"], flush=True)
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
