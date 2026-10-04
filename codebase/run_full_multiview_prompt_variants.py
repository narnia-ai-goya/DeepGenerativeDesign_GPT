#!/usr/bin/env python3
"""Run the proven angular bracket recipe on three text-directed image variants."""
from __future__ import annotations

import concurrent.futures
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import trimesh

from run_bracket_multiview_case_study import render_side
from run_connectivity_qd_sampling import GENERATOR, PYTHON, ROOT, generation_env
from run_semantic_qd_sparse_round import transform_matrix


STUDY = ROOT / "experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22"
ANGULAR = STUDY / "shape_language_angular_2026-09-23"
TEMPLATE = ANGULAR / "full_multiview_angular_2026-09-23/config.json"
IMAGE_POOL = STUDY / "shared_image_pool/round_02/outline_probe/inputs"
SOURCE_EVAL = STUDY / "shared_evaluations/round_02"
OUT = STUDY / "full_multiview_prompt_variants_2026-09-23"
CASES = {
    "swept": ("r2_outline_swept__medium", "Swept branching ribs with elongated diagonal openings"),
    "scalloped": ("r2_outline_scalloped__medium", "Scalloped outer contour with ring-like central opening"),
    "fork": ("r2_outline_fork__medium", "Forked arms around a large central void"),
    "tapered": ("r2_outline_tapered__medium", "Tapered radial ribs converging near the load interface"),
}


def prepare() -> list[dict]:
    template = json.loads(TEMPLATE.read_text())
    rows = []
    for name, (source_id, descriptor) in CASES.items():
        case = OUT / name
        input_dir = case / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        source_image = IMAGE_POOL / source_id / "그림1.png"
        source_dense = SOURCE_EVAL / source_id / "fea_on_bc6_framefix/dense_cache_mesh.obj"
        shutil.copyfile(source_image, input_dir / "그림1.png")
        render_side(source_dense, input_dir / "side_xminus.png", -1)
        render_side(source_dense, input_dir / "side_xplus.png", +1)
        cfg = json.loads(json.dumps(template))
        cfg["name"] = f"full_multiview_prompt_{name}"
        cfg["stages"]["mesh"].update({
            "load_dense_cache": None,
            "save_dense_cache": str((case / "dense_cache.npz").resolve()),
        })
        config = case / "config.json"
        config.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
        rows.append({"id": name, "source_id": source_id, "descriptor": descriptor,
                     "descriptor_note": "style label; exact original image-generation prompt was not archived",
                     "source_image": str(source_image.resolve()),
                     "source_side_geometry": str(source_dense.resolve()),
                     "config": str(config.resolve()), "input_dir": str(input_dir.resolve()),
                     "generation": str((case / "generation").resolve()),
                     "final": str((case / "post/final.obj").resolve())})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    return rows


def run_generation(row: dict, gpu: int) -> tuple[str, int]:
    case = OUT / row["id"]
    gen = case / "generation"
    gen.mkdir(exist_ok=True)
    if (gen / "mesh.obj").exists():
        return row["id"], 0
    env = generation_env(gpu)
    env["FEA_MAX_NODE_MAP_DISTANCE"] = "0.02"
    cmd = [str(PYTHON), str(GENERATOR), "--config", row["config"],
           "--target-dir", row["input_dir"], "--out", str(gen)]
    with (case / "generation.log").open("w") as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return row["id"], result.returncode


def run_post(row: dict) -> None:
    case = OUT / row["id"]
    post = case / "post"
    post.mkdir(exist_ok=True)
    if (post / "final.obj").exists():
        return
    raw = trimesh.load(case / "generation/mesh.obj", force="mesh", process=False)
    vertices = np.c_[raw.vertices, np.ones(len(raw.vertices))] @ transform_matrix()
    trimesh.Trimesh(vertices, raw.faces, process=False).export(post / "mesh_physical_aligned.obj")
    for script, source, output, log_name in (
        ("post_hybrid_union_clip.py", "mesh_physical_aligned.obj", "mesh_bc_preserved.obj", "boolean.log"),
        ("surface_remesh_pre.py", "mesh_bc_preserved.obj", "final.obj", "remesh.log"),
    ):
        cmd = [str(PYTHON), str(ROOT / "codebase/code" / script), "--config", row["config"],
               "--in", str(post / source), "--out", str(post / output)]
        with (post / log_name).open("w") as log:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{row['id']} {script} failed: {post / log_name}")


def main() -> None:
    rows = prepare()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_generation, row, gpu) for row, gpu in zip(rows, (0, 5, 6, 7))]
        for future in concurrent.futures.as_completed(futures):
            name, code = future.result()
            print(f"generation {name}: exit={code}", flush=True)
            if code:
                raise RuntimeError(f"generation failed: {name}")
    for row in rows:
        run_post(row)
        print(f"post {row['id']}: {row['final']}", flush=True)


if __name__ == "__main__":
    main()
