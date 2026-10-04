#!/usr/bin/env python3
"""Run deduplicated dense 3D evaluations for one Semantic BO-QD round."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess

from run_connectivity_qd_sampling import (BASE, GENERATOR, PYTHON, ROOT, dump,
                                           generation_env, load_mesh, render_mesh)


def prepare(experiment: Path, round_index: int) -> list[dict]:
    pool = experiment / "shared_image_pool" / f"round_{round_index:02d}"
    records = json.loads((pool / "unique_dense_jobs.json").read_text())["records"]
    base = json.loads((BASE / "config.json").read_text())
    protocol = json.loads((experiment / "protocol_snapshot.json").read_text())
    physical = protocol["absolute_paths"]
    jobs = []
    for record in records:
        case = experiment / "shared_evaluations" / f"round_{round_index:02d}" / record["id"]
        gen = case / "gen"; gen.mkdir(parents=True, exist_ok=True)
        config = json.loads(json.dumps(base))
        config["name"] = f"semantic_qd_dense_{record['id']}"
        config["seed"] = 42
        config["stages"]["mesh"].update({
            "skip_sparse": True, "load_dense_cache": None,
            "save_dense_cache": str((case / "dense_cache.npz").resolve()),
            # The lr162 generator grid is stored in its legacy dense-aligned frame,
            # while the shared FEA domain is physical CAD coordinates.  Keep the
            # density order but map its node coordinates through this inverse
            # affine before each in-loop FEA transfer.
            "fea_node_alignment": str(
                ROOT / "experiments/bracket/chatgpt_image_hi_2026-09-15/"
                       "thickness_dense_sparse_outw20_2026-09-15/"
                       "dense_aligned_domain/alignment.json"),
            "shape_qd_w": 0.0, "shape_anchor_w": 0.0, "shape_scaffold_w": 0.0,
            "shape_residual_w": 0.0, "shape_transport_radius": 0.0,
            "oc_flow_w": 0.0, "sp_shape_qd_w": 0.0, "sp_shape_anchor_w": 0.0,
        })
        # The proven lr162 base config points post-processing at legacy raw CAD exports.
        # They are watertight but contain thousands of extreme sliver triangles, so MeshFix can
        # reinterpret the load-peg seam and remove valid BC material.  The experiment protocol
        # already declares the uniform remeshed BC surfaces; make the executable config match it.
        config["stages"]["post"].update({
            "fix": physical["fixed_bc"],
            "load": physical["load_bc"],
            "peg_dilate_mm": 6.0,
        })
        config_path = case / "config_dense.json"; dump(config_path, config)
        jobs.append({"record": record, "case": case, "gen": gen,
                     "config": config_path, "input": Path(record["model_input"])})
    return jobs


def run_one(job: dict, gpu: int, force: bool) -> dict:
    mesh = job["gen"] / "mesh_dense.obj"
    if mesh.exists() and not force:
        return {"id": job["record"]["id"], "ok": True, "skipped": True, "gpu": gpu}
    command = [str(PYTHON), str(GENERATOR), "--config", str(job["config"]),
               "--target-dir", str(job["input"].parent), "--out", str(job["gen"])]
    with (job["case"] / "dense.log").open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, env=generation_env(gpu),
                                stdout=stream, stderr=subprocess.STDOUT)
    return {"id": job["record"]["id"], "ok": result.returncode == 0 and mesh.exists(),
            "returncode": result.returncode, "gpu": gpu}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent GPU jobs; default is deliberately conservative.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    jobs = prepare(args.experiment, args.round)
    status = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        futures = {pool.submit(run_one, job, i % 8, args.force): job
                   for i, job in enumerate(jobs)}
        for future in concurrent.futures.as_completed(futures):
            row = future.result(); status.append(row); print(row, flush=True)
    evaluation_root = args.experiment / "shared_evaluations" / f"round_{args.round:02d}"
    dump(evaluation_root / "dense_status.json", status)
    results = []
    for job in jobs:
        mesh_path = job["gen"] / "mesh_dense.obj"
        if not mesh_path.exists(): continue
        mesh = load_mesh(mesh_path); parts = mesh.split(only_watertight=False)
        preview = job["case"] / "dense_preview.png"
        render_mesh(mesh_path, preview, job["record"]["id"])
        results.append({**job["record"], "dense_mesh": str(mesh_path.resolve()),
                        "dense_preview": str(preview.resolve()), "dense_components": len(parts),
                        "dense_main_face_fraction": max(len(p.faces) for p in parts) / len(mesh.faces),
                        "dense_volume_mm3": abs(float(mesh.volume)) * 1e9})
    dump(evaluation_root / "dense_results.json", results)
    print((evaluation_root / "dense_results.json").resolve())


if __name__ == "__main__":
    main()
