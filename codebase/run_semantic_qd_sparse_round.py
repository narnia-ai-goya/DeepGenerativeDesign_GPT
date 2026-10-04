#!/usr/bin/env python3
"""Run sparse generation and post-processing for promoted Semantic BO-QD candidates."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess

import numpy as np
import trimesh

from run_connectivity_qd_sampling import (BASE, GENERATOR, POST, PYTHON, REMESH, ROOT,
                                           dump, generation_env, load_mesh, render_mesh)


def transform_matrix() -> np.ndarray:
    source = load_mesh(BASE / "generation/mesh.obj")
    target = load_mesh(BASE / "generation/mesh_physical_aligned.obj")
    indices = np.arange(0, len(source.vertices), 64)
    return np.linalg.lstsq(np.c_[source.vertices[indices], np.ones(len(indices))],
                           target.vertices[indices], rcond=None)[0]


def sparse_one(row: dict, experiment: Path, round_index: int, gpu: int) -> dict:
    case = experiment / "shared_evaluations" / f"round_{round_index:02d}" / row["id"]
    gen = case / "gen"; marker = case / "sparse_complete.json"
    if marker.exists() and (gen / "mesh.obj").exists():
        return {"id": row["id"], "ok": True, "skipped": True, "gpu": gpu}
    config = json.loads((case / "config_dense.json").read_text())
    config["name"] = f"semantic_qd_sparse_{row['id']}"
    config["stages"]["mesh"].update({
        "skip_sparse": False,
        "load_dense_cache": str((case / "dense_cache.npz").resolve()),
        "save_dense_cache": None,
    })
    config_path = case / "config_sparse.json"; dump(config_path, config)
    command = [str(PYTHON), str(GENERATOR), "--config", str(config_path),
               "--target-dir", str(Path(row["model_input"]).parent), "--out", str(gen)]
    with (case / "sparse.log").open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, env=generation_env(gpu),
                                stdout=stream, stderr=subprocess.STDOUT)
    ok = result.returncode == 0 and (gen / "mesh.obj").exists()
    if ok: dump(marker, {"mesh": str((gen / "mesh.obj").resolve())})
    return {"id": row["id"], "ok": ok, "returncode": result.returncode, "gpu": gpu}


def post_one(row: dict, experiment: Path, round_index: int, transform: np.ndarray) -> dict:
    case = experiment / "shared_evaluations" / f"round_{round_index:02d}" / row["id"]
    gen = case / "gen"; config = case / "config_sparse.json"
    raw = load_mesh(gen / "mesh.obj")
    aligned = trimesh.Trimesh(np.c_[raw.vertices, np.ones(len(raw.vertices))] @ transform,
                              raw.faces.copy(), process=False)
    aligned_path = gen / "mesh_physical_aligned.obj"; aligned.export(aligned_path)
    hybrid, final = gen / "mesh_bc_preserved.obj", gen / "final.obj"
    with (case / "post.log").open("w") as stream:
        rc1 = subprocess.run([str(PYTHON), str(POST), "--config", str(config), "--in",
                              str(aligned_path), "--out", str(hybrid)], cwd=ROOT,
                             stdout=stream, stderr=subprocess.STDOUT).returncode
    if rc1: return {"id": row["id"], "ok": False, "stage": "post"}
    with (case / "remesh.log").open("w") as stream:
        rc2 = subprocess.run([str(PYTHON), str(REMESH), "--config", str(config), "--in",
                              str(hybrid), "--out", str(final)], cwd=ROOT,
                             stdout=stream, stderr=subprocess.STDOUT).returncode
    if rc2 or not final.exists(): return {"id": row["id"], "ok": False, "stage": "remesh"}
    mesh = load_mesh(final); preview = case / "final_preview.png"; render_mesh(final, preview, row["id"])
    return {**row, "ok": True, "final_mesh": str(final.resolve()),
            "final_preview": str(preview.resolve()), "final_watertight": bool(mesh.is_watertight),
            "final_components": len(mesh.split(only_watertight=False)),
            "final_volume_mm3": abs(float(mesh.volume)) * 1e9}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path); parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent GPU jobs; default is deliberately conservative.")
    args = parser.parse_args()
    evaluation = args.experiment / "shared_evaluations" / f"round_{args.round:02d}"
    rows = json.loads((evaluation / "unique_sparse_fea_jobs.json").read_text())["records"]
    sparse_status = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, len(rows))) as pool:
        futures = {pool.submit(sparse_one, row, args.experiment, args.round, i % 8): row
                   for i, row in enumerate(rows)}
        for future in concurrent.futures.as_completed(futures):
            result = future.result(); sparse_status.append(result); print(result, flush=True)
    dump(evaluation / "sparse_status.json", sparse_status)
    successful = {r["id"] for r in sparse_status if r["ok"]}
    transform = transform_matrix(); final = []
    for row in rows:
        if row["id"] not in successful: continue
        result = post_one(row, args.experiment, args.round, transform)
        print({k: result[k] for k in ("id", "ok")}, flush=True)
        if result["ok"]: final.append(result)
    dump(evaluation / "final_results.json", final)
    print((evaluation / "final_results.json").resolve())


if __name__ == "__main__": main()
