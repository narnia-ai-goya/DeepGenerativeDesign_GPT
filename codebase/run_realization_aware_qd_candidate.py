#!/usr/bin/env python3
"""Evaluate one proposed image through dense, sparse, post, independent FEA, archive."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import trimesh

from run_connectivity_qd_sampling import generation_env, render_mesh
from run_semantic_qd_sparse_round import transform_matrix

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22"
BASE_CONFIG = STUDY / "shared_evaluations/round_05/r5_longitudinal_spine__medium/fea_on_bc6_framefix/config.json"
SELECTION = STUDY / "realization_aware_qd_archive_v2/next_image_batch/selection.json"
ADDITIONS = STUDY / "realization_aware_qd_archive_v2/verified_additions.json"
PY_D3D = Path("/home/goya/miniconda3/envs/direct3ds2/bin/python")
PY_FEN = Path("/home/goya/miniconda3/envs/fenics/bin/python")
CODE = ROOT / "codebase/code"


def run(cmd: list[str], log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f"exit={rc}: {log}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate_id")
    ap.add_argument("--selection", type=Path, default=SELECTION)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--round-name", default="loop_01")
    args = ap.parse_args()
    options = json.loads(args.selection.read_text())["ranked"]
    row = next((r for r in options if r["id"] == args.candidate_id), None)
    if row is None:
        raise ValueError(f"Candidate not in selection: {args.candidate_id}")
    wd = STUDY / "realization_aware_qd_archive_v2" / args.round_name / args.candidate_id
    wd.mkdir(parents=True, exist_ok=True)
    base = json.loads(BASE_CONFIG.read_text())
    input_dir = str(Path(row["model_input"]).parent)
    env = generation_env(args.gpu)
    env["FEA_MAX_NODE_MAP_DISTANCE"] = "0.02"

    dense_cfg = json.loads(json.dumps(base))
    dense_cfg["name"] = f"qdv2_{args.candidate_id}_dense"
    dense_cfg["stages"]["mesh"].update(skip_sparse=True, load_dense_cache=None,
                                         save_dense_cache=str(wd / "dense_cache.npz"))
    dense_path = wd / "config_dense.json"
    dense_path.write_text(json.dumps(dense_cfg, indent=2, ensure_ascii=False) + "\n")
    dense = wd / "dense"
    if not (wd / "dense_cache.npz").exists():
        run([str(PY_D3D), str(CODE / "generate_with_physics_guidance.py"),
             "--config", str(dense_path), "--target-dir", input_dir, "--out", str(dense)],
            wd / "dense.log", env)
    print("dense complete", flush=True)

    sparse_cfg = json.loads(json.dumps(base))
    sparse_cfg["name"] = f"qdv2_{args.candidate_id}_sparse"
    sparse_cfg["stages"]["mesh"].update(skip_sparse=False,
                                          load_dense_cache=str(wd / "dense_cache.npz"),
                                          save_dense_cache=None)
    sparse_path = wd / "config_sparse.json"
    sparse_path.write_text(json.dumps(sparse_cfg, indent=2, ensure_ascii=False) + "\n")
    sparse = wd / "sparse"
    if not (sparse / "mesh.obj").exists():
        run([str(PY_D3D), str(CODE / "generate_with_physics_guidance.py"),
             "--config", str(sparse_path), "--target-dir", input_dir, "--out", str(sparse)],
            wd / "sparse.log", env)
    print("sparse complete", flush=True)

    post = wd / "post"
    post.mkdir(exist_ok=True)
    aligned = post / "mesh_physical_aligned.obj"
    if not aligned.exists():
        raw = trimesh.load_mesh(sparse / "mesh.obj", force="mesh", process=False)
        transformed = np.c_[raw.vertices, np.ones(len(raw.vertices))] @ transform_matrix()
        trimesh.Trimesh(transformed, raw.faces, process=False).export(aligned)
    hybrid = post / "mesh_bc_preserved.obj"
    if not hybrid.exists():
        run([str(PY_D3D), str(CODE / "post_hybrid_union_clip.py"), "--config", str(sparse_path),
             "--in", str(aligned), "--out", str(hybrid)], post / "post.log", env)
    final = post / "final.obj"
    if not final.exists():
        run([str(PY_D3D), str(CODE / "surface_remesh_pre.py"), "--config", str(sparse_path),
             "--in", str(hybrid), "--out", str(final)], post / "remesh.log", env)
    print("final mesh complete", flush=True)

    feadir = post / "fea_independent"
    summary = feadir / "fea_tet_summary.json"
    if not summary.exists():
        fea_env = dict(env, FENICS_PY=str(PY_FEN))
        run([str(PY_D3D), str(CODE / "fea_prep_and_run.py"), "--config", str(sparse_path),
             "--in", str(final), "--out-dir", str(feadir)], post / "fea.log", fea_env)
    mesh = trimesh.load_mesh(final, force="mesh", process=False)
    fea = json.loads(summary.read_text())
    if not mesh.is_watertight or not 0 < fea["compliance"] < 1:
        raise RuntimeError("Final mesh or FEA failed validation; candidate not archived")
    render = wd / "final_iso.png"
    if not render.exists():
        render_mesh(final, render, args.candidate_id)
    print("independent FEA complete", flush=True)

    manifest = json.loads(ADDITIONS.read_text()) if ADDITIONS.exists() else {"records": []}
    new_id = f"{args.candidate_id}__qdv2_fea_on"
    addition = {"id": new_id, "semantic_niche_target": row["semantic_niche_target"],
                "source_round": 6, "final_mesh": str(final), "fea_summary": str(summary),
                "comparison_render": str(render), "input_image": row["image_512"]}
    manifest["records"] = [r for r in manifest["records"] if r["id"] != new_id] + [addition]
    ADDITIONS.write_text(json.dumps(manifest, indent=2) + "\n")
    run([str(PY_D3D), str(ROOT / "codebase/build_realization_aware_qd_archive.py")],
        wd / "archive_update.log", env)
    run([str(PY_D3D), str(ROOT / "codebase/validate_realization_aware_qd_geometry.py"),
         "--workers", "4"], wd / "geometry_gate_update.log", env)
    run([str(PY_D3D), str(ROOT / "codebase/build_realization_aware_qd_archive.py")],
        wd / "archive_validated_update.log", env)
    run([str(PY_D3D), str(ROOT / "codebase/select_realization_aware_qd_images.py")],
        wd / "next_selection.log", env)
    updated = json.loads((STUDY / "realization_aware_qd_archive_v2/archive.json").read_text())
    updated_row = next(r for r in updated["records"] if r["id"] == new_id)
    print(json.dumps({"id": new_id, "final_mesh": str(final),
                      "compliance_J": fea["compliance"], "volume_mm3": abs(mesh.volume) * 1e9,
                      "geometry_gate_pass": updated_row["geometry_gate_pass"],
                      "pareto_elite": updated_row["pareto_elite"],
                      "archive": str(STUDY / "realization_aware_qd_archive_v2/index.html")}, indent=2))


if __name__ == "__main__":
    main()
