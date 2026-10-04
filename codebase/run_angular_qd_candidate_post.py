#!/usr/bin/env python3
"""Complete an angular image-QD candidate: exact post Boolean and independent FEA."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import trimesh

from diagnose_bc_preservation import field, interior_points, load_mesh
from run_connectivity_qd_sampling import ROOT, PYTHON
from run_semantic_qd_sparse_round import transform_matrix


def call(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    with log_path.open("w") as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"exit={result.returncode}: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    args = parser.parse_args()
    case = args.case.resolve()
    config = case / "config_full.json"
    cfg = json.loads(config.read_text())
    post = case / "post"
    post.mkdir(exist_ok=True)
    raw_path = case / "generation/mesh.obj"
    aligned_path = post / "mesh_physical_aligned.obj"
    if not aligned_path.exists():
        raw = trimesh.load(raw_path, force="mesh", process=False)
        vertices = np.c_[raw.vertices, np.ones(len(raw.vertices))] @ transform_matrix()
        trimesh.Trimesh(vertices, raw.faces, process=False).export(aligned_path)
    hybrid = post / "mesh_bc_preserved.obj"
    if not hybrid.exists():
        call([str(PYTHON), str(ROOT / "codebase/code/post_hybrid_union_clip.py"),
              "--config", str(config), "--in", str(aligned_path), "--out", str(hybrid)],
             post / "boolean.log")
    final = post / "final.obj"
    if not final.exists():
        call([str(PYTHON), str(ROOT / "codebase/code/surface_remesh_pre.py"),
              "--config", str(config), "--in", str(hybrid), "--out", str(final)],
             post / "remesh.log")

    mesh = load_mesh(final)
    sdf = field(mesh)
    rng = np.random.default_rng(42)
    bc = {}
    for kind in ("fix", "load"):
        reference = load_mesh(Path(cfg["stages"]["post"][kind]))
        points = interior_points(reference, 12000, rng)
        bc[kind] = float((sdf(points) > -0.000375).mean())
    summary = {"case": str(case), "final_mesh": str(final),
               "volume_cm3": abs(float(mesh.volume)) * 1e6,
               "watertight": bool(mesh.is_watertight),
               "components": len(mesh.split(only_watertight=False)),
               "bc_containment": bc,
               "geometry_valid": bool(mesh.is_watertight and
                                      len(mesh.split(only_watertight=False)) == 1 and
                                      min(bc.values()) >= 0.99)}
    if summary["geometry_valid"]:
        fea = post / "fea_independent"
        fea.mkdir(exist_ok=True)
        fea_summary = fea / "fea_tet_summary.json"
        if not fea_summary.exists():
            env = dict(os.environ, FENICS_PY="/home/goya/miniconda3/envs/fenics/bin/python")
            try:
                call([str(PYTHON), str(ROOT / "codebase/code/fea_prep_and_run.py"),
                      "--config", str(config), "--in", str(final), "--out-dir", str(fea)],
                     post / "fea.log", env)
            except RuntimeError as error:
                summary["fea_error"] = str(error)
        if fea_summary.exists():
            data = json.loads(fea_summary.read_text())
            summary["compliance_J"] = data.get("compliance")
            summary["fea_summary"] = str(fea_summary)
    (case / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
