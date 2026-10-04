"""Independent 15 mm FEA of the final Dense-core-guided chair OBJ."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from pysdf import SDF
from scipy.spatial.transform import Rotation
import trimesh

from make_chair_domain import ROOT
from run_chair_existing_fea_on_015_2026_10_04 import OUT, SPEC


BASE = OUT / "diagnostics/dense_core_pilot"
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fea", choices=("on", "off"), default="on")
    parser.add_argument("--weight", type=int, default=None,
                        help="evaluate a fixed-6mm, FEA-off weight-search result instead")
    parser.add_argument("--weight-case", default=None,
                        help="evaluate a named weight-search case such as w060_fea_on")
    parser.add_argument("--baseline-off", action="store_true",
                        help="evaluate same-Dense Sparse FEA-OFF baseline")
    parser.add_argument("--source-obj", type=Path, default=None,
                        help="evaluate an arbitrary generated OBJ")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="independent FEA output directory for --source-obj")
    parser.add_argument("--source-aligned", action="store_true",
                        help="--source-obj is already in physical FEM coordinates")
    args = parser.parse_args()
    if args.source_obj is not None:
        if args.output_dir is None:
            parser.error("--source-obj requires --output-dir")
        source = args.source_obj.resolve()
        case = args.output_dir.resolve()
    elif args.baseline_off:
        source = OUT / "diagnostics/same_dense_sparse_fea_off/generation/mesh.obj"
        case = BASE / "weight_search/w000/independent_fea_15mm"
    elif args.weight_case is not None:
        source = BASE / f"weight_search/{args.weight_case}/generation/mesh.obj"
        case = BASE / f"weight_search/{args.weight_case}/independent_fea_15mm"
    elif args.weight is None:
        source = BASE / f"core_6mm_w30_fea_{args.fea}/generation/mesh.obj"
        case = BASE / f"core_6mm_w30_fea_{args.fea}/independent_fea_15mm"
    else:
        source = BASE / f"weight_search/w{args.weight:03d}/generation/mesh.obj"
        case = BASE / f"weight_search/w{args.weight:03d}/independent_fea_15mm"
    case.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(source, force="mesh", process=False)
    if not mesh.is_watertight:
        raise ValueError("final OBJ is not watertight")
    spec = json.loads((OUT.parents[4] / "single_view_spec_2026-10-03/specification.json").read_text())
    reg = spec["native_frame_registration"]
    rotation = Rotation.from_euler("x", reg["rotation_x_degrees"], degrees=True).as_matrix()
    if not args.source_aligned:
        mesh.vertices = ((mesh.vertices - np.asarray(reg["source_center_m"])) @ rotation.T
                         * reg["uniform_scale"] + np.asarray(reg["physical_center_m"]))
    mesh.export(case / "aligned_full.obj")
    geometry = np.load(OUT.parent / "diagnostics/msh_simp_topopt_015_2026-10-04/fenics_geometry.npz")
    centroids, volumes = geometry["centroids"], geometry["volumes"]
    if len(centroids) != 388856:
        raise ValueError("unexpected FEM mesh")
    sdf = SDF(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    inside = np.concatenate([sdf(centroids[i:i+25000].astype(np.float32)) > 0
                             for i in range(0, len(centroids), 25000)])
    rho = np.where(inside, 1., .001)
    np.save(case / "rho_cells.npy", rho)
    summary = {"source_obj": str(source), "watertight": True,
               "inside_cells": int(inside.sum()),
               "solid_volume_liters": float(volumes[inside].sum() * 1000),
               "density_method": "raw complete OBJ at 15 mm FEM centroids; void E_min=0.001",
               "fea_mesh": str(SPEC / "fea_domain/chair_015.msh")}
    (case / "geometry_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    cmd = ["/home/goya/miniconda3/envs/fenics/bin/python",
           str(ROOT / "codebase/code/fenics_fea_bracket.py"),
           "--domain-dir", str(SPEC / "fea_domain"),
           "--density", str(case / "rho_cells.npy"),
           "--output", str(case / "dc.npy"),
           "--mesh-cache", str(SPEC / "fea_domain/chair_015.msh"),
           "--mesh-size", ".015", "--penal", "2", "--E0", "1",
           "--load-magnitude", "800",
           "--second-load-stl", str(SPEC / "fea_domain_back/load.stl"),
           "--second-load-magnitude", "200", "--second-load-mode", "y"]
    env = {**os.environ, "LOAD_MODE": "-z", "BC_SURFACE_DIST": "1",
           "BC_DIST": ".025", "FEA_MAX_NODE_MAP_DISTANCE": ".04",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "FEA_KSP_TYPE": "gmres", "FEA_PC_TYPE": "gamg",
           "FEA_KSP_RTOL": "1e-7", "FEA_KSP_MAX_IT": "1500"}
    with (case / "fea.log").open("w") as stream:
        process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"FEA failed: {case / 'fea.log'}")
    info = json.loads((case / "dc_info.json").read_text())
    summary["compliance"] = info["compliance"]
    (case / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
