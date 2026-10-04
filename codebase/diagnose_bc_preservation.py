#!/usr/bin/env python3
"""Measure BC volume containment and legacy surface proximity across pipeline stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from pysdf import SDF


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=True)
    mesh.merge_vertices()
    return mesh


def field(mesh: trimesh.Trimesh) -> SDF:
    return SDF(np.ascontiguousarray(mesh.vertices, dtype=np.float32),
               np.ascontiguousarray(mesh.faces, dtype=np.uint32))


def interior_points(mesh: trimesh.Trimesh, count: int, rng: np.random.Generator) -> np.ndarray:
    sdf = field(mesh)
    chunks = []
    need = count
    while need > 0:
        q = rng.uniform(mesh.bounds[0], mesh.bounds[1], size=(max(need * 4, 10000), 3)).astype(np.float32)
        inside = q[sdf(q) > 0]
        take = inside[:need]
        chunks.append(take)
        need -= len(take)
    return np.concatenate(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()
    case = args.case.resolve()
    config = json.loads((case / "config_sparse.json").read_text())
    mesh_cfg = config["stages"]["mesh"]
    post_cfg = config["stages"]["post"]
    root = Path("/home/goya/SDL/3d_qd")
    bc_sets = {
        "generation": {"fix": Path(mesh_cfg["fix_stl"]), "load": Path(mesh_cfg["load_stl"])},
        "post": {"fix": Path(post_cfg["fix"]), "load": Path(post_cfg["load"])},
        "evaluation": {"fix": root / "data_real/bracket/fixed.stl",
                       "load": root / "data_real/bracket/load.stl"},
    }
    stages = {
        "sparse_raw": case / "gen/mesh.obj",
        "sparse_physical_aligned": case / "gen/mesh_physical_aligned.obj",
        "post_union_clip": case / "gen/mesh_bc_preserved.obj",
        "delivered_final": case / "gen/final.obj",
    }
    rng = np.random.default_rng(42)
    results = {"case": str(case), "bc_sets": {}, "stages": {}}
    bc_meshes = {}
    bc_points = {}
    for set_name, paths in bc_sets.items():
        results["bc_sets"][set_name] = {}
        bc_meshes[set_name] = {}
        bc_points[set_name] = {}
        for kind, path in paths.items():
            mesh = load_mesh(path)
            pts = interior_points(mesh, args.samples, rng)
            bc_meshes[set_name][kind] = mesh
            bc_points[set_name][kind] = pts
            results["bc_sets"][set_name][kind] = {
                "path": str(path.resolve()),
                "bounds_mm": (mesh.bounds * 1000).tolist(),
                "volume_mm3": float(abs(mesh.volume) * 1e9),
            }
    for stage_name, path in stages.items():
        mesh = load_mesh(path)
        sdf = field(mesh)
        stage_result = {
            "path": str(path.resolve()), "watertight": bool(mesh.is_watertight),
            "components": len(mesh.split(only_watertight=False)),
            "volume_mm3": float(abs(mesh.volume) * 1e9), "against": {},
        }
        for set_name in bc_sets:
            stage_result["against"][set_name] = {}
            for kind in ("fix", "load"):
                values = sdf(np.ascontiguousarray(bc_points[set_name][kind], dtype=np.float32))
                stage_result["against"][set_name][kind] = {
                    "interior_volume_containment_fraction": float((values > -0.000375).mean()),
                    "strict_interior_fraction": float((values > 0).mean()),
                    "p05_signed_distance_mm": float(np.percentile(values, 5) * 1000),
                }
        results["stages"][stage_name] = stage_result
    output = case / "bc_preservation_diagnosis.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(output)
    for stage, row in results["stages"].items():
        print(stage, "watertight", row["watertight"], "components", row["components"])
        for set_name, vals in row["against"].items():
            print(" ", set_name, "fix/load containment",
                  f"{vals['fix']['interior_volume_containment_fraction']:.3f}/"
                  f"{vals['load']['interior_volume_containment_fraction']:.3f}")


if __name__ == "__main__":
    main()
