#!/usr/bin/env python3
"""Replay surface_remesh_pre one operation at a time and trace BC containment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pymeshlab
import pymeshfix
import trimesh
from pysdf import SDF


def tm(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=True)
    mesh.merge_vertices()
    return mesh


def sdf(mesh: trimesh.Trimesh) -> SDF:
    return SDF(np.ascontiguousarray(mesh.vertices, dtype=np.float32),
               np.ascontiguousarray(mesh.faces, dtype=np.uint32))


def interior_points(mesh: trimesh.Trimesh, count: int = 10000) -> np.ndarray:
    rng = np.random.default_rng(42)
    fn = sdf(mesh)
    chunks, left = [], count
    while left:
        q = rng.uniform(mesh.bounds[0], mesh.bounds[1], size=(max(10000, 4 * left), 3)).astype(np.float32)
        q = q[fn(q) > 0][:left]
        chunks.append(q); left -= len(q)
    return np.concatenate(chunks)


def measure(path: Path, points: np.ndarray) -> dict:
    mesh = tm(path)
    values = sdf(mesh)(np.ascontiguousarray(points, dtype=np.float32))
    return {
        "path": str(path.resolve()), "vertices": len(mesh.vertices), "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "components": len(mesh.split(only_watertight=False)),
        "volume_mm3": float(abs(mesh.volume) * 1e9),
        "load_containment_tol_0p375mm": float((values > -0.000375).mean()),
        "load_strict_inside": float((values > 0).mean()),
        "load_signed_distance_p05_mm": float(np.percentile(values, 5) * 1000),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", type=Path)
    args = ap.parse_args()
    case = args.case.resolve()
    config = json.loads((case / "config_sparse.json").read_text())
    edge_mm = float(config["stages"]["post"]["edge_mm"])
    load = tm(Path(config["stages"]["post"]["load"]))
    points = interior_points(load)
    source = case / "gen/mesh_bc_preserved.obj"
    out = case / "bc_remesh_trace"
    out.mkdir(exist_ok=True)
    rows = []

    def record(label: str, path: Path) -> None:
        row = {"stage": label, **measure(path, points)}
        rows.append(row)
        print(label, f"contain={row['load_containment_tol_0p375mm']:.3f}",
              f"strict={row['load_strict_inside']:.3f}", f"wt={row['watertight']}",
              f"comp={row['components']}", f"vol={row['volume_mm3']:.0f}", flush=True)

    record("00_boolean_input", source)
    ms = pymeshlab.MeshSet(); ms.load_new_mesh(str(source))

    def save(label: str, suffix: str = ".ply") -> Path:
        path = out / f"{label}{suffix}"
        ms.save_current_mesh(str(path))
        record(label, path)
        return path

    def attempt(name: str, **kwargs) -> None:
        try:
            getattr(ms, name)(**kwargs)
        except Exception as exc:
            print(name, "FAILED", type(exc).__name__, exc, flush=True)

    ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(0.001))
    ms.meshing_remove_duplicate_faces()
    attempt("meshing_remove_duplicate_vertices")
    attempt("meshing_remove_null_faces")
    attempt("meshing_remove_folded_faces")
    attempt("meshing_remove_t_vertices")
    attempt("meshing_remove_unreferenced_vertices")
    attempt("meshing_repair_non_manifold_edges")
    attempt("meshing_repair_non_manifold_vertices")
    save("01_cleanup")

    attempt("meshing_remove_connected_component_by_diameter",
            mincomponentdiag=pymeshlab.PercentageValue(5.0))
    save("02_remove_small_components")

    ms.meshing_isotropic_explicit_remeshing(
        iterations=10, targetlen=pymeshlab.PureValue(edge_mm / 1000.0), adaptive=False)
    save("03_isotropic_remesh")

    attempt("meshing_remove_null_faces")
    attempt("meshing_remove_folded_faces")
    attempt("meshing_repair_non_manifold_edges")
    attempt("meshing_repair_non_manifold_vertices")
    save("04_post_cleanup")

    attempt("meshing_close_holes", maxholesize=2000)
    # Match surface_remesh_pre.py exactly at the pymeshlab -> trimesh boundary.
    tmp_path = save("05_close_holes", ".stl")

    mesh = tm(tmp_path)
    comps = mesh.split(only_watertight=False)
    if len(comps) > 1:
        mesh = max(comps, key=lambda c: len(c.faces)); mesh.remove_unreferenced_vertices()
    largest_path = out / "06_trimesh_largest.ply"; mesh.export(largest_path)
    record("06_trimesh_largest", largest_path)

    repair = pymeshfix.MeshFix(mesh.vertices.astype(np.float64), mesh.faces.astype(np.int32))
    repair.repair()
    repaired = trimesh.Trimesh(np.asarray(repair.points), np.asarray(repair.faces), process=True)
    repair_path = out / "07_meshfix.ply"; repaired.export(repair_path)
    record("07_meshfix", repair_path)

    comps = repaired.split(only_watertight=False)
    if len(comps) > 1:
        repaired = max(comps, key=lambda c: len(c.faces)); repaired.remove_unreferenced_vertices()
    final_path = out / "08_meshfix_largest.obj"; repaired.export(final_path)
    record("08_meshfix_largest", final_path)

    result = out / "trace.json"
    result.write_text(json.dumps(rows, indent=2) + "\n")
    print(result.resolve())


if __name__ == "__main__":
    main()
