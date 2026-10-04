#!/usr/bin/env python3
"""Locally fair the generated shoulder around fixed bracket pegs.

The fixture mesh and nearby delivered vertices stay fixed.  Only the narrow
transition to the generated body is smoothed; connectivity is unchanged.
Always validate the output against the design domain and BC before use.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def repair(mesh: trimesh.Trimesh, fix: trimesh.Trimesh, *, iterations: int,
           strength: float, inner_mm: float, outer_mm: float) -> tuple[trimesh.Trimesh, dict]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    distance = cKDTree(fix.vertices).query(vertices, workers=-1)[0] * 1000.0
    band = max((outer_mm - inner_mm) * 0.3, 0.1)
    weight = smoothstep((distance - inner_mm) / band) * smoothstep((outer_mm - distance) / band)
    pairs = np.asarray(mesh.edges_unique, dtype=np.int64)
    i = np.r_[pairs[:, 0], pairs[:, 1]]
    j = np.r_[pairs[:, 1], pairs[:, 0]]
    adjacency = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(vertices), len(vertices))).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    for _ in range(iterations):
        mean = adjacency @ vertices / np.maximum(degree[:, None], 1)
        vertices += strength * weight[:, None] * (mean - vertices)
    output = trimesh.Trimesh(vertices=vertices, faces=mesh.faces.copy(), process=False)
    report = {"iterations": iterations, "strength": strength,
              "inner_mm": inner_mm, "outer_mm": outer_mm,
              "moved_vertices": int(np.count_nonzero(weight > 0)),
              "max_displacement_mm": float(np.max(np.linalg.norm(vertices - mesh.vertices, axis=1)) * 1000),
              "watertight": bool(output.is_watertight),
              "volume_mm3": float(abs(output.volume) * 1e9)}
    return output, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--fix", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--strength", type=float, default=0.3)
    ap.add_argument("--inner-mm", type=float, default=0.8)
    ap.add_argument("--outer-mm", type=float, default=7.0)
    args = ap.parse_args()
    mesh = trimesh.load_mesh(args.inp, force="mesh", process=False)
    fix = trimesh.load_mesh(args.fix, force="mesh", process=True)
    fix.merge_vertices()
    result, report = repair(mesh, fix, iterations=args.iterations,
                            strength=args.strength, inner_mm=args.inner_mm,
                            outer_mm=args.outer_mm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.export(args.out)
    report["input"] = str(args.inp.resolve())
    report["output"] = str(args.out.resolve())
    (args.out.parent / f"{args.out.stem}_repair.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
