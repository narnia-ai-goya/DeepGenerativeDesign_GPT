"""Measure fixed-frame 3-view mesh novelty for a semantic Pareto-QD archive.

The three silhouette masks are computed from 3D occupancy after fixed and
load interface regions are removed. This checks realized shape independently
of the prompt embedding used to assign semantic archive cells.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/framecorrected_semantic_pareto_qd/archive.json"


def occupancy(path: Path, lo: np.ndarray, hi: np.ndarray, resolution: int) -> np.ndarray:
    mesh = trimesh.load_mesh(path, force="mesh", process=False)
    pitch = float(np.max(hi - lo) / (resolution - 1))
    points = mesh.voxelized(pitch).fill().points
    idx = np.rint((points - lo) * ((resolution - 1) / (hi - lo))).astype(int)
    valid = np.all((idx >= 0) & (idx < resolution), axis=1)
    result = np.zeros((resolution,) * 3, dtype=bool)
    idx = idx[valid]
    result[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return result


def views(occ: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(occ.any(axis=axis) for axis in (2, 1, 0))


def distance(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...]) -> float:
    return float(np.mean([np.logical_xor(x, y).sum() / max(1, np.logical_or(x, y).sum())
                          for x, y in zip(a, b)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--resolution", type=int, default=48)
    args = ap.parse_args()
    archive = args.archive.resolve()
    rows = json.loads(archive.read_text())["records"]
    domain = trimesh.load_mesh(ROOT / "data_real/bracket/original_DesignSpace.stl", process=False)
    lo, hi = domain.bounds
    bc = occupancy(ROOT / "data_real/bracket/fixed.stl", lo, hi, args.resolution)
    bc |= occupancy(ROOT / "data_real/bracket/load.stl", lo, hi, args.resolution)
    maps = []
    for row in rows:
        occ = occupancy(Path(row["final_mesh"]), lo, hi, args.resolution)
        occ[bc] = False
        maps.append(views(occ))
    matrix = np.zeros((len(rows), len(rows)), dtype=float)
    for i in range(len(rows)):
        for j in range(i):
            matrix[i, j] = matrix[j, i] = distance(maps[i], maps[j])
    prior = np.flatnonzero([r["source_round"] < 5 for r in rows])
    new = np.flatnonzero([r["source_round"] >= 5 for r in rows])
    prior_nn = np.array([np.min(matrix[i, prior[prior != i]]) for i in prior])
    threshold = float(np.quantile(prior_nn, 0.1))
    comparisons = []
    for i in new:
        j = int(prior[np.argmin(matrix[i, prior])])
        comparisons.append({"id": rows[i]["id"], "nearest_prior_id": rows[j]["id"],
                            "distance": float(matrix[i, j]),
                            "novelty_gate_threshold": threshold,
                            "passes_novelty_gate": bool(matrix[i, j] >= threshold),
                            "percentile_among_prior_nearest_distances":
                                float(np.mean(prior_nn <= matrix[i, j]) * 100)})
    result = {"method": "3-view silhouette mean Jaccard distance from 48^3 physical occupancy, BC masked",
              "resolution": args.resolution, "prior_count": len(prior), "new_count": len(new),
              "threshold_policy": "10th percentile of prior archive nearest-neighbor distances",
              "threshold": threshold, "prior_nearest_distances": prior_nn.tolist(),
              "comparisons": comparisons}
    path = archive.parent / "shape_novelty.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)
    for row in comparisons:
        print(row)


if __name__ == "__main__":
    main()
