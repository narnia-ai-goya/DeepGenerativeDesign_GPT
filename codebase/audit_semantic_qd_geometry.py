#!/usr/bin/env python3
"""Audit whether prompt-based QD cells correspond to distinct final bracket shapes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/framecorrected_semantic_pareto_qd/archive.json"


def occupancy(path: Path, origin: np.ndarray, pitch: float, shape: tuple[int, ...]) -> np.ndarray:
    mesh = trimesh.load_mesh(path, force="mesh", process=False)
    points = mesh.voxelized(pitch).fill().points
    indices = np.rint((points - origin) / pitch).astype(int)
    valid = np.all((indices >= 0) & (indices < np.asarray(shape)), axis=1)
    mask = np.zeros(shape, dtype=bool)
    indices = indices[valid]
    mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return mask


def views(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    return tuple(mask.any(axis=axis) for axis in (2, 1, 0))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.logical_xor(a, b).sum() / max(1, np.logical_or(a, b).sum()))


def view_distance(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...]) -> float:
    return float(np.mean([jaccard(x, y) for x, y in zip(a, b)]))


def equal_area(mask: np.ndarray, allowed: np.ndarray, count: int) -> np.ndarray:
    """Remove uniform member-width changes while keeping layout and openings."""
    signed = distance_transform_edt(mask) - distance_transform_edt(~mask)
    available = np.flatnonzero(allowed.ravel())
    count = min(count, len(available))
    score = signed.ravel()[available]
    chosen = available[np.argpartition(score, -count)[-count:]]
    result = np.zeros(mask.size, dtype=bool)
    result[chosen] = True
    return result.reshape(mask.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    archive = args.archive.resolve()
    rows = json.loads(archive.read_text())["records"]
    domain = trimesh.load_mesh(ROOT / "data_real/bracket/original_DesignSpace.stl", process=False)
    lo, hi = domain.bounds
    pitch = float(np.max(hi - lo) / (args.resolution - 1))
    shape = tuple((np.ceil((hi - lo) / pitch).astype(int) + 1).tolist())
    bc = occupancy(ROOT / "data_real/bracket/fixed.stl", lo, pitch, shape)
    bc |= occupancy(ROOT / "data_real/bracket/load.stl", lo, pitch, shape)
    maps = []
    for row in rows:
        occ = occupancy(Path(row["final_mesh"]), lo, pitch, shape)
        occ[bc] = False
        maps.append(views(occ))
    bc_views = views(bc)
    target_areas = [int(np.median([v[k].sum() for v in maps])) for k in range(3)]
    normalized = [tuple(equal_area(v[k], ~bc_views[k], target_areas[k]) for k in range(3))
                  for v in maps]
    n = len(rows)
    d = np.zeros((n, n))
    d_normalized = np.zeros((n, n))
    pairs = []
    for i in range(n):
        for j in range(i):
            d[i, j] = d[j, i] = view_distance(maps[i], maps[j])
            d_normalized[i, j] = d_normalized[j, i] = view_distance(normalized[i], normalized[j])
            pairs.append((i, j))
    # Frozen threshold calibrated on nearest-neighbor distances of prior designs.
    prior = np.flatnonzero([r["source_round"] < 5 for r in rows])
    prior_nn = np.array([min(d[i, j] for j in prior if i != j) for i in prior])
    threshold = float(np.quantile(prior_nn, 0.1))
    order = sorted(range(n), key=lambda i: (rows[i]["compliance_J"], rows[i]["volume_mm3"]))
    representatives = []
    assignment = {}
    for i in order:
        nearest = min(representatives, key=lambda j: d[i, j]) if representatives else None
        if nearest is None or d[i, nearest] >= threshold:
            representatives.append(i)
            assignment[rows[i]["id"]] = rows[i]["id"]
        else:
            assignment[rows[i]["id"]] = rows[nearest]["id"]
    within = [d[i, j] for i, j in pairs if rows[i]["cell"] == rows[j]["cell"]]
    between = [d[i, j] for i, j in pairs if rows[i]["cell"] != rows[j]["cell"]]
    cross_duplicates = sorted(
        ({"a": rows[i]["id"], "b": rows[j]["id"], "cell_a": rows[i]["cell"],
          "cell_b": rows[j]["cell"], "distance": float(d[i, j])}
         for i, j in pairs if rows[i]["cell"] != rows[j]["cell"] and d[i, j] < threshold),
        key=lambda x: x["distance"])
    z = np.asarray([r["descriptor_pca2"] for r in rows])
    text_d = [float(np.linalg.norm(z[i] - z[j])) for i, j in pairs]
    shape_d = [float(d[i, j]) for i, j in pairs]
    correlation = float(spearmanr(text_d, shape_d).statistic)
    normalized_within = [d_normalized[i, j] for i, j in pairs if rows[i]["cell"] == rows[j]["cell"]]
    normalized_between = [d_normalized[i, j] for i, j in pairs if rows[i]["cell"] != rows[j]["cell"]]
    normalized_shape_d = [float(d_normalized[i, j]) for i, j in pairs]
    result = {
        "archive": str(archive), "n_records": n, "prompt_cells": len({r["cell"] for r in rows}),
        "physical_grid_pitch_mm": pitch * 1000, "physical_grid_shape": shape,
        "distance": "mean three-view Jaccard after fixed/load BC mask; physical isotropic pitch",
        "threshold": threshold, "shape_clusters": len(representatives),
        "cluster_representatives": [rows[i]["id"] for i in representatives],
        "cluster_assignment": assignment,
        "same_prompt_cell_distance_median": float(np.median(within)),
        "different_prompt_cell_distance_median": float(np.median(between)),
        "text_pca2_vs_shape_distance_spearman": correlation,
        "equal_area_view_distance": {
            "purpose": "diagnostic for member-width-insensitive layout diversity, not the quality objective",
            "target_areas": target_areas,
            "same_prompt_cell_median": float(np.median(normalized_within)),
            "different_prompt_cell_median": float(np.median(normalized_between)),
            "text_pca2_vs_shape_distance_spearman": float(spearmanr(text_d, normalized_shape_d).statistic),
            "distance_matrix": d_normalized.tolist(),
        },
        "cross_cell_near_duplicates": cross_duplicates,
        "ids": [r["id"] for r in rows], "cells": [r["cell"] for r in rows],
        "distance_matrix": d.tolist(),
    }
    out = (args.out or archive.parent / "geometry_audit.json").resolve()
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out)
    print(f"prompt cells={result['prompt_cells']}, shape clusters={len(representatives)}, "
          f"cross-cell near duplicates={len(cross_duplicates)}, text-shape Spearman={correlation:.3f}")


if __name__ == "__main__":
    main()
