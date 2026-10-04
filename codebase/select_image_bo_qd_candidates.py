#!/usr/bin/env python3
"""Select the next image-space QD batch with a lightweight black-box surrogate.

The image model and the 3D generator remain frozen.  A Gaussian process is fitted only to
observed descriptor/realization outcomes and ranks a discrete library of structured prompts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

from designer_steering import load_state, select_batch, steer_candidates


ROOT = Path("/home/goya/SDL/3d_qd")
SOURCE = ROOT / "experiments/bracket/image_qd_2026-09-22"
OUT = ROOT / "experiments/bracket/image_bo_qd_iteration_01_2026-09-22"


CANDIDATES = [
    # name, void target, branch target, minimum strut thickness at native 162 px, prompt genome
    ("reinforced_double_arch", .372, 20, 8.5,
     "two nested load-bearing arches with a thick cross-tie and two redundant diagonal braces"),
    ("radial_five_spoke", .348, 25, 8.0,
     "five thick radial spokes joining a broad central hub to every mounting region"),
    ("thick_ladder", .315, 16, 9.5,
     "two heavy longitudinal rails joined by three wide transverse rungs and short diagonal gussets"),
    ("redundant_twin_fan", .288, 25, 8.5,
     "paired fan-shaped webs with redundant branches and no single narrow bridge"),
    ("boxed_k_truss", .338, 22, 8.5,
     "a boxed K-truss with doubled diagonals around the load lugs"),
    ("diamond_ring_web", .362, 23, 8.5,
     "a thick diamond ring web crossed by two continuous load paths"),
    ("triple_spine", .300, 22, 9.0,
     "three parallel curved spines connected by broad transverse bridges"),
    ("reinforced_x_arch", .355, 21, 9.0,
     "a broad X brace nested inside a shallow arch, with doubled joints"),
    ("offset_double_y", .326, 22, 8.5,
     "two offset Y-shaped load paths sharing wide junctions"),
    ("perimeter_spoke", .365, 25, 8.0,
     "a reinforced perimeter ring with six thick inward spokes"),
    ("low_void_crossgrid", .292, 19, 10.0,
     "a coarse cross-grid of very thick rails with large rounded openings"),
    ("buttressed_arch", .378, 18, 10.0,
     "one broad arch supported by four thick buttresses and a continuous lower tie"),
]


def cell(void: float, branches: float) -> tuple[int, int]:
    return (int(np.digitize(void, [.30, .33, .36])),
            int(np.digitize(branches, [18, 21, 24])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--steering", type=Path, default=None,
                        help="optional designer-steering JSON; changes soft ranking only")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    archive = json.loads((args.source / "image_archive.json").read_text())["records"]
    verified = {r["archetype"]: r for r in json.loads((args.source / "image_qd_verified.json").read_text())}
    observations = []
    for row in archive:
        if row["archetype"] not in verified:
            continue
        result = verified[row["archetype"]]
        valid = bool(result["final_watertight"] and result["final_components"] == 1
                     and result["fix_coverage_1p5mm"] >= .70
                     and result["load_coverage_1p5mm"] >= .60)
        # Realization quality is intentionally geometry-only in this iteration.  Independent
        # compliance becomes an additional objective after this inexpensive gate is stable.
        quality = min(result["fix_coverage_1p5mm"], result["load_coverage_1p5mm"])
        observations.append({
            "name": row["archetype"], "void": row["void_fraction"],
            "branches": row["load_path_branch_clusters"],
            "thickness": row["skeleton_thickness_p10_px162"],
            "quality": quality, "valid": valid, "cell": row["cell"],
        })

    X = np.array([[r["void"], r["branches"], r["thickness"]] for r in observations])
    y = np.array([r["quality"] for r in observations])
    labels = np.array([r["valid"] for r in observations], dtype=int)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    reg = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(length_scale=np.ones(3), nu=1.5)
               + WhiteKernel(noise_level=.01),
        normalize_y=True, random_state=42, n_restarts_optimizer=8).fit(Xs, y)
    clf = GaussianProcessClassifier(
        kernel=ConstantKernel(1.0) * Matern(length_scale=np.ones(3), nu=1.5),
        random_state=42, n_restarts_optimizer=8).fit(Xs, labels)

    occupied = {tuple(r["cell"]) for r in observations if r["valid"]}
    incumbents = {}
    for r in observations:
        if r["valid"]:
            c = tuple(r["cell"]); incumbents[c] = max(incumbents.get(c, 0), r["quality"])

    pool = []
    for name, void, branches, thickness, genome in CANDIDATES:
        x = scaler.transform([[void, branches, thickness]])
        mean, std = reg.predict(x, return_std=True); mean, std = float(mean[0]), float(std[0])
        p_valid = float(clf.predict_proba(x)[0, 1])
        c = cell(void, branches)
        incumbent = incumbents.get(c, .55)
        z = (mean - incumbent) / max(std, 1e-9)
        ei = (mean - incumbent) * norm.cdf(z) + std * norm.pdf(z)
        empty_bonus = .16 if c not in occupied else 0.0
        # Uncertainty is useful with only six expensive observations, while validity prevents
        # repeatedly selecting the fragile high-void/low-branch corner represented by arch_tie.
        acquisition = p_valid * (max(0.0, ei) + .30 * std + empty_bonus)
        pool.append({"id": name, "name": name, "target_void_fraction": void,
                     "target_branch_clusters": branches,
                     "target_thickness_px162": thickness, "cell": list(c),
                     "genome": genome, "predicted_quality": mean,
                     "predictive_std": std, "p_valid": p_valid,
                     "expected_improvement": float(ei), "acquisition": acquisition,
                     "base_acquisition": acquisition})

    steering = load_state(args.steering) if args.steering else None
    ranked_pool = steer_candidates(pool, steering) if steering else pool
    selected = select_batch(ranked_pool, args.batch_size, one_per_cell=True)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "gradient-free batch BOP-Elites over a structured prompt library",
        "designer_steering": str(args.steering.resolve()) if args.steering else None,
        "steering_revision": steering["revision"] if steering else None,
        "archive_adapter": "legacy image morphology cells; semantic-mass archive is the target framework",
        "foundation_model_training": False,
        "features": ["void_fraction", "branch_clusters", "minimum_strut_thickness_px162"],
        "quality": "minimum of fixed/load surface coverage at 1.5 mm",
        "validity_gate": "watertight, one component, fixed coverage >= 0.70, load coverage >= 0.60",
        "observations": observations, "candidate_pool": ranked_pool, "selected": selected,
    }
    path = args.output / "bo_selection.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(path)
    for row in selected:
        score = row.get("steered_acquisition", row["acquisition"])
        print(row["name"], row["cell"], f"a={score:.4f}",
              f"p_valid={row['p_valid']:.3f}", row["genome"])


if __name__ == "__main__":
    main()
