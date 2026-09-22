"""Audit behavior-axis choices on completed verified bracket realizations.

This is a retrospective descriptor-development analysis. It must not be used
as an unbiased comparison between acquisition methods because the observations
were collected using a different, previously fixed descriptor space.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
from scipy.stats import spearmanr


FEATURES = [
    "projection_openness", "macro_void_fraction_3mm", "normalized_void_scale",
    "material_centroid_x", "material_centroid_y", "material_centroid_z",
    "material_spread_x", "material_spread_y", "material_spread_z",
    "material_anisotropy", "spatial_entropy", "surface_compactness",
    "stress_concentration_proxy",
]


def abs_spearman(a, b):
    value = spearmanr(a, b).statistic
    return abs(float(value)) if np.isfinite(value) else 1.0


def weighted_gini(values, weights):
    order = np.argsort(values)
    x = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    cumulative_w = np.cumsum(w)
    cumulative_wx = np.cumsum(w * x)
    if cumulative_wx[-1] <= 0:
        return 0.0
    area = np.sum((cumulative_wx[1:] + cumulative_wx[:-1]) * np.diff(cumulative_w))
    return float(1.0 - area / (cumulative_w[-1] * cumulative_wx[-1]))


def stress_proxy(case_dir):
    """Volume-weighted Gini of 99th-percentile-winsorized von Mises stress."""
    mesh = meshio.read(Path(case_dir) / "gen/fea/tet_vm.vtu")
    tets = mesh.cells_dict["tetra"]
    points = mesh.points
    stress = mesh.cell_data_dict["von_mises_Pa"]["tetra"]
    a, b, c, d = [points[tets[:, i]] for i in range(4)]
    volume = np.abs(np.einsum("ij,ij->i", b-a, np.cross(c-a, d-a))) / 6.0
    stress = np.minimum(stress, np.quantile(stress, 0.99))
    return weighted_gini(stress, volume)


def robust_bounds(values):
    lo, hi = np.quantile(values, [0.025, 0.975])
    return [float(lo), float(hi)]


def bin_values(values, bounds, bins=4):
    lo, hi = bounds
    return np.clip(((np.asarray(values) - lo) / (hi - lo) * bins).astype(int), 0, bins-1)


def occupancy_metrics(first, second, bounds):
    i = bin_values(first, bounds[0]); j = bin_values(second, bounds[1])
    counts = np.bincount(4*i+j, minlength=16)
    probability = counts[counts > 0] / counts.sum()
    entropy = -float(np.sum(probability * np.log(probability))) / math.log(16)
    return int(np.sum(counts > 0)), entropy, counts.reshape(4, 4).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    study, out = args.study.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    summary = json.loads((study / "summary.json").read_text())
    rows = [row for row in summary["results"] if row.get("valid")]
    with np.load(study / "descriptor_reference.npz") as reference:
        xyz = reference["xyz"]
        characteristic_length_mm = float(np.linalg.norm(xyz.max(0)-xyz.min(0)) * 1000)

    observations = []
    for index, row in enumerate(rows, 1):
        values = dict(row["features"])
        values["normalized_void_scale"] = values["void_clearance_mean_mm"] / characteristic_length_mm
        values["stress_concentration_proxy"] = stress_proxy(row["case_dir"])
        observations.append({
            "id": row["id"], "method": row["method"],
            "replicate_group": row["replicate_group"],
            "compliance_J": row["compliance_J"],
            "material_volume_fraction": values["material_volume_fraction"],
            **{name: values[name] for name in FEATURES},
        })
        print(f"{index}/{len(rows)} {row['id']}", flush=True)

    volume = np.array([row["material_volume_fraction"] for row in observations])
    log_compliance = np.log(np.array([row["compliance_J"] for row in observations]))
    bounds = {name: robust_bounds([row[name] for row in observations]) for name in FEATURES}
    repeated = {}
    for row in observations:
        if row["replicate_group"].startswith("recipe_"):
            repeated.setdefault(row["replicate_group"], []).append(row)

    per_axis = {}
    for name in FEATURES:
        values = np.array([row[name] for row in observations])
        span = bounds[name][1] - bounds[name][0]
        repeat_ranges = [np.ptp([row[name] for row in group]) for group in repeated.values()]
        per_axis[name] = {
            "robust_bounds": bounds[name],
            "abs_spearman_with_volume": abs_spearman(values, volume),
            "abs_spearman_with_log_compliance": abs_spearman(values, log_compliance),
            "mean_replicate_range_over_robust_span": float(np.mean(repeat_ranges) / span),
        }

    pairs = []
    for first, second in itertools.combinations(FEATURES, 2):
        a = np.array([row[first] for row in observations])
        b = np.array([row[second] for row in observations])
        occupied, entropy, counts = occupancy_metrics(a, b, [bounds[first], bounds[second]])
        objective_overlap = max(
            per_axis[first]["abs_spearman_with_volume"],
            per_axis[first]["abs_spearman_with_log_compliance"],
            per_axis[second]["abs_spearman_with_volume"],
            per_axis[second]["abs_spearman_with_log_compliance"],
        )
        repeat_instability = max(
            per_axis[first]["mean_replicate_range_over_robust_span"],
            per_axis[second]["mean_replicate_range_over_robust_span"],
        )
        mutual = abs_spearman(a, b)
        score = (.35 * occupied/16 + .20 * entropy + .15 * (1-mutual)
                 + .15 * (1-objective_overlap) + .15 * max(0, 1-repeat_instability))
        pairs.append({
            "axes": [first, second], "occupied_cells": occupied,
            "coverage": occupied/16, "occupancy_entropy": entropy,
            "axis_abs_spearman": mutual,
            "max_objective_abs_spearman": objective_overlap,
            "max_repeat_instability": repeat_instability,
            "diagnostic_score": score, "cell_counts": counts,
        })
    pairs.sort(key=lambda item: item["diagnostic_score"], reverse=True)

    proposed = next(item for item in pairs if item["axes"] ==
                    ["normalized_void_scale", "stress_concentration_proxy"])
    current = next(item for item in pairs if item["axes"] ==
                   ["normalized_void_scale", "material_anisotropy"])
    quantitative = pairs[0]
    report = {
        "status": "retrospective_descriptor_development_only",
        "source": str(study / "summary.json"),
        "valid_realizations": len(observations),
        "characteristic_length_mm": characteristic_length_mm,
        "objectives": ["minimize compliance_J", "minimize material_volume_fraction"],
        "recommended_axes": ["normalized_void_scale", "strain_energy_concentration"],
        "recommended_definitions": {
            "normalized_void_scale": "mean void-to-material clearance / design-domain diagonal",
            "strain_energy_concentration":
                "1-exp(-KL(element strain-energy share || element volume share))",
        },
        "why": [
            "The first axis distinguishes fine porous structures from large openings and arches.",
            "The second distinguishes distributed load transfer from a few concentrated load paths.",
            "The concentration descriptor normalizes total energy, so it is distinct from compliance.",
            "Both definitions are dimensionless and can transfer from the bracket to a bridge domain.",
        ],
        "existing_data_limitation":
            "Existing VTU files lack element strain energy. The audit uses a winsorized, volume-weighted von Mises Gini proxy; future FEA runs export the exact descriptor.",
        "recommended_pair_proxy_diagnostics": proposed,
        "current_pair_diagnostics": current,
        "best_purely_quantitative_existing_pair": quantitative,
        "per_axis": per_axis,
        "ranked_pairs": pairs,
    }
    (out / "behavior_axis_audit.json").write_text(json.dumps(report, indent=2) + "\n")

    x = np.array([row["normalized_void_scale"] for row in observations])
    y = np.array([row["stress_concentration_proxy"] for row in observations])
    fig, ax = plt.subplots(figsize=(7.2, 5.6), layout="constrained")
    colors = {"shared": "#777777", "raqd": "#2463a6", "random": "#d0782a"}
    for method, color in colors.items():
        mask = np.array([row["method"] == method for row in observations])
        ax.scatter(x[mask], y[mask], s=44, alpha=.8, label=method, color=color)
    for value in np.linspace(*bounds["normalized_void_scale"], 5): ax.axvline(value, color="#999", lw=.7, alpha=.45)
    for value in np.linspace(*bounds["stress_concentration_proxy"], 5): ax.axhline(value, color="#999", lw=.7, alpha=.45)
    ax.set(xlabel="Normalized void scale", ylabel="Stress-concentration proxy",
           title=f"Proposed behavior space proxy ({proposed['occupied_cells']}/16 occupied cells)")
    ax.legend(); ax.grid(alpha=.12)
    fig.savefig(out / "recommended_axes_proxy.png", dpi=190)
    fig.savefig(out / "recommended_axes_proxy.svg")
    plt.close(fig)

    md = f"""# Behavior-axis audit

## Recommendation

Use **normalized void scale × strain-energy concentration**.

- `normalized_void_scale = mean void-to-material clearance / design-domain diagonal`
- `strain_energy_concentration = 1 - exp(-KL(element energy share || element volume share))`

The horizontal axis moves from fine porosity to large openings/arches. The vertical axis moves from distributed load transfer to a few concentrated load paths. Compliance remains the total strain energy objective; the second behavior axis records only its normalized spatial distribution.

## Existing 51-result check

The existing FEA files did not save element strain energy, so this audit uses the volume-weighted Gini coefficient of 99th-percentile-winsorized von Mises stress as a proxy. The proposed proxy pair occupies **{proposed['occupied_cells']}/16 cells**, has axis correlation **{proposed['axis_abs_spearman']:.3f}**, and maximum correlation with either objective **{proposed['max_objective_abs_spearman']:.3f}**.

The strongest pair by the numerical diagnostic alone is `{quantitative['axes'][0]} × {quantitative['axes'][1]}` ({quantitative['occupied_cells']}/16 cells), but its coordinate-specific meaning transfers poorly to bridge domains. The old `void clearance × material anisotropy` proxy occupies {current['occupied_cells']}/16 cells in this retrospective rebinning.

This is descriptor development, not an unbiased method comparison. Bounds and formulas must be fixed before the next independent RAB-MOQD run.

## Outputs

- `{out / 'behavior_axis_audit.json'}`
- `{out / 'recommended_axes_proxy.png'}`
- `{out / 'recommended_axes_proxy.svg'}`
"""
    (out / "behavior_axis_audit.md").write_text(md)
    print(out / "behavior_axis_audit.json")


if __name__ == "__main__":
    main()
