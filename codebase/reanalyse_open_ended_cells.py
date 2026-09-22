"""Development-only reanalysis using quantile thresholds and open outer cells."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rab_moqd_acquisition import hypervolume_2d, nondominated, objective_point

ROOT = Path(__file__).resolve().parents[1]
WARM = ROOT / "experiments/bracket/rab_moqd_warmstart_nu030_2026-09-14.json"
SUITE = ROOT / "experiments/bracket/rab_moqd_experiment_suite_2026-09-14/summary.json"
JOINT = ROOT / "experiments/bracket/rab_moqd_joint_residual_2026-09-14/summary.json"
OUT = ROOT / "experiments/bracket/rab_moqd_open_cell_reanalysis_2026-09-14.json"
OBJECTIVE_BOUNDS = [[.0035, .014], [.35, .70]]


def cell_of(values, thresholds):
    return tuple(int(np.searchsorted(edge, value, side="right"))
                 for value, edge in zip(values, thresholds))


def archive(rows, thresholds):
    fronts = {}
    for row in rows:
        if not row.get("valid"): continue
        cell = cell_of(row["realized_descriptors"], thresholds)
        point = objective_point(row["compliance_J"], row["measured_design_volume_fraction"],
                                OBJECTIVE_BOUNDS)
        fronts.setdefault(cell, []).append(point)
    return {cell: nondominated(points) for cell, points in fronts.items()}


def metrics(rows, thresholds):
    fronts = archive(rows, thresholds)
    return {"occupied_cells": len(fronts),
            "qd_hypervolume": sum(hypervolume_2d(front) for front in fronts.values())}


def main():
    warm = json.loads(WARM.read_text())["results"]
    suite = json.loads(SUITE.read_text())
    joint = json.loads(JOINT.read_text())["attempts"]
    descriptor_values = np.asarray([row["realized_descriptors"] for row in warm])
    thresholds = np.quantile(descriptor_values, [.25, .50, .75], axis=0).T.tolist()
    base = metrics(warm, thresholds)
    methods = {
        name: [row for row in suite["comparison_results"] if row["method"] == name]
        for name in ("posterior_sampled", "posterior_mean", "sobol_random")}
    methods["joint_residual"] = joint
    comparison = {}
    for name, rows in methods.items():
        result = metrics([*warm, *rows], thresholds)
        comparison[name] = {**result,
                            "new_cells": result["occupied_cells"]-base["occupied_cells"],
                            "qd_hypervolume_gain": result["qd_hypervolume"]-base["qd_hypervolume"]}
    payload = {
        "status": "retrospective_descriptor_development_only",
        "cell_definition": "three frozen development quartile thresholds per axis; outer cells are unbounded",
        "thresholds": {"normalized_void_scale": thresholds[0],
                       "strain_energy_concentration": thresholds[1]},
        "base": base, "comparison": comparison,
        "warning": "Do not use this post-hoc rebinning as the final method comparison. Freeze it before a new run.",
    }
    OUT.write_text(json.dumps(payload, indent=2)+"\n")
    print(OUT)


if __name__ == "__main__": main()
