"""Grouped cross-validation of RAB-MOQD descriptor/objective surrogates."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

from rab_moqd_acquisition import cell_of
from raqd_posterior import STYLES, genome_to_unit
from raqd_v2_posterior import MixedExactGP

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/bracket/rab_moqd_warmstart_nu030_2026-09-14.json"
OUT = ROOT / "experiments/bracket/rab_moqd_surrogate_audit_2026-09-14"
RANGES = [[.020, .041], [.58, .78]]


def encode(genomes):
    rows = []
    for genome in genomes:
        style, x = genome_to_unit(genome)
        one_hot = [float(style == index) for index in range(len(STYLES))]
        rows.append([*one_hot, *x])
    return np.asarray(rows)


def regression_metrics(y, mean, std):
    scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), 1e-12)
    safe_std = np.maximum(std, 1e-8*scale)
    standardized_error = np.abs(y-mean)/safe_std
    return {
        "mae": float(np.mean(np.abs(y-mean))),
        "nrmse_over_iqr": float(np.sqrt(np.mean((y-mean)**2))/scale),
        "coverage_90": float(np.mean(np.abs(y-mean) <= norm.ppf(.95)*safe_std)),
        "mean_std_over_iqr": float(np.mean(safe_std)/scale),
        "gaussian_nlpd": float(np.mean(.5*np.log(2*math.pi*safe_std**2)
                                         + .5*((y-mean)/safe_std)**2)),
        "std_multiplier_for_grouped_90_coverage":
            float(np.quantile(standardized_error, .9)/norm.ppf(.95)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.loads(SOURCE.read_text()); rows = payload["results"]
    genomes = [row["genome"] for row in rows]
    groups = np.array([row["replicate_group"] for row in rows])
    x = encode(genomes)
    outputs = {
        "normalized_void_scale": np.array([row["realized_descriptors"][0] for row in rows]),
        "strain_energy_concentration": np.array([row["realized_descriptors"][1] for row in rows]),
        "volume_fraction": np.array([row["measured_design_volume_fraction"] for row in rows]),
        "log_compliance": np.log(np.array([row["compliance_J"] for row in rows])),
    }
    splitter = list(GroupKFold(n_splits=8).split(x, groups=groups))
    predictions = {model: {name: {"mean": np.empty(len(rows)), "std": np.empty(len(rows))}
                           for name in outputs}
                   for model in ("mixed_exact_gp", "extra_trees", "constant_gaussian")}
    for name, y in outputs.items():
        for fold, (train, test) in enumerate(splitter):
            train_genomes = [genomes[index] for index in train]
            test_genomes = [genomes[index] for index in test]
            gp = MixedExactGP().fit(train_genomes, y[train])
            mean, std = gp.predict(test_genomes, realization=True)
            predictions["mixed_exact_gp"][name]["mean"][test] = mean
            predictions["mixed_exact_gp"][name]["std"][test] = std

            forest = ExtraTreesRegressor(n_estimators=256, min_samples_leaf=2,
                                         max_features=1.0, random_state=9100+fold, n_jobs=-1)
            forest.fit(x[train], y[train])
            tree_prediction = np.asarray([tree.predict(x[test]) for tree in forest.estimators_])
            predictions["extra_trees"][name]["mean"][test] = tree_prediction.mean(0)
            predictions["extra_trees"][name]["std"][test] = np.maximum(
                tree_prediction.std(0), .05*np.std(y[train]))

            predictions["constant_gaussian"][name]["mean"][test] = y[train].mean()
            predictions["constant_gaussian"][name]["std"][test] = y[train].std(ddof=1)

    metrics = {model: {name: regression_metrics(outputs[name], values[name]["mean"],
                                                 values[name]["std"])
                       for name in outputs}
               for model, values in predictions.items()}
    actual_cells = [cell_of((outputs["normalized_void_scale"][i],
                             outputs["strain_energy_concentration"][i]), RANGES)
                    for i in range(len(rows))]
    for model in predictions:
        d0 = predictions[model]["normalized_void_scale"]["mean"]
        d1 = predictions[model]["strain_energy_concentration"]["mean"]
        predicted_cells = [cell_of((d0[i], d1[i]), RANGES) for i in range(len(rows))]
        eligible = [i for i, cell in enumerate(actual_cells) if cell is not None]
        metrics[model]["cell_mean_prediction"] = {
            "eligible": len(eligible),
            "accuracy": float(np.mean([predicted_cells[i] == actual_cells[i] for i in eligible])),
            "manhattan_error": float(np.mean([
                sum(abs(predicted_cells[i][j]-actual_cells[i][j]) for j in range(2))
                if predicted_cells[i] is not None else 4 for i in eligible])),
        }

    replicate_groups = []
    for group in sorted(set(groups)):
        indices = np.where(groups == group)[0]
        if len(indices) < 2: continue
        cells = [actual_cells[index] for index in indices]
        modal = max((cell for cell in cells if cell is not None), key=cells.count, default=None)
        replicate_groups.append({"group": group, "n": len(indices),
                                 "cells": [list(cell) if cell else None for cell in cells],
                                 "modal_agreement": cells.count(modal)/len(cells) if modal else 0})
    report = {
        "status": "grouped_cross_validation_on_development_data",
        "source": str(SOURCE), "samples": len(rows),
        "folds": 8, "grouping": "identical genomes remain in the same fold",
        "descriptor_ranges": RANGES, "metrics": metrics,
        "observed_replicate_cell_stability": replicate_groups,
        "mixed_exact_gp_joint_standardized_residuals": [
            {name: float((outputs[name][index]-predictions["mixed_exact_gp"][name]["mean"][index]) /
                         max(predictions["mixed_exact_gp"][name]["std"][index], 1e-12))
             for name in outputs} for index in range(len(rows))],
    }
    (OUT / "surrogate_audit.json").write_text(json.dumps(report, indent=2)+"\n")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), layout="constrained")
    model_names = list(predictions)
    labels = ["Exact GP", "Extra Trees", "Constant"]
    for axis, output in zip(axes[:2], ("normalized_void_scale", "strain_energy_concentration")):
        axis.bar(labels, [metrics[m][output]["nrmse_over_iqr"] for m in model_names],
                 color=["#2866a4", "#23836c", "#888"])
        axis.set(title=output.replace("_", " "), ylabel="Grouped-CV NRMSE / IQR")
        axis.tick_params(axis="x", rotation=15)
    axes[2].bar(labels, [metrics[m]["cell_mean_prediction"]["accuracy"] for m in model_names],
                color=["#2866a4", "#23836c", "#888"])
    axes[2].set(title="Joint behavior cell", ylabel="Mean-prediction accuracy", ylim=(0, 1))
    axes[2].tick_params(axis="x", rotation=15)
    fig.savefig(OUT / "surrogate_comparison.png", dpi=190); plt.close(fig)
    print(OUT / "surrogate_audit.json")


if __name__ == "__main__": main()
