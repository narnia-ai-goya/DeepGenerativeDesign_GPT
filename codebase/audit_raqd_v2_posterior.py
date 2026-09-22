#!/usr/bin/env python3
"""Group-wise cross-validation of the RA-QD v2 volume posterior."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from raqd_v2_posterior import MixedExactGP


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "experiments/bracket/raqd_online_2026-09-13"


def unique_valid(summary):
    found = {}
    for method in ("raqd", "random"):
        for row in summary["methods"][method]["attempts"]:
            if row.get("valid"):
                found.setdefault(row["id"], row)
    return list(found.values())


def metrics(y, mean, std):
    feasible = np.abs(y - .5) <= .025
    p = norm.cdf((.525 - mean) / std) - norm.cdf((.475 - mean) / std)
    return {
        "n": len(y), "mae": float(np.mean(np.abs(y - mean))),
        "rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
        "coverage_80": float(np.mean(np.abs(y - mean) <= norm.ppf(.9) * std)),
        "coverage_90": float(np.mean(np.abs(y - mean) <= norm.ppf(.95) * std)),
        "mean_predictive_std": float(np.mean(std)),
        "feasibility_brier": float(np.mean((p - feasible) ** 2)),
        "mean_predicted_feasibility": float(np.mean(p)),
        "observed_feasibility": float(np.mean(feasible)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    args = ap.parse_args()
    study = args.study.resolve()
    rows = unique_valid(json.loads((study / "summary.json").read_text()))
    genomes = [r["genome"] for r in rows]
    y = np.asarray([r["measured_design_volume_fraction"] for r in rows])
    groups = np.asarray([r.get("replicate_group", r["id"]) for r in rows])

    full = MixedExactGP().fit(genomes, y)
    predicted = np.empty(len(rows)); uncertainty = np.empty(len(rows))
    baseline_mean = np.empty(len(rows)); baseline_std = np.empty(len(rows))
    for group in sorted(set(groups)):
        test = groups == group; train = ~test
        gp = MixedExactGP().fit([g for g, keep in zip(genomes, train) if keep], y[train],
                                theta=full.theta, optimize=False)
        predicted[test], uncertainty[test] = gp.predict(
            [g for g, keep in zip(genomes, test) if keep], realization=True)
        baseline_mean[test] = y[train].mean()
        baseline_std[test] = y[train].std(ddof=1)

    report = {
        "study": str(study),
        "validation": "leave-one-replicate-group-out; shared three-seed recipes held out together",
        "exact_gp": metrics(y, predicted, uncertainty),
        "constant_gaussian_baseline": metrics(y, baseline_mean, baseline_std),
        "full_fit": full.diagnostics(),
        "rows": [{"id": r["id"], "actual": float(actual), "mean": float(mean),
                  "std": float(std), "feasibility_probability": float(
                      norm.cdf((.525-mean)/std)-norm.cdf((.475-mean)/std))}
                 for r, actual, mean, std in zip(rows, y, predicted, uncertainty)],
    }
    (study / "v2_posterior_audit.json").write_text(json.dumps(report, indent=2) + "\n")

    order = np.argsort(predicted)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    axes[0].errorbar(predicted, y, xerr=norm.ppf(.95)*uncertainty, fmt="o", ms=4,
                     alpha=.72, color="#176b87", ecolor="#a9bbc1")
    axes[0].plot([.35,.68],[.35,.68],"--",color="#333")
    axes[0].axhspan(.475,.525,color="#177a65",alpha=.12)
    axes[0].set(xlabel="LO-group-out GP mean ± 90% interval", ylabel="Actual final volume",
                title="A. Realization prediction", xlim=(.3,.72), ylim=(.3,.72))
    idx = np.arange(len(y))
    axes[1].fill_between(idx, predicted[order]-norm.ppf(.95)*uncertainty[order],
                         predicted[order]+norm.ppf(.95)*uncertainty[order], color="#9ec8d6", alpha=.5)
    axes[1].plot(idx,predicted[order],color="#176b87",label="GP mean")
    axes[1].scatter(idx,y[order],s=15,color="#222",label="actual")
    axes[1].axhspan(.475,.525,color="#177a65",alpha=.12,label="feasible band")
    axes[1].set(xlabel="Cases sorted by predicted mean",ylabel="Final volume fraction",
                title="B. Calibration across held-out groups")
    axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout(); fig.savefig(study / "v2_posterior_audit.png", dpi=190); plt.close(fig)
    print(study / "v2_posterior_audit.json")
    print(study / "v2_posterior_audit.png")


if __name__ == "__main__":
    main()
