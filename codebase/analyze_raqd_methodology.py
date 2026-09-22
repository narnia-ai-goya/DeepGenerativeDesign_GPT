#!/usr/bin/env python3
"""Diagnose the completed bracket study and emit evidence for RA-QD v2 design."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "experiments/bracket/raqd_online_2026-09-13"


def finite(value):
    return value is not None and np.isfinite(float(value))


def unique_rows(summary):
    rows = {}
    for method in ("raqd", "random"):
        for row in summary["methods"][method]["attempts"]:
            rows.setdefault(row["id"], row)
    return list(rows.values())


def bootstrap_ci(values, statistic, rng, draws=10000):
    values = np.asarray(values, dtype=float)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    stats = np.apply_along_axis(statistic, 1, sampled)
    return [float(v) for v in np.quantile(stats, [.025, .975])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    args = parser.parse_args()
    study = args.study.resolve()
    summary = json.loads((study / "summary.json").read_text())
    rng = np.random.default_rng(20260913)

    unique = [r for r in unique_rows(summary) if r.get("valid")]
    raqd = [r for r in summary["methods"]["raqd"]["attempts"] if r["id"].startswith("raqd_")]
    random = [r for r in summary["methods"]["random"]["attempts"] if r["id"].startswith("random_")]
    proposed = [r for r in raqd if r.get("posterior_sample")]
    optimized = [r for r in proposed if not r["posterior_sample"].get("epsilon_random")]

    pred_v = np.array([r["posterior_sample"]["volume_fraction"] for r in proposed])
    true_v = np.array([r["measured_design_volume_fraction"] for r in proposed])
    pred_d = np.array([r["posterior_sample"]["descriptors"] for r in proposed])
    true_d = np.array([r["realized_descriptors"] for r in proposed])
    pred_c = np.array([r["posterior_sample"]["compliance_J"] for r in proposed])
    true_c = np.array([r["compliance_J"] for r in proposed])

    stages = [r for r in unique if all(k in r.get("realization_path", {})
                                      for k in ("dense", "sparse", "final"))]
    dense_v = np.array([r["realization_path"]["dense"]["material_volume_fraction"] for r in stages])
    sparse_v = np.array([r["realization_path"]["sparse"]["material_volume_fraction"] for r in stages])
    final_v = np.array([r["realization_path"]["final"]["material_volume_fraction"] for r in stages])
    delta = final_v - sparse_v
    stable = np.abs(delta) <= .05
    coef = np.polyfit(sparse_v[stable], final_v[stable], 1)
    predicted_final = np.polyval(coef, sparse_v[stable])
    r2 = 1 - np.sum((final_v[stable] - predicted_final) ** 2) / np.sum(
        (final_v[stable] - final_v[stable].mean()) ** 2)
    target_sparse = float((.5 - coef[1]) / coef[0])

    styles = {}
    for style in sorted({r["genome"]["style"] for r in unique}):
        rows = [r for r in unique if r["genome"]["style"] == style]
        styles[style] = {
            "n": len(rows),
            "volume_feasible": sum(bool(r.get("constraints_satisfied")) for r in rows),
            "volume_feasible_rate": float(np.mean([r.get("constraints_satisfied", False) for r in rows])),
            "mean_final_volume_fraction": float(np.mean([r["measured_design_volume_fraction"] for r in rows])),
        }

    replicate_groups = []
    shared = [r for r in unique if r["id"].startswith("shared_")]
    for group in sorted({r["replicate_group"] for r in shared}):
        rows = [r for r in shared if r["replicate_group"] == group]
        replicate_groups.append({
            "group": group,
            "n": len(rows),
            "final_volume_range": float(np.ptp([r["measured_design_volume_fraction"] for r in rows])),
            "descriptor_0_range": float(np.ptp([r["realized_descriptors"][0] for r in rows])),
            "descriptor_1_range": float(np.ptp([r["realized_descriptors"][1] for r in rows])),
            "compliance_ratio_max_over_min": float(max(r["compliance_J"] for r in rows) /
                                                    min(r["compliance_J"] for r in rows)),
        })

    parameter_correlations = {}
    for parameter in ("cfg", "sp_cfg", "sp_guide_w_peak"):
        x = [r["genome"][parameter] for r in unique]
        parameter_correlations[parameter] = {
            "spearman_with_final_volume": float(spearmanr(x, final_v).statistic),
            "spearman_with_compliance": float(spearmanr(x, [r["compliance_J"] for r in unique]).statistic),
        }

    diagnostic = {
        "study": str(study),
        "valid_unique_evaluations": len(unique),
        "posterior_selection": {
            "adaptive_raqd": len(proposed),
            "non_epsilon_raqd": len(optimized),
            "sampled_volume_in_band_non_epsilon": sum(
                abs(r["posterior_sample"]["volume_fraction"] - .5) <= .025 for r in optimized),
            "realized_volume_feasible_non_epsilon": sum(r["constraints_satisfied"] for r in optimized),
            "realized_volume_feasible_all_raqd": sum(r["constraints_satisfied"] for r in proposed),
            "realized_volume_feasible_random": sum(r["constraints_satisfied"] for r in random),
            "volume_mae": float(np.mean(np.abs(pred_v - true_v))),
            "volume_bias_pred_minus_actual": float(np.mean(pred_v - true_v)),
            "descriptor_mae": [float(v) for v in np.mean(np.abs(pred_d - true_d), axis=0)],
            "log_compliance_mae": float(np.mean(np.abs(np.log(pred_c) - np.log(true_c)))),
            "target_cell_hit_rate": float(np.mean([
                r.get("target_cell") == r.get("realized_cell") for r in proposed])),
        },
        "stage_transfer": {
            "n": len(stages),
            "stable_n_abs_delta_le_0.05": int(stable.sum()),
            "sparse_to_final_delta_median": float(np.median(delta[stable])),
            "sparse_to_final_delta_mae": float(np.mean(np.abs(delta[stable]))),
            "sparse_to_final_delta_mae_95pct_bootstrap_ci": bootstrap_ci(
                np.abs(delta[stable]), np.mean, rng),
            "sparse_final_spearman": float(spearmanr(sparse_v[stable], final_v[stable]).statistic),
            "sparse_final_linear_slope": float(coef[0]),
            "sparse_final_linear_intercept": float(coef[1]),
            "sparse_final_linear_r2": float(r2),
            "estimated_sparse_target_for_final_0.5": target_sparse,
            "outliers": [{"id": r["id"], "sparse": float(s), "final": float(f), "delta": float(d)}
                         for r, s, f, d, keep in zip(stages, sparse_v, final_v, delta, stable) if not keep],
            "dense_final_spearman": float(spearmanr(dense_v, final_v).statistic),
        },
        "replicate_noise": {
            "groups": replicate_groups,
            "median_final_volume_range": float(np.median([g["final_volume_range"] for g in replicate_groups])),
            "median_descriptor_0_range": float(np.median([g["descriptor_0_range"] for g in replicate_groups])),
            "median_descriptor_1_range": float(np.median([g["descriptor_1_range"] for g in replicate_groups])),
        },
        "style_summary": styles,
        "parameter_rank_correlations": parameter_correlations,
    }
    (study / "methodology_diagnostics.json").write_text(
        json.dumps(diagnostic, indent=2, allow_nan=False) + "\n")

    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    ax.scatter(pred_v, true_v, c=["#d55e00" if r.get("constraints_satisfied") else "#777777" for r in proposed])
    ax.axvspan(.475, .525, color="#56b4e9", alpha=.15)
    ax.axhspan(.475, .525, color="#009e73", alpha=.12)
    ax.plot([.35, .66], [.35, .66], "--", color="#333333", lw=1)
    ax.set(xlabel="Posterior sample: volume fraction", ylabel="Realized final volume fraction",
           title="A. One draw did not calibrate feasibility", xlim=(.35, .66), ylim=(.35, .66))

    ax = axes[1]
    ax.scatter(sparse_v[stable], final_v[stable], color="#0072b2", alpha=.78)
    if (~stable).any():
        ax.scatter(sparse_v[~stable], final_v[~stable], marker="x", s=70, color="#d55e00", label="outlier")
    line = np.linspace(min(sparse_v), max(sparse_v), 100)
    ax.plot(line, np.polyval(coef, line), color="#222222", lw=1.5,
            label=f"fit: R²={r2:.3f}")
    ax.axhspan(.475, .525, color="#009e73", alpha=.12)
    ax.axvline(target_sparse, color="#cc79a7", ls="--", lw=1.5,
               label=f"target sparse={target_sparse:.3f}")
    ax.set(xlabel="Sparse-stage volume fraction", ylabel="Final volume fraction",
           title="B. Sparse stage predicts final volume")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    labels = ["RA-QD\n(non-ε)", "RA-QD\n(all)", "Random"]
    hits = [sum(r["constraints_satisfied"] for r in optimized),
            sum(r["constraints_satisfied"] for r in proposed),
            sum(r["constraints_satisfied"] for r in random)]
    ns = [len(optimized), len(proposed), len(random)]
    rates = np.array(hits) / ns
    bars = ax.bar(labels, 100 * rates, color=["#d55e00", "#e69f00", "#0072b2"])
    for bar, h, n in zip(bars, hits, ns):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.6, f"{h}/{n}", ha="center")
    ax.set(ylabel="Realized volume-feasible rate (%)", title="C. Feasibility stayed rare", ylim=(0, 22))
    fig.tight_layout()
    fig.savefig(study / "methodology_diagnostics.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(study / "methodology_diagnostics.json")
    print(study / "methodology_diagnostics.png")


if __name__ == "__main__":
    main()
