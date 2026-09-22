"""Expand the Gaussian-posterior versus Sobol comparison to 12 evaluations each."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rab_moqd_acquisition import (archive_fronts, cell_of, hypervolume_2d,
                                  propose_rab_moqd)
from raqd_posterior import sobol_genomes
from run_rab_moqd_smoke import (DESCRIPTOR_RANGES, OBJECTIVE_BOUNDS, evaluate,
                                save)

ROOT = Path(__file__).resolve().parents[1]
WARMSTART = ROOT / "experiments/bracket/rab_moqd_warmstart_nu030_2026-09-14.json"
INITIAL = ROOT / "experiments/bracket/rab_moqd_experiment_suite_2026-09-14/summary.json"
GRID = ROOT / "experiments/bracket/rab_moqd_combined_analysis_2026-09-14/combined_analysis.json"
DEFAULT_OUT = ROOT / "experiments/bracket/rab_moqd_expanded_2026-09-14"
DIMS = (8, 8)
METHODS = ("posterior_sampled", "sobol_random")


def archive_metrics(rows, thresholds):
    fronts = archive_fronts(rows, DESCRIPTOR_RANGES, OBJECTIVE_BOUNDS, DIMS,
                            cell_thresholds=thresholds)
    return len(fronts), sum(hypervolume_2d(front) for front in fronts.values())


def cumulative(warm, rows, thresholds):
    base_cells, base_hv = archive_metrics(warm, thresholds)
    result = [{"evaluation": 0, "id": "warm_start", "valid": True,
               "occupied_cells": base_cells, "new_cells": 0,
               "qd_hypervolume": base_hv, "qd_hypervolume_gain": 0.0}]
    accumulated = list(warm)
    for index, row in enumerate(rows, 1):
        accumulated.append(row)
        cells, hv = archive_metrics(accumulated, thresholds)
        result.append({"evaluation": index, "id": row["id"],
                       "valid": bool(row.get("valid")), "occupied_cells": cells,
                       "new_cells": cells-base_cells, "qd_hypervolume": hv,
                       "qd_hypervolume_gain": hv-base_hv})
    return result


def run_jobs(jobs, gpus, out, characteristic_length_mm):
    def worker(item):
        index, job = item
        return evaluate(job, gpus[index % len(gpus)], out, characteristic_length_mm)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        return list(pool.map(worker, enumerate(jobs)))


def write_report(out, summary):
    colors = {"posterior_sampled": "#2767a5", "sobol_random": "#d07a2b"}
    labels = {"posterior_sampled": "Gaussian posterior", "sobol_random": "Sobol random"}
    fig, (ax_hv, ax_gain) = plt.subplots(1, 2, figsize=(12.2, 4.8), layout="constrained")
    for name in METHODS:
        curve = summary["methods"][name]["cumulative"]
        x = [step["evaluation"] for step in curve]
        ax_hv.plot(x, [step["qd_hypervolume"] for step in curve], marker="o", ms=4,
                   lw=2, color=colors[name], label=labels[name])
        ax_gain.plot(x, [step["qd_hypervolume_gain"] for step in curve], marker="o", ms=4,
                     lw=2, color=colors[name], label=labels[name])
    ax_hv.set(xlabel="Cumulative evaluations", ylabel="QD-HV",
              title="Cumulative QD-HV (open 8×8 grid)")
    ax_gain.set(xlabel="Cumulative evaluations", ylabel="QD-HV gain",
                title="Improvement over common warm start")
    for ax in (ax_hv, ax_gain):
        ax.grid(alpha=.2); ax.legend()
    fig.savefig(out / "cumulative_qd_hv.png", dpi=190); plt.close(fig)

    rows = []
    for name in METHODS:
        item = summary["methods"][name]
        rows.append(f'<tr><td>{labels[name]}</td><td>{item["evaluations"]}</td>'
                    f'<td>{item["valid"]}</td><td>{item["occupied_cells"]}</td>'
                    f'<td>{item["qd_hypervolume"]:.5f}</td>'
                    f'<td>{item["qd_hypervolume_gain"]:+.5f}</td>'
                    f'<td>{item["auc_gain"]:.5f}</td></tr>')
    cards = []
    for result in summary["new_results"]:
        image = (f'<img src="{html.escape(os.path.relpath(result["preview"], out))}">'
                 if result.get("preview") else "")
        detail = (f'C={1000*result["compliance_J"]:.3f} mJ · '
                  f'V={result["measured_design_volume_fraction"]:.3f}'
                  if result.get("valid") else html.escape("; ".join(result["invalid_reasons"])))
        cards.append(f'<article>{image}<h3>{html.escape(result["id"])}</h3>'
                     f'<p>{detail}</p></article>')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Expanded RAB-MOQD comparison</title><style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:34px auto;padding:0 22px;color:#17212b;line-height:1.6}}.note{{background:#eef7f4;border-left:4px solid #177b65;padding:13px 17px}}img.plot{{width:100%;max-width:1120px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd4da;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef2f4}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:14px}}article{{border:1px solid #d5dce2;border-radius:10px;padding:12px}}article img{{width:100%}}code{{overflow-wrap:anywhere}}</style><h1>Expanded RAB-MOQD comparison</h1><p class="note">동일한 warm-start와 사전 고정된 열린 8×8 grid에서 Gaussian posterior와 Sobol random을 각각 12회까지 누적했다. Gaussian posterior는 3개 평가마다 관측값으로 갱신했다.</p><img class="plot" src="cumulative_qd_hv.png"><table><thead><tr><th>방법</th><th>평가</th><th>유효</th><th>점유 cell</th><th>최종 QD-HV</th><th>증가</th><th>누적 gain AUC</th></tr></thead><tbody>{''.join(rows)}</tbody></table><h2>새 realizations</h2><div class="cards">{''.join(cards)}</div><p><code>{out / "summary.json"}</code></p></html>'''
    (out / "report.html").write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gpus", default="1,2,3,4,5,6")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    gpus = [int(value) for value in args.gpus.split(",")]

    warm_payload = json.loads(WARMSTART.read_text())
    warm = warm_payload["results"]
    initial = json.loads(INITIAL.read_text())["comparison_results"]
    prior = {name: [row for row in initial if row["method"] == name] for name in METHODS}
    thresholds = json.loads(GRID.read_text())["recommended_next_grid"]["thresholds"]
    protocol = {
        "status": "frozen_before_expanded_evaluation",
        "created_at": time.time(), "warmstart": str(WARMSTART.resolve()),
        "initial_results": str(INITIAL.resolve()), "grid_source": str(GRID.resolve()),
        "grid": "open 8x8 warm-start quantile thresholds", "cell_thresholds": thresholds,
        "objective_bounds": OBJECTIVE_BOUNDS, "methods": list(METHODS),
        "initial_evaluations_per_method": len(prior["posterior_sampled"]),
        "new_rounds": args.rounds, "batch_size_per_method": args.batch_size,
        "target_evaluations_per_method": len(prior["posterior_sampled"])
                                         + args.rounds*args.batch_size,
        "posterior_update": "after each three-evaluation batch",
        "comparison": "realized cumulative QD-HV and cumulative-gain AUC",
    }
    save(out / "protocol.json", protocol)

    new_results = []
    posterior_rows = list(prior["posterior_sampled"])
    sobol_rows = list(prior["sobol_random"])
    for round_index in range(args.rounds):
        plan_path = out / f"round_{round_index+1:02d}_plan.json"
        if plan_path.exists():
            jobs = json.loads(plan_path.read_text())["jobs"]
        else:
            start = len(posterior_rows)
            proposals, diagnostics = propose_rab_moqd(
                [*warm, *posterior_rows], args.batch_size, 2026091410+round_index,
                DESCRIPTOR_RANGES, OBJECTIVE_BOUNDS, DIMS, sample_posterior=True,
                cell_thresholds=thresholds)
            posterior_jobs = [
                {"id": f"posterior_sampled_{start+i:02d}", "method": "posterior_sampled",
                 "round": round_index+2, "replicate_group": f"posterior_sampled_{start+i:02d}",
                 "seed": 51000+100*round_index+i, **proposal}
                for i, proposal in enumerate(proposals)]
            random_start = len(sobol_rows)
            genomes = sobol_genomes(args.batch_size, 2026091420+round_index,
                                    skip=2048+round_index*args.batch_size)
            sobol_jobs = [
                {"id": f"sobol_random_{random_start+i:02d}", "method": "sobol_random",
                 "round": round_index+2, "replicate_group": f"sobol_random_{random_start+i:02d}",
                 "seed": 52000+100*round_index+i, "target_cell": None,
                 "genome": genome, "posterior": None}
                for i, genome in enumerate(genomes)]
            jobs = [*posterior_jobs, *sobol_jobs]
            save(plan_path, {"jobs": jobs, "note": "frozen before round evaluation"})
            save(out / f"round_{round_index+1:02d}_diagnostics.json", diagnostics)
        results = run_jobs(jobs, gpus, out, warm_payload["characteristic_length_mm"])
        new_results.extend(results)
        posterior_rows.extend(row for row in results if row["method"] == "posterior_sampled")
        sobol_rows.extend(row for row in results if row["method"] == "sobol_random")
        save(out / "checkpoint.json", {"completed_rounds": round_index+1,
                                        "new_results": new_results})

    all_rows = {"posterior_sampled": posterior_rows, "sobol_random": sobol_rows}
    base_cells, base_hv = archive_metrics(warm, thresholds)
    methods = {}
    for name, rows_for_method in all_rows.items():
        curve = cumulative(warm, rows_for_method, thresholds)
        gains = np.asarray([step["qd_hypervolume_gain"] for step in curve])
        methods[name] = {
            "evaluations": len(rows_for_method),
            "valid": sum(bool(row.get("valid")) for row in rows_for_method),
            "occupied_cells": curve[-1]["occupied_cells"],
            "new_cells": curve[-1]["new_cells"],
            "qd_hypervolume": curve[-1]["qd_hypervolume"],
            "qd_hypervolume_gain": curve[-1]["qd_hypervolume_gain"],
            "auc_gain": float(np.trapezoid(gains)), "cumulative": curve,
        }
    summary = {"status": "complete", "protocol": str((out / "protocol.json").resolve()),
               "base_occupied_cells": base_cells, "base_qd_hypervolume": base_hv,
               "methods": methods, "new_results": new_results}
    save(out / "summary.json", summary)
    write_report(out, summary)
    print(out / "report.html")


if __name__ == "__main__":
    main()
