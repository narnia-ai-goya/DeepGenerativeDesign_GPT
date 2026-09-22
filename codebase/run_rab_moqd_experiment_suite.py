"""Run small RAB-MOQD baselines, posterior ablation, and seed verification."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from rab_moqd_acquisition import archive_fronts, hypervolume_2d, propose_rab_moqd
from raqd_posterior import sobol_genomes
from run_rab_moqd_smoke import (DESCRIPTOR_RANGES, DIMS, OBJECTIVE_BOUNDS,
                                evaluate, save)

ROOT = Path(__file__).resolve().parents[1]
FULL_RAB = ROOT / "experiments/bracket/rab_moqd_smoke_2026-09-13/summary.json"
WARMSTART = ROOT / "experiments/bracket/rab_moqd_warmstart_nu030_2026-09-14.json"
DEFAULT_OUT = ROOT / "experiments/bracket/rab_moqd_experiment_suite_2026-09-14"


def qd_hv(rows):
    fronts = archive_fronts(rows, DESCRIPTOR_RANGES, OBJECTIVE_BOUNDS, DIMS)
    return len(fronts), sum(hypervolume_2d(front) for front in fronts.values())


def run_jobs(jobs, gpus, out, characteristic_length_mm):
    queues = [jobs[index::len(gpus)] for index in range(len(gpus))]
    def worker(gpu_and_queue):
        gpu, queue = gpu_and_queue
        return [evaluate(job, gpu, out, characteristic_length_mm) for job in queue]
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        groups = list(pool.map(worker, zip(gpus, queues)))
    by_id = {row["id"]: row for group in groups for row in group}
    return [by_id[job["id"]] for job in jobs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    args = parser.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    warm_payload = json.loads(WARMSTART.read_text()); warm = warm_payload["results"]
    full_summary = json.loads(FULL_RAB.read_text())
    full_rows = full_summary["attempts"]

    plan_path = out / "comparison_plan.json"
    if plan_path.exists():
        jobs = json.loads(plan_path.read_text())["jobs"]
    else:
        sampled_items, sampled_diagnostics = propose_rab_moqd(
            warm, 3, 2026091400, DESCRIPTOR_RANGES, OBJECTIVE_BOUNDS, DIMS,
            sample_posterior=True)
        sampled_jobs = [{"id": f"posterior_sampled_{i:02d}", "method": "posterior_sampled",
                         "round": 1, "replicate_group": f"posterior_sampled_{i:02d}",
                         "seed": 49900+i, **item} for i, item in enumerate(sampled_items)]
        mean_items, diagnostics = propose_rab_moqd(
            warm, 3, 2026091401, DESCRIPTOR_RANGES, OBJECTIVE_BOUNDS, DIMS,
            sample_posterior=False)
        mean_jobs = [{"id": f"posterior_mean_{i:02d}", "method": "posterior_mean",
                      "round": 1, "replicate_group": f"posterior_mean_{i:02d}",
                      "seed": 50000+i, **item} for i, item in enumerate(mean_items)]
        random_jobs = [{"id": f"sobol_random_{i:02d}", "method": "sobol_random",
                        "round": 1, "replicate_group": f"sobol_random_{i:02d}",
                        "seed": 50100+i, "target_cell": None,
                        "genome": genome, "posterior": None}
                       for i, genome in enumerate(sobol_genomes(3, 2026091402, skip=512))]
        jobs = [*sampled_jobs, *mean_jobs, *random_jobs]
        save(plan_path, {"jobs": jobs, "note": "frozen before comparison evaluation"})
        save(out / "posterior_sampled_diagnostics.json", sampled_diagnostics)
        save(out / "posterior_mean_diagnostics.json", diagnostics)
    protocol = {
        "status": "small_method_development_suite_not_final_benchmark",
        "created_at": time.time(), "warmstart": str(WARMSTART),
        "pipeline_smoke_results": str(FULL_RAB), "new_evaluations": len(jobs)+4,
        "comparison": ["posterior-sampled RAB-MOQD", "posterior-mean ablation", "Sobol random"],
        "evaluations_per_comparison_method": 3,
        "replication": "two additional seeds for each of the two best full-RAB proposals",
        "descriptor_ranges": DESCRIPTOR_RANGES, "objective_bounds": OBJECTIVE_BOUNDS,
        "volume_hard_constraint": False,
    }
    save(out / "protocol.json", protocol)
    gpus = [int(value) for value in args.gpus.split(",")]
    comparison_rows = run_jobs(jobs, gpus, out, warm_payload["characteristic_length_mm"])

    base_cells, base_hv = qd_hv(warm)
    methods = {"posterior_sampled": [r for r in comparison_rows if r["method"] == "posterior_sampled"],
               "posterior_mean": [r for r in comparison_rows if r["method"] == "posterior_mean"],
               "sobol_random": [r for r in comparison_rows if r["method"] == "sobol_random"]}
    comparison = {}
    for name, rows in methods.items():
        cells, hv = qd_hv([*warm, *rows])
        comparison[name] = {"evaluations": len(rows), "valid": sum(r["valid"] for r in rows),
                            "occupied_cells": cells, "new_cells": cells-base_cells,
                            "qd_hypervolume": hv, "qd_hypervolume_gain": hv-base_hv}

    # Replicate the two full-posterior recipes with the largest realized archive gain.
    ranked = []
    for row in methods["posterior_sampled"]:
        _, hv = qd_hv([*warm, row])
        ranked.append((hv-base_hv, row))
    selected = [row for _, row in sorted(ranked, key=lambda pair: pair[0], reverse=True)[:2]]
    replicate_jobs = []
    for recipe_index, source in enumerate(selected):
        for repeat in range(2):
            replicate_jobs.append({
                "id": f"verify_{recipe_index:02d}_seed{repeat+1:02d}", "method": "verification",
                "round": 2, "replicate_group": source["id"], "seed": 50200+10*recipe_index+repeat,
                "target_cell": source["target_cell"], "genome": source["genome"],
                "posterior": source.get("posterior"), "source_result": source["id"],
            })
    save(out / "replication_plan.json", {"jobs": replicate_jobs})
    repeats = run_jobs(replicate_jobs, gpus[:len(replicate_jobs)], out,
                       warm_payload["characteristic_length_mm"])
    verification = []
    for source in selected:
        group = [source, *[row for row in repeats if row["source_result"] == source["id"]]]
        cells = []
        from rab_moqd_acquisition import cell_of
        for row in group:
            cell = cell_of(row["realized_descriptors"], DESCRIPTOR_RANGES, DIMS) if row["valid"] else None
            cells.append(list(cell) if cell else None)
        modal = max((cell for cell in cells if cell is not None), key=cells.count, default=None)
        verification.append({
            "source_result": source["id"], "evaluations": len(group),
            "valid": sum(row["valid"] for row in group), "realized_cells": cells,
            "modal_cell": modal, "modal_cell_agreement": cells.count(modal)/len(cells) if modal else 0,
            "compliance_J": [row.get("compliance_J") for row in group],
            "volume_fraction": [row.get("measured_design_volume_fraction") for row in group],
            "descriptors": [row.get("realized_descriptors") for row in group],
        })
    summary = {"status": "complete", "base_occupied_cells": base_cells,
               "base_qd_hypervolume": base_hv, "comparison": comparison,
               "comparison_results": comparison_rows, "verification": verification,
               "verification_results": repeats}
    save(out / "summary.json", summary)
    print(out / "summary.json")


if __name__ == "__main__":
    main()
