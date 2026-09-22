"""Evaluate a joint-residual posterior-sampling RAB-MOQD batch."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from rab_moqd_acquisition import cell_of, propose_rab_moqd
from run_rab_moqd_experiment_suite import qd_hv
from run_rab_moqd_smoke import (DESCRIPTOR_RANGES, DIMS, OBJECTIVE_BOUNDS,
                                evaluate, save)

ROOT = Path(__file__).resolve().parents[1]
WARM = ROOT / "experiments/bracket/rab_moqd_warmstart_nu030_2026-09-14.json"
AUDIT = ROOT / "experiments/bracket/rab_moqd_surrogate_audit_2026-09-14/surrogate_audit.json"
DEFAULT_OUT = ROOT / "experiments/bracket/rab_moqd_joint_residual_2026-09-14"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gpus", default="0,1,2")
    args = parser.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    warm_payload = json.loads(WARM.read_text()); warm = warm_payload["results"]
    audit = json.loads(AUDIT.read_text())
    order = ["normalized_void_scale", "strain_energy_concentration",
             "volume_fraction", "log_compliance"]
    residuals = [[row[name] for name in order]
                 for row in audit["mixed_exact_gp_joint_standardized_residuals"]]
    plan_path = out / "proposal.json"
    if plan_path.exists(): jobs = json.loads(plan_path.read_text())["jobs"]
    else:
        proposals, diagnostics = propose_rab_moqd(
            warm, 3, 2026091403, DESCRIPTOR_RANGES, OBJECTIVE_BOUNDS, DIMS,
            joint_standardized_residuals=residuals)
        jobs = [{"id": f"joint_residual_{i:02d}", "method": "joint_residual",
                 "round": 1, "replicate_group": f"joint_residual_{i:02d}",
                 "seed": 50300+i, **item} for i, item in enumerate(proposals)]
        save(plan_path, {"jobs": jobs, "residual_source": str(AUDIT)})
        save(out / "acquisition_diagnostics.json", diagnostics)
    gpus = [int(value) for value in args.gpus.split(",")]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(evaluate, job, gpus[i], out,
                               warm_payload["characteristic_length_mm"])
                   for i, job in enumerate(jobs)]
        rows = [future.result() for future in futures]
    base_cells, base_hv = qd_hv(warm); cells, hv = qd_hv([*warm, *rows])
    enriched = []
    for row in rows:
        cell = cell_of(row["realized_descriptors"], DESCRIPTOR_RANGES, DIMS) if row["valid"] else None
        enriched.append({**row, "realized_cell": list(cell) if cell else None,
                         "target_hit": cell == tuple(row["target_cell"]) if cell else False})
    summary = {"status": "complete", "method": "joint standardized residual bootstrap",
               "evaluations": len(rows), "valid": sum(row["valid"] for row in rows),
               "target_hits": sum(row["target_hit"] for row in enriched),
               "occupied_cells_before": base_cells, "occupied_cells_after": cells,
               "qd_hypervolume_before": base_hv, "qd_hypervolume_after": hv,
               "qd_hypervolume_gain": hv-base_hv, "attempts": enriched}
    save(out / "summary.json", summary)
    print(out / "summary.json")


if __name__ == "__main__": main()
