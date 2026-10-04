#!/usr/bin/env python3
"""Plan round 1 with a condition-specific designer-steered candidate pool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def by_id(rows): return {row["id"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pool_root = args.experiment / "shared_image_pool" / "round_01"
    common_path = pool_root / "common/candidate_images.json"
    periodic_path = pool_root / "periodic/candidate_images.json"
    common = json.loads(common_path.read_text())["records"]
    periodic = json.loads(periodic_path.read_text())["records"]
    common_by = by_id(common); periodic_by = by_id(periodic)
    rng = random.Random(args.seed + 1)
    random_rows = rng.sample(common, 6)
    auto_ids = ["r1_common_arch_tie__low", "r1_common_diagonal_truss__medium",
                "r1_common_radial_fan__medium", "r1_common_branching__high",
                "r1_common_longitudinal_spine__high", "r1_common_ring_lattice__high"]
    initial_ids = [f"r1_common_{niche}__medium" for niche in
                   ("arch_tie", "diagonal_truss", "radial_fan", "branching",
                    "longitudinal_spine", "ring_lattice")]
    periodic_ids = ["r1_periodic_branch_smooth__medium", "r1_periodic_branch_double__high",
                    "r1_periodic_ring_oval__medium", "r1_periodic_ring_double__medium",
                    "r1_periodic_radial_thick__medium", "r1_periodic_radial_gusset__high"]
    selections = {"random": random_rows,
                  "auto_bo_qd": [common_by[i] for i in auto_ids],
                  "initial_intent": [common_by[i] for i in initial_ids],
                  "periodic_steering": [periodic_by[i] for i in periodic_ids]}
    sources = {"random": common_path, "auto_bo_qd": common_path,
               "initial_intent": common_path, "periodic_steering": periodic_path}
    for condition, rows in selections.items():
        target = args.experiment / "runs" / condition / f"seed_{args.seed}" / "rounds/round_01"
        target.mkdir(parents=True, exist_ok=True)
        payload = {"selection_stage": "image_to_dense", "condition": condition,
                   "seed": args.seed, "round": 1, "budget": 6,
                   "candidate_pool": str(sources[condition].resolve()), "selected": rows,
                   "designer_revision": 6 if condition == "periodic_steering" else None}
        (target / "dense_selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    unique = {row["id"]: row for rows in selections.values() for row in rows}
    (pool_root / "unique_dense_jobs.json").write_text(json.dumps({
        "records": list(unique.values()), "logical_evaluations": 24,
        "unique_computations": len(unique), "periodic_pool_is_condition_specific": True}, indent=2) + "\n")
    print((pool_root / "unique_dense_jobs.json").resolve())
    for condition, rows in selections.items(): print(condition, ", ".join(r["id"] for r in rows))


if __name__ == "__main__": main()
