#!/usr/bin/env python3
"""Promote matched-budget dense candidates to sparse/FEA for round 0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


PREFERENCE = {"arch_tie": 2.2, "ring_lattice": 1.7, "branching": 1.5,
              "radial_fan": 1.3, "diagonal_truss": 1.2, "longitudinal_spine": 1.1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()
    evaluation = args.experiment / "shared_evaluations" / f"round_{args.round:02d}"
    dense = {r["id"]: r for r in json.loads((evaluation / "dense_results.json").read_text())}
    warm = json.loads((args.experiment / "shared_warm_start" / f"seed_{args.seed}" /
                       "warm_start_records.json").read_text())["records"]
    occupied_niches = {r["semantic_niche"] for r in warm if r["hard_gate_pass"]}
    selections = {}
    rng = random.Random(args.seed + 1000 + args.round)
    for condition in ("random", "auto_bo_qd", "initial_intent", "periodic_steering"):
        path = args.experiment / "runs" / condition / f"seed_{args.seed}" / "rounds" / f"round_{args.round:02d}"
        selected_ids = [r["id"] for r in json.loads((path / "dense_selection.json").read_text())["selected"]]
        rows = [dense[i] for i in selected_ids if i in dense]
        feasible = [r for r in rows if r["dense_main_face_fraction"] >= .995]
        if condition == "random":
            chosen = rng.sample(feasible, min(3, len(feasible)))
        elif condition == "auto_bo_qd":
            ranked = sorted(feasible, key=lambda r: (
                r["semantic_niche_target"] not in occupied_niches,
                r["dense_main_face_fraction"], -r["dense_components"]), reverse=True)
            chosen, used = [], set()
            for row in ranked:
                if row["semantic_niche_target"] in used: continue
                chosen.append(row); used.add(row["semantic_niche_target"])
                if len(chosen) == 3: break
        elif condition == "periodic_steering":
            best_by_niche = {}
            for row in feasible:
                niche = row["semantic_niche_target"]
                old = best_by_niche.get(niche)
                if old is None or (row["dense_main_face_fraction"], -row["dense_components"]) > \
                        (old["dense_main_face_fraction"], -old["dense_components"]):
                    best_by_niche[niche] = row
            chosen = sorted(best_by_niche.values(), key=lambda r: (
                PREFERENCE[r["semantic_niche_target"]], r["dense_main_face_fraction"]),
                reverse=True)[:3]
        else:
            chosen = sorted(feasible, key=lambda r: (
                PREFERENCE[r["semantic_niche_target"]], r["dense_main_face_fraction"]),
                reverse=True)[:3]
        payload = {"condition": condition, "seed": args.seed, "round": args.round,
                   "budget": 3, "dense_connectivity_gate": "main face fraction >= 0.995",
                   "selected": chosen}
        (path / "sparse_fea_selection.json").write_text(json.dumps(payload, indent=2) + "\n")
        selections[condition] = chosen
    unique = {r["id"]: r for rows in selections.values() for r in rows}
    destination = evaluation / "unique_sparse_fea_jobs.json"
    destination.write_text(json.dumps({"records": list(unique.values()),
                                       "logical_evaluations": 12,
                                       "unique_computations": len(unique)}, indent=2) + "\n")
    print(destination.resolve())
    for condition, rows in selections.items():
        print(condition, ", ".join(r["id"] for r in rows))


if __name__ == "__main__": main()
