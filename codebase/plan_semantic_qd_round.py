#!/usr/bin/env python3
"""Create matched-budget round-0 selections for the four pilot conditions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


MASS_BIN = {"low": 0, "medium": 1, "high": 2}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()
    pool_path = args.experiment / "shared_image_pool" / f"round_{args.round:02d}" / "candidate_images.json"
    pool = json.loads(pool_path.read_text())["records"]
    by = {(r["semantic_niche_target"], r["mass_level_target"]): r for r in pool}
    niches = list(dict.fromkeys(r["semantic_niche_target"] for r in pool))

    rng = random.Random(args.seed)
    random_rows = rng.sample(pool, 6)
    auto_levels = ["low", "low", "high", "high", "high", "medium"]
    auto_rows = [by[(niche, level)] for niche, level in zip(niches, auto_levels)]
    intent_rows = [by[(niche, "medium")] for niche in niches]
    selections = {
        "random": random_rows,
        "auto_bo_qd": auto_rows,
        "initial_intent": intent_rows,
        "periodic_steering": intent_rows,
    }
    for condition, rows in selections.items():
        target = args.experiment / "runs" / condition / f"seed_{args.seed}" / "rounds" / f"round_{args.round:02d}"
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "selection_stage": "image_to_dense",
            "condition": condition, "seed": args.seed, "round": args.round,
            "budget": 6, "candidate_pool": str(pool_path.resolve()),
            "selected": rows,
            "note": "Round 0 uses target labels only; realized niches and mass are measured after 3D generation.",
        }
        (target / "dense_selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    unique = {r["id"]: r for rows in selections.values() for r in rows}
    destination = args.experiment / "shared_image_pool" / f"round_{args.round:02d}" / "unique_dense_jobs.json"
    destination.write_text(json.dumps({"records": list(unique.values()),
                                       "logical_evaluations": 24,
                                       "unique_computations": len(unique)}, indent=2) + "\n")
    print(destination.resolve())
    for condition, rows in selections.items():
        print(condition, ", ".join(r["id"] for r in rows))


if __name__ == "__main__":
    main()
