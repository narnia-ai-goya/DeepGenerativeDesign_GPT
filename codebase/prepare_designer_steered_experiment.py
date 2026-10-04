#!/usr/bin/env python3
"""Materialize a preregistered designer-steered Semantic BO-QD experiment."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from designer_steering import initial_state, save_state


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent
DEFAULT_CONFIG = ROOT / "configs/designer_steered_semantic_bo_qd_experiment.json"
DEFAULT_OUT = DATA_ROOT / "experiments/bracket/designer_steered_semantic_bo_qd_2026-09-22"


MASS_LANGUAGE = {
    "low": "very lightweight and open, using only a few continuous load-bearing members",
    "medium": "balanced material usage with broad load paths and several clear openings",
    "high": "robust and redundant while retaining clearly separated structural openings",
}


def validate(config: dict) -> None:
    archive = config["archive"]
    niches = archive["semantic_niches"]
    edges = archive["mass_edges"]
    if len(niches) * (len(edges) - 1) != archive["cells"]:
        raise ValueError("archive.cells does not match niches × mass bins")
    if edges != sorted(edges) or len(set(edges)) != len(edges):
        raise ValueError("mass_edges must be strictly increasing")
    if config["hard_gates"].get("designer_override") is not False:
        raise ValueError("Designer override of hard gates is prohibited")
    for phase in ("pilot", "main"):
        p = config[phase]
        if p["dense_evaluations_per_round"] > p["image_candidates_per_round"]:
            raise ValueError(f"{phase}: dense budget exceeds image budget")
        if p["sparse_fea_evaluations_per_round"] > p["dense_evaluations_per_round"]:
            raise ValueError(f"{phase}: sparse/FEA budget exceeds dense budget")


def prompt_bank(config: dict) -> list[dict]:
    reference = "the supplied real bracket envelope with four mounting bores and two left load lugs"
    fixed = ("Preserve the exact outer envelope and all colored boundary-condition interfaces. "
             "Show a single front orthographic view, matte black metal on pure white background, "
             "with readable structural openings and no added hardware.")
    rows = []
    for niche, description in config["archive"]["semantic_niches"].items():
        for mass_level, language in MASS_LANGUAGE.items():
            rows.append({
                "prompt_id": f"{niche}__{mass_level}",
                "semantic_niche_target": niche,
                "mass_level_target": mass_level,
                "prompt": f"Design {reference}: {description}; {language}. {fixed}",
            })
    return rows


def steering_state(condition: str, config: dict) -> dict:
    state = initial_state()
    state["experiment_condition"] = condition
    state["archive_axes"] = deepcopy(config["archive"]["axes"])
    state["hard_gates"]["checks"] = list(config["hard_gates"])
    state["hard_gates"]["checks"].remove("designer_override")
    state["intervention_policy"] = config["conditions"][condition]["designer_intervention"]
    if condition in {"initial_intent", "periodic_steering"}:
        state["semantic_anchors"] = {
            "continuous_load_path": 0.8,
            "redundant_branching": 0.5,
            "large_clear_openings": 0.4,
        }
        state["target_mass_range"] = [0.28, 0.40]
    else:
        state["semantic_anchors"] = {}
        state["target_mass_range"] = [config["archive"]["mass_edges"][0],
                                      config["archive"]["mass_edges"][-1]]
    return state


def materialize(config: dict, phase: str, out: Path) -> None:
    validate(config)
    phase_config = config[phase]
    out.mkdir(parents=True, exist_ok=True)
    snapshot = deepcopy(config)
    snapshot["selected_phase"] = phase
    snapshot["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    snapshot["absolute_paths"] = {
        key: str((DATA_ROOT / config[key]).resolve())
        for key in ("reference_image", "design_domain", "fixed_bc", "load_bc")
    }
    (out / "protocol_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    (out / "prompt_bank.json").write_text(
        json.dumps({"prompts": prompt_bank(config)}, indent=2, ensure_ascii=False) + "\n")

    budget_rows = []
    for condition, condition_config in config["conditions"].items():
        for seed in phase_config["seeds"]:
            run = out / "runs" / condition / f"seed_{seed}"
            (run / "rounds").mkdir(parents=True, exist_ok=True)
            manifest = {
                "condition": condition,
                "seed": seed,
                "selector": condition_config["selector"],
                "designer_intervention": condition_config["designer_intervention"],
                "warm_start_source": str((out / "shared_warm_start" / f"seed_{seed}").resolve()),
                "budgets": phase_config,
                "status": "prepared",
            }
            (run / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            save_state(run / "designer_steering_state.json", steering_state(condition, config))
            for round_index in range(phase_config["rounds"]):
                round_dir = run / "rounds" / f"round_{round_index:02d}"
                round_dir.mkdir(exist_ok=True)
                (round_dir / "budget.json").write_text(json.dumps({
                    "image_candidates": phase_config["image_candidates_per_round"],
                    "dense_evaluations": phase_config["dense_evaluations_per_round"],
                    "sparse_fea_evaluations": phase_config["sparse_fea_evaluations_per_round"],
                    "designer_checkpoint": condition == "periodic_steering",
                }, indent=2) + "\n")
            budget_rows.append({
                "condition": condition, "seed": seed,
                "warm_start_full": phase_config["shared_warm_start_full_evaluations"],
                "rounds": phase_config["rounds"],
                "images": phase_config["rounds"] * phase_config["image_candidates_per_round"],
                "dense": phase_config["rounds"] * phase_config["dense_evaluations_per_round"],
                "sparse_fea": phase_config["rounds"] * phase_config["sparse_fea_evaluations_per_round"],
            })
    for seed in phase_config["seeds"]:
        shared = out / "shared_warm_start" / f"seed_{seed}"
        shared.mkdir(parents=True, exist_ok=True)
        (shared / "README.txt").write_text(
            "The same fully evaluated warm-start records are referenced by every condition.\n")
    with (out / "matched_budgets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(budget_rows[0]))
        writer.writeheader(); writer.writerows(budget_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("pilot", "main"), default="pilot")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    materialize(config, args.phase, args.output)
    print(args.output.resolve())
    print((args.output / "protocol_snapshot.json").resolve())
    print((args.output / "matched_budgets.csv").resolve())


if __name__ == "__main__":
    main()
