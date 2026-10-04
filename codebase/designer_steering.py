#!/usr/bin/env python3
"""Persistent, auditable designer steering for black-box BO-QD.

Designer input changes soft search priorities only.  Physics and geometry gates are
recorded here for visibility, but remain owned by the evaluation pipeline.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
ACTION_TYPES = {
    "set_target_cell", "lock_candidate", "reject_candidate",
    "request_local_exploration", "set_mass_range", "set_anchor_weight", "note",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def initial_state() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": 0,
        "updated_at": utc_now(),
        "archive_axes": {
            "x": "normalized_mass",
            "y": "semantic_shape_niche",
            "cell_quality": "independent_FEA_performance",
        },
        "semantic_anchors": {},
        "target_mass_range": [0.20, 0.80],
        "target_cells": {},
        "locked_candidates": [],
        "rejected_candidates": [],
        "local_exploration": [],
        "designer_notes": [],
        "hard_gates": {
            "editable": False,
            "owner": "automatic_evaluation_pipeline",
            "checks": [
                "boundary_condition_preservation", "envelope_validity",
                "single_connected_component", "minimum_thickness",
                "independent_FEA_acceptance",
            ],
        },
        "interaction_log": [],
    }


def validate_state(state: dict[str, Any]) -> None:
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported steering schema: {state.get('schema_version')}")
    lo, hi = state["target_mass_range"]
    if not (0 <= float(lo) <= float(hi) <= 1):
        raise ValueError("target_mass_range must satisfy 0 <= low <= high <= 1")
    if state.get("hard_gates", {}).get("editable") is not False:
        raise ValueError("hard_gates must remain immutable")
    overlap = set(state["locked_candidates"]) & set(state["rejected_candidates"])
    if overlap:
        raise ValueError(f"Candidates cannot be locked and rejected: {sorted(overlap)}")
    for key, value in state["target_cells"].items():
        if float(value) < 0:
            raise ValueError(f"Negative target-cell weight: {key}")
    for key, value in state["semantic_anchors"].items():
        if float(value) < 0:
            raise ValueError(f"Negative semantic-anchor weight: {key}")


def load_state(path: str | Path) -> dict[str, Any]:
    state = json.loads(Path(path).read_text())
    validate_state(state)
    return state


def save_state(path: str | Path, state: dict[str, Any]) -> Path:
    validate_state(state)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(target)
    return target.resolve()


def cell_key(semantic_niche: str, mass_bin: str) -> str:
    return f"{semantic_niche}::{mass_bin}"


def apply_action(state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    """Return a new state after one designer action."""
    kind = action.get("type")
    if kind not in ACTION_TYPES:
        raise ValueError(f"Unknown or prohibited action type: {kind!r}")
    result = deepcopy(state)
    actor = str(action.get("actor", "designer"))

    if kind == "set_target_cell":
        key = cell_key(str(action["semantic_niche"]), str(action["mass_bin"]))
        weight = float(action["weight"])
        if weight < 0:
            raise ValueError("Cell weight must be non-negative")
        result["target_cells"][key] = weight
    elif kind == "lock_candidate":
        candidate = str(action["candidate_id"])
        if candidate in result["rejected_candidates"]:
            result["rejected_candidates"].remove(candidate)
        if candidate not in result["locked_candidates"]:
            result["locked_candidates"].append(candidate)
    elif kind == "reject_candidate":
        candidate = str(action["candidate_id"])
        if candidate in result["locked_candidates"]:
            result["locked_candidates"].remove(candidate)
        if candidate not in result["rejected_candidates"]:
            result["rejected_candidates"].append(candidate)
    elif kind == "request_local_exploration":
        request = {
            "candidate_id": str(action["candidate_id"]),
            "radius": float(action.get("radius", 0.15)),
            "weight": float(action.get("weight", 1.5)),
        }
        if request["radius"] <= 0 or request["weight"] < 0:
            raise ValueError("Local exploration needs radius > 0 and weight >= 0")
        result["local_exploration"].append(request)
    elif kind == "set_mass_range":
        result["target_mass_range"] = [float(action["low"]), float(action["high"])]
    elif kind == "set_anchor_weight":
        weight = float(action["weight"])
        if weight < 0:
            raise ValueError("Anchor weight must be non-negative")
        result["semantic_anchors"][str(action["anchor"])] = weight
    elif kind == "note":
        result["designer_notes"].append({"time": utc_now(), "actor": actor,
                                           "text": str(action["text"])})

    result["revision"] = int(result.get("revision", 0)) + 1
    result["updated_at"] = utc_now()
    result["interaction_log"].append({
        "revision": result["revision"], "time": result["updated_at"],
        "actor": actor, "action": {k: v for k, v in action.items() if k != "actor"},
    })
    validate_state(result)
    return result


def steer_candidates(candidates: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply soft preferences without changing validity or hard-gate outcomes.

    Expected candidate fields are ``id``/``name``, ``base_acquisition`` or
    ``acquisition``, and optionally ``semantic_niche``, ``mass_bin``,
    ``normalized_mass``, ``semantic_scores``, and ``parent_id``.
    """
    validate_state(state)
    rejected = set(state["rejected_candidates"])
    locked = set(state["locked_candidates"])
    lo, hi = state["target_mass_range"]
    requests = {row["candidate_id"]: row for row in state["local_exploration"]}
    output = []
    for source in candidates:
        row = deepcopy(source)
        ident = str(row.get("id", row.get("name")))
        base = float(row.get("base_acquisition", row.get("acquisition", 0.0)))
        multiplier = 1.0
        reasons: list[str] = []

        semantic = row.get("semantic_niche")
        mass_bin = row.get("mass_bin")
        key = cell_key(str(semantic), str(mass_bin)) if semantic is not None and mass_bin is not None else None
        if key in state["target_cells"]:
            value = float(state["target_cells"][key])
            multiplier *= value
            reasons.append(f"target_cell×{value:g}")

        mass = row.get("normalized_mass")
        if mass is not None and not (lo <= float(mass) <= hi):
            multiplier *= 0.25
            reasons.append("outside_preferred_mass_range×0.25")

        scores = row.get("semantic_scores", {})
        for anchor, weight in state["semantic_anchors"].items():
            if anchor in scores:
                factor = 1.0 + float(weight) * max(0.0, float(scores[anchor]))
                multiplier *= factor
                reasons.append(f"anchor:{anchor}×{factor:.3g}")

        parent = row.get("parent_id")
        if parent in requests:
            value = float(requests[parent]["weight"])
            multiplier *= value
            reasons.append(f"local:{parent}×{value:g}")

        row["base_acquisition"] = base
        row["designer_multiplier"] = multiplier
        row["designer_reasons"] = reasons
        row["designer_locked"] = ident in locked
        row["designer_rejected"] = ident in rejected
        row["steered_acquisition"] = None if ident in rejected else base * multiplier
        output.append(row)
    return output


def select_batch(candidates: list[dict[str, Any]], batch_size: int,
                 one_per_cell: bool = True) -> list[dict[str, Any]]:
    """Select locked candidates first, then highest steered acquisition."""
    usable = [r for r in candidates if not r.get("designer_rejected")]
    ranked = sorted(usable, key=lambda r: (
        bool(r.get("designer_locked")), float(r.get("steered_acquisition") or 0.0)
    ), reverse=True)
    selected, used = [], set()
    for row in ranked:
        key = (row.get("semantic_niche"), row.get("mass_bin"))
        if key == (None, None):
            key = tuple(row.get("cell", [])) or str(row.get("id", row.get("name")))
        if one_per_cell and key in used and not row.get("designer_locked"):
            continue
        selected.append(row); used.add(key)
        if len(selected) >= batch_size:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    show = sub.add_parser("show"); show.add_argument("--compact", action="store_true")
    target = sub.add_parser("target"); target.add_argument("semantic_niche"); target.add_argument("mass_bin"); target.add_argument("weight", type=float)
    lock = sub.add_parser("lock"); lock.add_argument("candidate_id")
    reject = sub.add_parser("reject"); reject.add_argument("candidate_id")
    explore = sub.add_parser("explore"); explore.add_argument("candidate_id"); explore.add_argument("--radius", type=float, default=.15); explore.add_argument("--weight", type=float, default=1.5)
    mass = sub.add_parser("mass"); mass.add_argument("low", type=float); mass.add_argument("high", type=float)
    anchor = sub.add_parser("anchor"); anchor.add_argument("name"); anchor.add_argument("weight", type=float)
    note = sub.add_parser("note"); note.add_argument("text")
    args = parser.parse_args()

    if args.command == "init":
        path = save_state(args.state, initial_state()); print(path); return
    state = load_state(args.state)
    if args.command == "show":
        print(json.dumps(state, indent=None if args.compact else 2, ensure_ascii=False)); return
    if args.command == "target":
        action = {"type": "set_target_cell", "semantic_niche": args.semantic_niche,
                  "mass_bin": args.mass_bin, "weight": args.weight}
    elif args.command == "lock": action = {"type": "lock_candidate", "candidate_id": args.candidate_id}
    elif args.command == "reject": action = {"type": "reject_candidate", "candidate_id": args.candidate_id}
    elif args.command == "explore": action = {"type": "request_local_exploration", "candidate_id": args.candidate_id, "radius": args.radius, "weight": args.weight}
    elif args.command == "mass": action = {"type": "set_mass_range", "low": args.low, "high": args.high}
    elif args.command == "anchor": action = {"type": "set_anchor_weight", "anchor": args.name, "weight": args.weight}
    else: action = {"type": "note", "text": args.text}
    print(save_state(args.state, apply_action(state, action)))


if __name__ == "__main__":
    main()
