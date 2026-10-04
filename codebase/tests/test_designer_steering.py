from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from designer_steering import apply_action, initial_state, select_batch, steer_candidates


def candidates():
    return [
        {"id": "a", "semantic_niche": "arch", "mass_bin": "low",
         "normalized_mass": .30, "acquisition": 1.0,
         "semantic_scores": {"organic": .8}},
        {"id": "b", "semantic_niche": "arch", "mass_bin": "low",
         "normalized_mass": .35, "acquisition": 1.4,
         "semantic_scores": {"organic": .1}},
        {"id": "c", "semantic_niche": "fan", "mass_bin": "high",
         "normalized_mass": .90, "acquisition": 2.0},
    ]


def test_designer_weights_are_soft_and_rejection_is_explicit():
    state = initial_state()
    state = apply_action(state, {"type": "set_target_cell", "semantic_niche": "arch",
                                 "mass_bin": "low", "weight": 3})
    state = apply_action(state, {"type": "set_anchor_weight", "anchor": "organic", "weight": 1})
    state = apply_action(state, {"type": "reject_candidate", "candidate_id": "b"})
    rows = {r["id"]: r for r in steer_candidates(candidates(), state)}
    assert rows["a"]["steered_acquisition"] > rows["a"]["base_acquisition"]
    assert rows["b"]["steered_acquisition"] is None
    assert rows["c"]["steered_acquisition"] == pytest.approx(.5)
    assert state["hard_gates"]["editable"] is False


def test_locked_candidate_is_selected_first_and_log_is_auditable():
    state = apply_action(initial_state(), {"type": "lock_candidate", "candidate_id": "a"})
    rows = steer_candidates(candidates(), state)
    selected = select_batch(rows, 2)
    assert selected[0]["id"] == "a"
    assert state["revision"] == 1
    assert state["interaction_log"][0]["action"]["type"] == "lock_candidate"


def test_hard_gate_override_is_prohibited():
    with pytest.raises(ValueError):
        apply_action(initial_state(), {"type": "override_gate", "gate": "FEA"})
