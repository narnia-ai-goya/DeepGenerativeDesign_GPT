import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_designer_steered_experiment import materialize, validate


ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "configs/designer_steered_semantic_bo_qd_experiment.json").read_text())


def test_archive_and_budget_protocol_is_valid():
    cfg = config(); validate(cfg)
    assert cfg["archive"]["cells"] == 6 * 4
    assert cfg["hard_gates"]["designer_override"] is False
    assert len(set(cfg["pilot"]["seeds"])) == len(cfg["pilot"]["seeds"])


def test_materialized_conditions_have_identical_budgets(tmp_path):
    cfg = config(); materialize(cfg, "pilot", tmp_path)
    manifests = [json.loads(p.read_text()) for p in tmp_path.glob("runs/*/*/run_manifest.json")]
    assert len(manifests) == len(cfg["conditions"]) * len(cfg["pilot"]["seeds"])
    budgets = {json.dumps(m["budgets"], sort_keys=True) for m in manifests}
    assert len(budgets) == 1
    states = [json.loads(p.read_text()) for p in tmp_path.glob("runs/*/*/designer_steering_state.json")]
    assert all(s["hard_gates"]["editable"] is False for s in states)
    periodic = next(s for s in states if s["experiment_condition"] == "periodic_steering")
    auto = next(s for s in states if s["experiment_condition"] == "auto_bo_qd")
    assert periodic["semantic_anchors"] and not auto["semantic_anchors"]
