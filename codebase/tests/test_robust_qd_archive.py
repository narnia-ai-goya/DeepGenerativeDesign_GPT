import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robust_qd_archive import RobustRealizationArchive


def result(seed, feasible=True, compliance=.01, descriptor=(.2, .2)):
    return {"genome": {"style": "bionic_bone", "cfg": 5., "sp_cfg": 5.,
                       "sp_guide_w_peak": 50.}, "seed": seed, "valid": True,
            "constraints_satisfied": feasible, "compliance_J": compliance,
            "realized_descriptors": list(descriptor)}


def test_requires_replicates_and_uses_risk_quantile():
    archive = RobustRealizationArchive((2,2), ((0,1),(0,1)))
    archive.record(result(1, compliance=.01))
    assert archive.summary()["verified_elites"] == 0
    archive.record(result(2, compliance=.02))
    summary = archive.summary()
    assert summary["verified_elites"] == 1
    assert .01 < summary["elites"][0]["robust_compliance_J"] < .02
