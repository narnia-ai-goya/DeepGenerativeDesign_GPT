import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raqd_v2_posterior import MixedExactGP


def genome(x, style="bionic_bone"):
    return {"style": style, "cfg": 3 + 6*x, "sp_cfg": 3 + 4*x,
            "sp_guide_w_peak": 20 + 60*x}


def test_exact_gp_predicts_and_returns_bounded_interval_probability():
    xs = np.linspace(.05, .95, 15)
    genomes = [genome(x, "bionic_bone" if i % 2 else "de_trilattice")
               for i, x in enumerate(xs)]
    y = .45 + .1 * xs + .01 * np.sin(8 * xs)
    gp = MixedExactGP().fit(genomes, y)
    mean, std, probability = gp.interval_probability([genome(.5)], .475, .525)
    assert abs(mean[0] - .5) < .03
    assert std[0] > 0
    assert 0 <= probability[0] <= 1
