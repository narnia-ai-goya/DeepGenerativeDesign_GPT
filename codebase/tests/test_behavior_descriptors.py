import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from behavior_descriptors import linear_tet_strain_energy, strain_energy_concentration


def test_uniform_energy_density_has_zero_concentration():
    volume = np.array([1.0, 2.0, 4.0])
    assert np.isclose(strain_energy_concentration(3.0 * volume, volume), 0.0)


def test_concentration_is_scale_invariant_and_increases_when_localized():
    volume = np.ones(4)
    spread = np.ones(4)
    localized = np.array([97.0, 1.0, 1.0, 1.0])
    score = strain_energy_concentration(localized, volume)
    assert score > strain_energy_concentration(spread, volume)
    assert np.isclose(score, strain_energy_concentration(10.0 * localized, volume))
    assert 0.0 < score < 1.0


def test_linear_tet_rigid_translation_has_zero_energy():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    displacement = np.tile([.2, -.3, .4], (4, 1))
    energy, volume = linear_tet_strain_energy(points, [[0, 1, 2, 3]], displacement,
                                               110e9, .34)
    assert np.isclose(energy[0], 0.0, atol=1e-20)
    assert np.isclose(volume[0], 1/6)
