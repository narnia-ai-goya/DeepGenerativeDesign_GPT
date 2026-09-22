import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rab_moqd_acquisition import cell_of, hypervolume_2d, nondominated


def test_hypervolume_and_dominance_for_minimization():
    points = [(0.2, 0.8), (0.8, 0.2), (0.9, 0.9)]
    assert set(nondominated(points)) == {(0.2, 0.8), (0.8, 0.2)}
    assert np.isclose(hypervolume_2d(points), .28)


def test_adding_better_point_increases_hypervolume():
    before = hypervolume_2d([(0.5, 0.5)])
    after = hypervolume_2d([(0.5, 0.5), (0.4, 0.4)])
    assert after > before


def test_open_threshold_cells_include_values_outside_training_range():
    thresholds = [[0.2, 0.4, 0.6], [10.0, 20.0, 30.0]]
    assert cell_of((-5.0, 35.0), thresholds=thresholds) == (0, 3)
    assert cell_of((0.4, 20.0), thresholds=thresholds) == (2, 2)
