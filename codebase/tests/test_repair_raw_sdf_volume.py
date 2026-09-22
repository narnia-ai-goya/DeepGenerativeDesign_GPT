import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repair_raw_sdf_volume import calibrated_level, extract_mesh, sample_reference


def test_calibrated_level_matches_fixed_reference_fraction():
    n = 25
    axis = np.arange(n, dtype=float) + .5
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    sdf = np.sqrt((x-12.5)**2 + (y-12.5)**2 + (z-12.5)**2).astype(np.float32)
    xyz = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    values = sample_reference(sdf, xyz, np.zeros(3), np.ones(3))
    level = calibrated_level(values, .2)
    assert abs(np.mean(values <= level) - .2) < .01
    mesh = extract_mesh(sdf, level, np.zeros(3), np.ones(3))
    assert len(mesh.vertices) > 100
    assert abs(mesh.volume) > 100
    assert np.all(mesh.bounds[0] > 0)
