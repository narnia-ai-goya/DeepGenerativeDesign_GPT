"""The shape descriptor should discount width, while retaining layout."""
import numpy as np
from scipy.ndimage import binary_dilation

from audit_semantic_qd_geometry import equal_area, jaccard
from build_realization_aware_qd_archive import canonical_intents


def test_equal_area_reduces_uniform_width_change_but_not_layout_change():
    base = np.zeros((48, 48), dtype=bool)
    base[8:40, 20:24] = True
    base[20:24, 8:40] = True
    thick = binary_dilation(base, iterations=3)
    shifted = np.roll(base, 9, axis=0)
    allowed = np.ones_like(base)
    n = int(base.sum())
    normalized_base = equal_area(base, allowed, n)
    normalized_thick = equal_area(thick, allowed, n)
    normalized_shifted = equal_area(shifted, allowed, n)
    assert jaccard(normalized_base, normalized_thick) < jaccard(base, thick) / 3
    assert jaccard(normalized_base, normalized_shifted) > 0.3


def test_canonical_intent_excludes_mass_wording():
    rows = [{"semantic_niche_target": "branching"},
            {"semantic_niche_target": "longitudinal_spine"}]
    texts = canonical_intents(rows)
    assert "branching structural network" in texts["branching"]
    assert "balanced material usage" not in texts["branching"]
    assert "vertical" in texts["longitudinal_spine"]
