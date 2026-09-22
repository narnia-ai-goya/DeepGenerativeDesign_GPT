import math

from raqd_core import RealizationArchive, cell_index, normalized_gap


def test_cells_reject_values_outside_frozen_calibration():
    assert cell_index([0., 1.], (4, 4), ((0., 1.), (0., 1.))) == (0, 3)
    assert cell_index([-.01, .5], (4, 4), ((0., 1.), (0., 1.))) is None


def test_gap_uses_requested_cell_center():
    assert normalized_gap((1, 2), [.375, .625], (4, 4), ((0., 1.), (0., 1.))) == 0
    assert math.isclose(normalized_gap((0, 0), [.375, .125], (4, 4), ((0., 1.), (0., 1.))), .25)


def test_only_verified_constraint_satisfying_results_enter_realization_archive():
    archive = RealizationArchive()
    invalid = {'id': 'bad', 'target_cell': [1, 1], 'realized_descriptors': [.3, .3],
               'valid': True, 'constraints_satisfied': False, 'compliance_J': .1}
    observed, inserted = archive.record(invalid)
    assert not inserted and observed['realized_cell'] == [1, 1]
    valid = {**invalid, 'id': 'good', 'constraints_satisfied': True, 'compliance_J': .2}
    observed, inserted = archive.record(valid)
    assert inserted and archive.elites[(1, 1)]['id'] == 'good'
    assert archive.summary()['target_hit_rate'] == 1


def test_transition_preserves_misses_and_lower_compliance_replaces_elite():
    archive = RealizationArchive()
    base = {'target_cell': [3, 3], 'realized_descriptors': [.3, .3],
            'valid': True, 'constraints_satisfied': True}
    archive.record({**base, 'id': 'first', 'compliance_J': .2})
    observed, inserted = archive.record({**base, 'id': 'better', 'compliance_J': .1})
    assert inserted and observed['realized_cell'] == [1, 1]
    assert archive.elites[(1, 1)]['id'] == 'better'
    assert archive.summary()['target_hit_rate'] == 0
    assert archive.transitions[((3, 3), (1, 1))] == 2


def test_failed_pipeline_without_descriptors_is_preserved_but_not_archived():
    archive = RealizationArchive()
    observed, inserted = archive.record({'id': 'failed', 'target_cell': [0, 0],
                                         'valid': False, 'constraints_satisfied': False})
    assert not inserted
    assert observed['realized_cell'] is None and observed['realization_gap'] is None
    assert archive.summary()['attempts'] == 1
