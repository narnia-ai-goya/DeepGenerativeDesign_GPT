import math
from pathlib import Path
import random
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qd_archive import GridArchive, PARAMETERS, descriptors_from_occupancy, propose


class ArchiveTests(unittest.TestCase):
    def candidate(self, ident, c, d=(.1, .2), valid=True):
        return {'id': ident, 'compliance_J': c, 'descriptors': d, 'valid': valid,
                'genome': {'style': 'a', 'cfg': 7., 'sp_cfg': 5., 'sp_guide_w_peak': 80.}}

    def test_boundaries_and_outside_not_clipped(self):
        a = GridArchive()
        self.assertEqual(a.index((0, 1)), (0, 3))
        self.assertEqual(a.index((.25, .75)), (1, 3))
        for d in [(1.001, .2), (-.001, .2), (math.nan, .5)]:
            self.assertIsNone(a.index(d))

    def test_minimization_feasibility_and_distinct_cells(self):
        a = GridArchive()
        self.assertTrue(a.add(self.candidate('first', .01)))
        self.assertFalse(a.add(self.candidate('worse', .02)))
        self.assertFalse(a.add(self.candidate('invalid', .001, valid=False)))
        self.assertFalse(a.add(self.candidate('nan', math.nan)))
        self.assertTrue(a.add(self.candidate('better', .005)))
        self.assertTrue(a.add(self.candidate('other', .03, (.8, .2))))
        self.assertEqual(a.elites[(0, 0)]['id'], 'better')
        self.assertEqual(a.summary()['coverage'], 2/16)

    def test_spatial_descriptor_distinguishes_equal_volume(self):
        upper = [False, False, True, True]
        self.assertEqual(descriptors_from_occupancy([1, 1, 0, 0], upper), [.5, 0.])
        self.assertEqual(descriptors_from_occupancy([0, 0, 1, 1], upper), [.5, 1.])
        with self.assertRaises(ValueError): descriptors_from_occupancy([0]*4, upper)

    def test_archive_drives_parent_and_mutations_stay_in_bounds(self):
        a = GridArchive(); a.add(self.candidate('elite', .01))
        rng = random.Random(42)
        for _ in range(100):
            genome, parent = propose(rng, ['a', 'b'], a)
            self.assertEqual(parent, 'elite')
            for k, (lo, hi) in PARAMETERS.items(): self.assertTrue(lo <= genome[k] <= hi)
        self.assertIsNone(propose(rng, ['a', 'b'])[1])


if __name__ == '__main__':
    unittest.main()
