"""Exercise the real coordinator without expensive mesh generation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_qd_pilot as pilot


class ScheduleTests(unittest.TestCase):
    def test_equal_budget_feedback_and_resume_plans(self):
        protocol = {'dims': [4, 4], 'ranges': [[0, 1], [0, 1]],
                    'rounds': 2, 'random_seed': 20260913}
        calls = []

        def fake_evaluate(job, gpu, out, frozen):
            calls.append(job)
            # New rounds improve quality, so second-round parent must be a first-round elite.
            return {**job, 'valid': True, 'descriptors': [.4, .4],
                    'compliance_J': .03 - .01*job['round'], 'seconds': 1}

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            argv = ['pilot', '--out', tmp, '--gpus', '0,1,2']
            with patch.object(pilot, 'prepare', return_value=protocol), \
                 patch.object(pilot, 'reference_samples'), \
                 patch.object(pilot, 'evaluate', side_effect=fake_evaluate), \
                 patch.object(sys, 'argv', argv):
                pilot.main()
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['phase'], 'complete')
            self.assertEqual(len(calls), 15)
            self.assertEqual(len({j['id'] for j in calls}), 15)
            for method in ['map_elites', 'random']:
                self.assertEqual(summary['methods'][method]['evaluations'], 9)
                self.assertEqual([x['id'] for x in summary['methods'][method]['history'][:3]],
                                 ['initial_00', 'initial_01', 'initial_02'])
            for job in calls:
                if job['method'] == 'random': self.assertIsNone(job['parent'])
                if job['method'] == 'map_elites' and job['round'] == 2:
                    self.assertTrue(job['parent'].startswith('map_elites_r1_'))
            plans = {p.name: p.read_bytes() for p in out.glob('jobs_*.json')}
            calls.clear()
            with patch.object(pilot, 'prepare', return_value=protocol), \
                 patch.object(pilot, 'reference_samples'), \
                 patch.object(pilot, 'evaluate', side_effect=fake_evaluate), \
                 patch.object(sys, 'argv', argv):
                pilot.main()
            self.assertEqual(plans, {p.name: p.read_bytes() for p in out.glob('jobs_*.json')})


if __name__ == '__main__':
    unittest.main()
