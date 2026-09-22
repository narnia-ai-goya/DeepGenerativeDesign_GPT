"""Small analytic checks without loading the pretrained generation backbone."""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_conditioning_case import hole_stats, image_metrics
import run_from_image as pipeline


class ConditioningStudyTests(unittest.TestCase):
    def test_failed_retry_does_not_report_stale_artifacts_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for name in ['mesh.obj', 'hybrid.obj', 'final.obj']:
                (out/name).write_text('old result')
            c = {'config': 'configs/caliper.json'}
            with patch.object(pipeline, 'run', return_value=7), \
                 patch.object(pipeline, 'find_nvml_preload', return_value=None):
                self.assertFalse(pipeline.stage_gen('caliper', c, 'test', True, wd=out))
            with patch.object(pipeline, 'run', return_value=7) as run:
                self.assertFalse(pipeline.stage_post('caliper', c, 'test', True, wd=out))
                self.assertEqual(run.call_count, 1, 'Do not remesh stale union after failure')
            (out/'fea').mkdir()
            summary = out/'fea/fea_tet_summary.json'
            summary.write_text(json.dumps({'compliance': 0.01}))
            with patch.object(pipeline, 'run', return_value=7):
                self.assertFalse(pipeline.stage_fea('caliper', c, 'test', True, wd=out))
            self.assertFalse(summary.exists(), 'Remove old FEA summary before retry')

    def test_enclosed_voids_exclude_exterior_notches(self):
        solid = np.zeros((64, 64), dtype=bool)
        solid[8:56, 8:56] = True
        solid[20:30, 20:30] = False
        solid[35:45, 35:45] = False
        self.assertEqual(hole_stats(solid, 0.5)['count'], 2)
        self.assertEqual(hole_stats(solid, 0.5)['area_mm2'], 50)
        solid[:25, 22:25] = False
        self.assertEqual(hole_stats(solid, 0.5)['count'], 1)
        self.assertIsNone(image_metrics(solid, np.ones_like(solid),
                                       np.zeros_like(solid), 0.5)['visible_bc_retention'])

    def test_corrected_topology_gradient_keeps_dense_interior_solid(self):
        tree = ast.parse((ROOT/'code/generate_with_physics_guidance.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'topology_preservation_loss')
        # Execute the actual call-site convention assignments, not a test copy.
        assignments = {n.targets[0].id: n for n in ast.walk(tree)
                       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                       and n.targets[0].id in ('_sdf_sp_in', 'topology_sdf')}
        ns = {'np': np, 'torch': torch, 'F': F}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<topology>', 'exec'), ns)
        dense = torch.zeros((64, 64, 64))
        dense[16:48, 16:48, 16:48] = 1
        coords = torch.tensor([[0, 260, 260, 260]])
        for corrected in (False, True):
            raw = torch.tensor([[0.4]], requires_grad=True)
            ns.update(args=SimpleNamespace(mc_threshold=0.3, sp_topology_sdf_fix=corrected),
                      sdf_sp=raw)
            exec(compile(ast.Module(body=[assignments['_sdf_sp_in'], assignments['topology_sdf']],
                                    type_ignores=[]), '<callsite>', 'exec'), ns)
            _, loss = ns['topology_preservation_loss'](ns['topology_sdf'], coords, dense,
                                                       interior_w=10, interior_erode=2)
            loss.backward()
            if corrected:
                self.assertLess(raw.grad.item(), 0, 'Descent must increase positive-inside density')
            else:
                self.assertGreater(raw.grad.item(), 0, 'Historical sign bug should be reproduced')


if __name__ == '__main__':
    unittest.main()
