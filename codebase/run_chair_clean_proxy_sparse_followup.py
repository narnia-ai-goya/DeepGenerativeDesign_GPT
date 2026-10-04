#!/usr/bin/env python3
"""Sparse follow-up for the best clean shape-target dense chair candidate."""
from __future__ import annotations

import json
import subprocess

from make_chair_domain import ROOT
from run_chair_backrest_multi_examples import BC_STUDY, OUT as SOURCE_ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env
from validate_chair_bc_geometry import audit

SOURCE = SOURCE_ROOT / 'diagonal_braced'
PILOT = ROOT / 'experiments/chair/sofa_style_2026-09-28/clean_proxy_dense_2026-09-30'
OUT = PILOT / 'hull_only_pw2_sparse_d13'


def main():
    OUT.mkdir(exist_ok=True)
    config = json.loads((SOURCE / 'sparse_pw2_d13/config_sparse.json').read_text())
    config['name'] = 'chair_clean_proxy_hull_only_pw2_sparse_d13'
    config['stages']['mesh'].update(
        shape_anchor_bank=str(PILOT / 'image_hull_bc_only.npz'),
        load_dense_cache=str(PILOT / 'hull_only_pw2/dense_cache.npz'),
        save_dense_cache=None, skip_sparse=False,
        fea_w=0.0, sp_fea_w=0.0)
    path = OUT / 'config.json'
    path.write_text(json.dumps(config, indent=2) + '\n')
    cmd = [str(PYTHON), str(GENERATOR), '--config', str(path),
           '--target-dir', str(SOURCE / 'input_lr162'), '--out', str(OUT / 'generation')]
    with (OUT / 'run.log').open('w') as log:
        process = subprocess.run(cmd, cwd=ROOT, env=generation_env(0),
                                 stdout=log, stderr=subprocess.STDOUT)
    result = {'command': cmd, 'exit_code': process.returncode,
              'mesh': str(OUT / 'generation/mesh.obj')}
    if process.returncode == 0:
        result['audit'] = audit(OUT / 'generation/mesh.obj',
                                BC_STUDY / 'voxel.npz', PILOT / 'image_hull_bc_only.npz')
        (OUT / 'audit.json').write_text(json.dumps(result['audit'], indent=2) + '\n')
    (OUT / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'exit_code': process.returncode,
                      'bc_pass': result.get('audit', {}).get('bc_geometry_pass'),
                      'shape_pass': result.get('audit', {}).get('shape_gate_pass')}, indent=2))
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
