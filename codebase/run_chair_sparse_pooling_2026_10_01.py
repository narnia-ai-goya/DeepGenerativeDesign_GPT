#!/usr/bin/env python3
"""Controlled sparse pooling ablation from the same diagonal-braced dense cache."""
from __future__ import annotations

import concurrent.futures
import json
import subprocess

from make_chair_domain import ROOT
from run_chair_backrest_multi_examples import BC_STUDY, OUT as SOURCE_ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env
from validate_chair_bc_geometry import audit

SOURCE = SOURCE_ROOT / 'diagonal_braced'
OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/sparse_pooling_2026-10-01'
CASES = (
    ('no_pool', 0, {'sp_param_pool_schedule': '', 'sp_pool_anchor_w': 0.0}),
    ('soft_long', 1, {'sp_param_pool_schedule': '0.5:4,0.8:2,1.0:1',
                      'sp_param_pool_blend': True, 'sp_pool_anchor_w': 0.0}),
    ('hard_long', 2, {'sp_param_pool_schedule': '0.5:4,0.8:2,1.0:1',
                      'sp_param_pool_blend': False, 'sp_pool_anchor_w': 0.0}),
    ('anchor_long', 3, {'sp_param_pool_schedule': '0.1:4,0.2:2,1.0:1',
                        'sp_param_pool_blend': True, 'sp_pool_anchor_w': 10.0,
                        'sp_pool_anchor_schedule': '0.5:4,0.8:2,1.0:1'}),
)


def run_case(name, gpu, changes):
    case = OUT / name
    case.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SOURCE / 'sparse_pw2_d13/config_sparse.json').read_text())
    cfg['name'] = 'chair_sparse_pooling_' + name
    cfg['stages']['mesh'].update(changes)
    cfg['stages']['mesh'].update(fea_w=0.0, sp_fea_w=0.0)
    config = case / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    cmd = [str(PYTHON), str(GENERATOR), '--config', str(config),
           '--target-dir', str(SOURCE / 'input_lr162'), '--out', str(case / 'generation')]
    with (case / 'run.log').open('w') as log:
        result = subprocess.run(cmd, cwd=ROOT, env=generation_env(gpu),
                                stdout=log, stderr=subprocess.STDOUT)
    record = {'name': name, 'gpu': gpu, 'config': str(config), 'command': cmd,
              'exit_code': result.returncode,
              'mesh': str(case / 'generation/mesh.obj')}
    if result.returncode == 0:
        record['audit'] = audit(case / 'generation/mesh.obj',
                                BC_STUDY / 'voxel.npz', SOURCE / 'prototype.npz')
        (case / 'audit.json').write_text(json.dumps(record['audit'], indent=2) + '\n')
    (case / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'name': name, 'exit_code': result.returncode,
                      'components': record.get('audit', {}).get('mesh_components'),
                      'bc_pass': record.get('audit', {}).get('bc_geometry_pass')}), flush=True)
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(CASES)) as pool:
        futures = [pool.submit(run_case, *row) for row in CASES]
        records = [f.result() for f in futures]
    (OUT / 'runs.json').write_text(json.dumps(records, indent=2) + '\n')
    return max(r['exit_code'] for r in records)


if __name__ == '__main__':
    raise SystemExit(main())
