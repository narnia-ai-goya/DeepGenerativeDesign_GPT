#!/usr/bin/env python3
"""Ablate sparse morphological max-pool r_min geometry guidance."""
from __future__ import annotations

import concurrent.futures
import json
import subprocess

from make_chair_domain import ROOT
from run_chair_backrest_multi_examples import BC_STUDY, OUT as SOURCE_ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env
from validate_chair_bc_geometry import audit

SOURCE = SOURCE_ROOT / 'diagonal_braced'
OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/sparse_rmin_pooling_2026-10-01'
CASES = (
    ('solid_k2_w1', 0, 2, 1.0, 0.0),
    ('dual_k2_w1', 1, 2, 1.0, 0.3),
    ('dual_k4_w1', 2, 4, 1.0, 0.3),
    ('dual_k2_w3', 3, 2, 3.0, 0.3),
)


def run_case(name, gpu, radius, weight, void_weight):
    case = OUT / name
    case.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SOURCE / 'sparse_pw2_d13/config_sparse.json').read_text())
    cfg['name'] = 'chair_sparse_rmin_pooling_' + name
    cfg['stages']['mesh'].update(sp_sdf_inside_low=True,
                                sp_rmin_w=weight,
                                sp_r_min_voxels=radius,
                                sp_rmin_void_weight=void_weight,
                                fea_w=0.0, sp_fea_w=0.0)
    config = case / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    cmd = [str(PYTHON), str(GENERATOR), '--config', str(config),
           '--target-dir', str(SOURCE / 'input_lr162'), '--out', str(case / 'generation')]
    with (case / 'run.log').open('w') as log:
        result = subprocess.run(cmd, cwd=ROOT, env=generation_env(gpu),
                                stdout=log, stderr=subprocess.STDOUT)
    record = {'name': name, 'gpu': gpu, 'radius': radius,
              'weight': weight, 'void_weight': void_weight,
              'config': str(config), 'command': cmd, 'exit_code': result.returncode,
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
