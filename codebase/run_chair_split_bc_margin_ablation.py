#!/usr/bin/env python3
"""Hold foot BC at 13 mm; vary only the seat/back load-STL BC margin."""
from __future__ import annotations

import concurrent.futures
import json
import subprocess

from make_chair_domain import ROOT
from run_chair_backrest_multi_examples import BC_STUDY, OUT as SOURCE_ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env
from validate_chair_bc_geometry import audit

SOURCE = SOURCE_ROOT / 'diagonal_braced'
OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/split_bc_margin_2026-09-30'
CASES = (('load_0mm', 0.0, 0), ('load_6mm', 6.0, 1))


def run_one(row):
    name, load_margin, gpu = row
    case = OUT / name
    case.mkdir(parents=True, exist_ok=True)
    config = json.loads((SOURCE / 'sparse_pw2_d13/config_sparse.json').read_text())
    config['name'] = 'chair_split_bc_' + name
    config['stages']['mesh'].update(force_bc_dilate_mm=13.0,
                                   force_load_dilate_mm=load_margin,
                                   fea_w=0.0, sp_fea_w=0.0)
    path = case / 'config.json'
    path.write_text(json.dumps(config, indent=2) + '\n')
    cmd = [str(PYTHON), str(GENERATOR), '--config', str(path),
           '--target-dir', str(SOURCE / 'input_lr162'), '--out', str(case / 'generation')]
    with (case / 'run.log').open('w') as log:
        process = subprocess.run(cmd, cwd=ROOT, env=generation_env(gpu),
                                 stdout=log, stderr=subprocess.STDOUT)
    result = {'name': name, 'load_margin_mm': load_margin, 'fix_margin_mm': 13.0,
              'gpu': gpu, 'exit_code': process.returncode, 'command': cmd,
              'mesh': str(case / 'generation/mesh.obj')}
    if process.returncode == 0:
        result['audit'] = audit(case / 'generation/mesh.obj',
                                BC_STUDY / 'voxel.npz', SOURCE / 'prototype.npz')
        (case / 'audit.json').write_text(json.dumps(result['audit'], indent=2) + '\n')
    (case / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(run_one, CASES))
    (OUT / 'runs.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps([{'name': r['name'], 'exit_code': r['exit_code'],
                       'bc_pass': r.get('audit', {}).get('bc_geometry_pass'),
                       'shape_pass': r.get('audit', {}).get('shape_gate_pass')}
                      for r in rows], indent=2))
    return max(r['exit_code'] for r in rows)


if __name__ == '__main__':
    raise SystemExit(main())
