#!/usr/bin/env python3
"""Test sparse thickness loss using the decoder's negative-inside SDF convention."""
from __future__ import annotations

import concurrent.futures
import json
import subprocess

from make_chair_domain import ROOT
from run_chair_backrest_multi_examples import BC_STUDY, OUT as SOURCE_ROOT
from run_chair_joint_stage_diagnostic import world_pre_refiner
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env
from validate_chair_bc_geometry import audit

SOURCE = SOURCE_ROOT / 'diagonal_braced'
OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/thickness_sign_diagnostic_2026-09-30'
CASES = (('corrected_w10', 10.0, 0), ('corrected_w3', 3.0, 1))


def run_one(row):
    name, thick_weight, gpu = row
    case = OUT / name
    case.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SOURCE / 'sparse_pw2_d13/config_sparse.json').read_text())
    cfg['name'] = 'chair_thickness_sign_' + name
    cfg['stages']['mesh'].update(sp_sdf_inside_low=True, sp_thick_w=thick_weight,
                                fea_w=0.0, sp_fea_w=0.0)
    path = case / 'config.json'
    path.write_text(json.dumps(cfg, indent=2) + '\n')
    cmd = [str(PYTHON), str(GENERATOR), '--config', str(path),
           '--target-dir', str(SOURCE / 'input_lr162'), '--out', str(case / 'generation')]
    env = generation_env(gpu)
    env['D3DS2_SAVE_PRE_REFINER'] = '1'
    with (case / 'run.log').open('w') as log:
        process = subprocess.run(cmd, cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
    result = {'name': name, 'thick_weight': thick_weight, 'gpu': gpu,
              'exit_code': process.returncode, 'command': cmd}
    if process.returncode == 0:
        pre = world_pre_refiner(case / 'generation/mesh_pre_refiner.obj',
                               case / 'generation/mesh_pre_refiner_world.obj')
        final = case / 'generation/mesh.obj'
        result.update(pre_refiner_mesh=str(pre), final_mesh=str(final),
                      pre_audit=audit(pre, BC_STUDY / 'voxel.npz', SOURCE / 'prototype.npz'),
                      final_audit=audit(final, BC_STUDY / 'voxel.npz', SOURCE / 'prototype.npz'))
        (case / 'pre_audit.json').write_text(json.dumps(result['pre_audit'], indent=2) + '\n')
        (case / 'final_audit.json').write_text(json.dumps(result['final_audit'], indent=2) + '\n')
    (case / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(run_one, CASES))
    (OUT / 'runs.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps([{'name': r['name'], 'exit_code': r['exit_code'],
                       'pre_bc': r.get('pre_audit', {}).get('bc_geometry_pass'),
                       'final_bc': r.get('final_audit', {}).get('bc_geometry_pass')}
                      for r in rows], indent=2))
    return max(r['exit_code'] for r in rows)


if __name__ == '__main__':
    raise SystemExit(main())
