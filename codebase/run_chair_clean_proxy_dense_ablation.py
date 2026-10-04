#!/usr/bin/env python3
"""Separate image-derived chair shape target from functional support corridors."""
from __future__ import annotations

import concurrent.futures
import json
import subprocess

import numpy as np
import trimesh
from skimage.measure import marching_cubes

from make_chair_domain import ROOT
from run_chair_backrest_multi_examples import BC_STUDY, OUT as SOURCE_ROOT, visual_hull
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env
from validate_chair_bc_geometry import audit

SOURCE = SOURCE_ROOT / 'diagonal_braced'
OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/clean_proxy_dense_2026-09-30'
CASES = (('hull_only_pw2', 2.0, 0), ('hull_only_pw0p5', 0.5, 1))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    bc = np.load(BC_STUDY / 'voxel.npz')
    hull, projected = visual_hull('diagonal_braced', bc)
    old = np.load(SOURCE / 'prototype.npz')['prototypes'][0] > 0.5
    paths = np.load(BC_STUDY / 'all_support_corridors.npz')['mask'].astype(bool)
    target = hull | bc['bc'].astype(bool)
    if not np.all(target <= bc['bracket']):
        raise RuntimeError('Shape target exits envelope')
    if not np.all(target[bc['bc'].astype(bool)]):
        raise RuntimeError('Shape target excludes BC')
    bank = OUT / 'image_hull_bc_only.npz'
    np.savez_compressed(bank, prototypes=target[None].astype(np.float32))
    verts, faces, _, _ = marching_cubes(np.pad(target, 1), 0.5)
    verts = bc['origin'] + (verts - .5) * bc['pitch_xyz']
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(OUT / 'image_hull_bc_only.obj')
    (OUT / 'target_audit.json').write_text(json.dumps({
        'original_proxy_voxels': int(old.sum()),
        'image_hull_bc_voxels': int(target.sum()),
        'corridor_only_voxels_removed': int((old & ~target).sum()),
        'corridor_voxels': int(paths.sum()),
        'projected_voxels': projected,
        'all_bc_preserved': bool(np.all(target[bc['bc'].astype(bool)])),
    }, indent=2) + '\n')
    return bank


def run_one(row, bank):
    name, path_weight, gpu = row
    case = OUT / name
    case.mkdir(exist_ok=True)
    cfg = json.loads((SOURCE / 'config_dense_pw2.json').read_text())
    cfg['name'] = 'chair_clean_proxy_' + name
    cfg['stages']['mesh'].update(
        shape_anchor_bank=str(bank), pw=path_weight,
        load_path_mask=str(BC_STUDY / 'all_support_corridors.npz'),
        save_dense_cache=str(case / 'dense_cache.npz'), load_dense_cache=None,
        skip_sparse=True, fea_w=0.0, sp_fea_w=0.0)
    config = case / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(config),
               '--target-dir', str(SOURCE / 'input_lr162'), '--out', str(case / 'generation')]
    with (case / 'run.log').open('w') as log:
        process = subprocess.run(command, cwd=ROOT, env=generation_env(gpu),
                                 stdout=log, stderr=subprocess.STDOUT)
    result = {'name': name, 'path_weight': path_weight, 'gpu': gpu,
              'exit_code': process.returncode, 'command': command,
              'mesh': str(case / 'generation/mesh_dense.obj')}
    if process.returncode == 0:
        result['audit'] = audit(case / 'generation/mesh_dense.obj',
                                BC_STUDY / 'voxel.npz', bank)
        (case / 'audit.json').write_text(json.dumps(result['audit'], indent=2) + '\n')
    (case / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    bank = prepare()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda row: run_one(row, bank), CASES))
    (OUT / 'runs.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps([{'name': r['name'], 'exit_code': r['exit_code'],
                       'bc_pass': r.get('audit', {}).get('bc_geometry_pass'),
                       'shape_pass': r.get('audit', {}).get('shape_gate_pass')}
                      for r in rows], indent=2))
    return max(r['exit_code'] for r in rows)


if __name__ == '__main__':
    raise SystemExit(main())
