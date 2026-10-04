#!/usr/bin/env python3
"""Back-project marked empty image pixels and guide decoded sparse SDF directly."""
from __future__ import annotations

import concurrent.futures
import json
import subprocess
import sys

import numpy as np
import trimesh

from make_chair_domain import ROOT
from run_chair_backrest_multi_examples import BC_STUDY, OUT as SOURCE_ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env
from validate_chair_bc_geometry import audit

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim

SOURCE = SOURCE_ROOT / 'diagonal_braced'
OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/direct_negative_space_2026-09-30'
FOCUS = ROOT / 'experiments/chair/sofa_style_2026-09-28/joint_focus_projection_2026-09-30/joint_focus_targets.npz'
CASES = (('direct_w1', 1.0, 0), ('direct_w5', 5.0, 1))


def prepare_mask():
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(FOCUS)
    bc = np.load(BC_STUDY / 'voxel.npz')
    env = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents)) * 1.5
    half_extent = float(env.extents.max()) / 2 * 1.15
    ijk = np.indices((64, 64, 64)).reshape(3, -1).T
    world = bc['origin'] + (ijk + .5) * bc['pitch_xyz']
    combined = np.zeros(len(world), dtype=bool)
    per_view = {}
    for name, elev, azim in (('front', 15, 0), ('right', 15, 90)):
        eye, up = camera_from_elev_azim(center, radius, elev, azim)
        forward = (center - eye) / np.linalg.norm(center - eye)
        right = np.cross(forward, up); right /= np.linalg.norm(right)
        true_up = np.cross(right, forward); true_up /= np.linalg.norm(true_up)
        relative = world - center
        xx = np.floor((relative @ right / (2 * half_extent) + .5) * 64).astype(int)
        yy = np.floor((.5 - relative @ true_up / (2 * half_extent)) * 64).astype(int)
        valid = (xx >= 0) & (xx < 64) & (yy >= 0) & (yy < 64)
        selected = np.zeros(len(world), dtype=bool)
        focus = data[f'focus_{name}'].astype(bool)
        selected[valid] = focus[yy[valid], xx[valid]]
        combined |= selected
        per_view[name] = int(selected.sum())
    mask = combined.reshape(64, 64, 64) & bc['bracket'].astype(bool) & ~bc['bc'].astype(bool)
    if not mask.any():
        raise RuntimeError('back-projected empty-space mask is empty')
    path = OUT / 'negative_space_mask.npz'
    np.savez_compressed(path, mask=mask.astype(np.uint8))
    (OUT / 'mask_audit.json').write_text(json.dumps({
        'path': str(path), 'voxels': int(mask.sum()),
        'per_view_before_envelope_and_bc': per_view,
        'inside_envelope': bool(np.all(mask <= bc['bracket'])),
        'bc_overlap': int((mask & bc['bc']).sum()),
    }, indent=2) + '\n')
    return path


def run_one(row, mask):
    name, weight, gpu, *options = row
    late_inner = int(options[0]) if options else 3
    guide_lr = float(options[1]) if len(options) > 1 else .005
    case = OUT / name
    case.mkdir(exist_ok=True)
    cfg = json.loads((SOURCE / 'sparse_pw2_d13/config_sparse.json').read_text())
    cfg['name'] = 'chair_direct_negative_space_' + name
    cfg['stages']['mesh'].update(sp_sdf_inside_low=True,
                                sp_negative_space_mask=str(mask),
                                sp_negative_space_w=weight,
                                sp_n_inner_late=late_inner,
                                sp_guide_lr=guide_lr,
                                sp_image_proj_w=0.0,
                                fea_w=0.0, sp_fea_w=0.0)
    path = case / 'config.json'
    path.write_text(json.dumps(cfg, indent=2) + '\n')
    cmd = [str(PYTHON), str(GENERATOR), '--config', str(path),
           '--target-dir', str(SOURCE / 'input_lr162'), '--out', str(case / 'generation')]
    with (case / 'run.log').open('w') as log:
        result = subprocess.run(cmd, cwd=ROOT, env=generation_env(gpu),
                                stdout=log, stderr=subprocess.STDOUT)
    record = {'name': name, 'weight': weight, 'gpu': gpu,
              'late_inner': late_inner, 'guide_lr': guide_lr,
              'exit_code': result.returncode, 'command': cmd,
              'mesh': str(case / 'generation/mesh.obj')}
    if result.returncode == 0:
        record['audit'] = audit(case / 'generation/mesh.obj',
                                BC_STUDY / 'voxel.npz', SOURCE / 'prototype.npz')
        (case / 'audit.json').write_text(json.dumps(record['audit'], indent=2) + '\n')
    (case / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    mask = prepare_mask()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda row: run_one(row, mask), CASES))
    (OUT / 'runs.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps([{'name': r['name'], 'exit_code': r['exit_code'],
                       'bc_pass': r.get('audit', {}).get('bc_geometry_pass'),
                       'components': r.get('audit', {}).get('mesh_components')}
                      for r in rows], indent=2))
    return max(r['exit_code'] for r in rows)


if __name__ == '__main__':
    raise SystemExit(main())
