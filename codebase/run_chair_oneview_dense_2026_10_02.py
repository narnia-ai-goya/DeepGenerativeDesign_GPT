#!/usr/bin/env python3
"""One-view chair dense ablation with one-view image projection guidance."""
from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env
from validate_chair_bc_geometry import audit

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SOURCE = BASE / 'open_arm'
REFERENCE = BASE / 'fresh_dense_no_prototype_2026-10-02/open_arm/image_strong/config_dense.json'
OUT = BASE / 'fresh_dense_no_prototype_2026-10-02/oneview_open_arm'
BC = BASE / 'consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
VIEWS = {'front': 'v00_front_lo', 'right': 'v02_right_lo', 'top': 'v_top'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('view', choices=(*VIEWS, 'front_allproj'))
    ap.add_argument('--gpu', type=int, default=5)
    args = ap.parse_args()
    out = OUT / args.view
    out.mkdir(parents=True, exist_ok=True)
    input_view = 'front' if args.view == 'front_allproj' else args.view
    if args.view == 'front_allproj':
        target_path = SOURCE / 'camera_projection_targets.npz'
    else:
        source = np.load(SOURCE / 'camera_projection_targets.npz')
        target = {'active_threshold': source['active_threshold'],
                  f'target_{input_view}': source[f'target_{input_view}'],
                  f'weight_{input_view}': source[f'weight_{input_view}'],
                  f'camera_grid_{input_view}': source[f'camera_grid_{input_view}']}
        target_path = out / 'oneview_projection_target.npz'
        np.savez_compressed(target_path, **target)
    cfg = json.loads(REFERENCE.read_text())
    cfg['name'] = 'chair_open_arm_oneview_' + args.view
    cfg['views'] = VIEWS[input_view]
    cfg['stages']['mesh'].update({
        'views': VIEWS[input_view], 'n_views': 1,
        'image_proj_target': str(target_path),
        'load_dense_cache': None,
        'save_dense_cache': str(out / 'dense_cache.npz'),
        'skip_sparse': True,
    })
    config = out / 'config_dense.json'
    config.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(config),
               '--target-dir', str(SOURCE / 'input_lr162'), '--out', str(out / 'generation')]
    with (out / 'generation.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=generation_env(args.gpu),
                                stdout=log, stderr=subprocess.STDOUT)
    (out / 'run.json').write_text(json.dumps({'command': command,
        'exit_code': result.returncode}, indent=2) + '\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    mesh = out / 'generation/mesh_dense.obj'
    check = audit(mesh, BC)
    (out / 'bc_audit.json').write_text(json.dumps(check, indent=2) + '\n')
    print(json.dumps({'mesh': str(mesh), 'bc_pass': check['bc_geometry_pass'],
                      'components': check['mesh_components']}, indent=2))


if __name__ == '__main__':
    main()
