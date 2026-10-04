#!/usr/bin/env python3
"""Fresh chair dense generation without the previous 3D prototype prior."""
from __future__ import annotations

import argparse
import json
import subprocess

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env
from validate_chair_bc_geometry import audit

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28/front_support_only_2026-10-02/complex_truss_armchair'
OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/fresh_dense_no_prototype_2026-10-02'
COMPLEX_INPUT = ROOT / 'experiments/chair/image_concepts_2026-09-27/complex_truss_armchair/multiview_registered'
OTHER_INPUTS = {
    'open_arm': ROOT / 'experiments/chair/sofa_style_2026-09-28/open_arm',
    'solid_side': ROOT / 'experiments/chair/sofa_style_2026-09-28/solid_side',
}
BC = ROOT / 'experiments/chair/sofa_style_2026-09-28/consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=5)
    ap.add_argument('--variant', choices=('clean', 'image_guided', 'image_strong'), default='clean')
    ap.add_argument('--case', choices=('complex_truss', *OTHER_INPUTS), default='complex_truss')
    args = ap.parse_args()
    source = COMPLEX_INPUT if args.case == 'complex_truss' else OTHER_INPUTS[args.case]
    out = ((OUT if args.variant == 'clean' else OUT / args.variant)
           if args.case == 'complex_truss' else OUT / args.case / args.variant)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((BASE / 'sparse_pw2_d13/config_sparse.json').read_text())
    cfg['name'] = 'chair_complex_truss_fresh_dense_' + args.variant
    cfg['stages']['mesh'].update({
        'load_dense_cache': None,
        'save_dense_cache': str(out / 'dense_cache.npz'),
        'skip_sparse': True,
        'load_path_mask': None,
        'shape_anchor_bank': None,
        'shape_qd_target': -1,
        'shape_anchor_w': 0.0,
        'shape_scaffold_w': 0.0,
        'shape_anchor_expand_vox': 0,
        'sp_shape_anchor_w': 0.0,
        'image_proj_w': 0.0,
        'sp_image_proj_w': 0.0,
        'pw': 0.0,
        'out_w': 20.0,
        'cfg': 7.0,
        'heaviside_proj': False,
        'mc_threshold': 0.4,
        'fea_w': 0.0,
        'sp_fea_w': 0.0,
        'image_proj_target': str(source / 'camera_projection_targets.npz'),
    })
    if args.variant in ('image_guided', 'image_strong'):
        cfg['stages']['mesh'].update({
            'load_path_mask': str(BASE / 'front_support_corridors.npz'),
            'image_proj_w': 50.0 if args.variant == 'image_guided' else 150.0,
            'image_proj_warmup': 0.25,
            'pw': 2.0,
            'out_w': 100.0,
            'cfg': 9.0,
            'heaviside_proj': True,
            'mc_threshold': 0.3,
        })
    config = out / 'config_dense.json'
    config.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(config),
               '--target-dir', str(source / 'input_lr162'), '--out', str(out / 'generation')]
    with (out / 'generation.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=generation_env(args.gpu),
                                stdout=log, stderr=subprocess.STDOUT)
    (out / 'run.json').write_text(json.dumps({'command': command,
        'exit_code': result.returncode}, indent=2) + '\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    dense = out / 'generation/mesh_dense.obj'
    check = audit(dense, BC)
    (out / 'bc_audit.json').write_text(json.dumps(check, indent=2) + '\n')
    print(json.dumps({'mesh': str(dense), 'cache': str(out / 'dense_cache.npz'),
                      'bc_pass': check['bc_geometry_pass'],
                      'components': check['mesh_components']}, indent=2))


if __name__ == '__main__':
    main()
