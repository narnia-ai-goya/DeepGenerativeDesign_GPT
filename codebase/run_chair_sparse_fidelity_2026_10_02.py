#!/usr/bin/env python3
"""Probe whether sparse support expansion and image/shape guidance restore chair details."""
from __future__ import annotations

import argparse
import json
import subprocess

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env
from validate_chair_bc_geometry import audit

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28/front_support_only_2026-10-02/complex_truss_armchair'
OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/sparse_fidelity_2026-10-02'
INPUT = ROOT / 'experiments/chair/image_concepts_2026-09-27/complex_truss_armchair/multiview_registered/input_lr162'
BC = ROOT / 'experiments/chair/sofa_style_2026-09-28/consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
VARIANTS = {
    'expand1': dict(shape_anchor_expand_vox=1, shape_anchor_max_added=1500),
    'expand1_guided': dict(shape_anchor_expand_vox=1, shape_anchor_max_added=1500,
                           sp_shape_anchor_w=50.0, sp_image_proj_w=20.0),
    'prototype_core': dict(shape_anchor_expand_vox=1, shape_anchor_max_added=8500,
                           sp_shape_anchor_w=5.0, sp_image_proj_w=0.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variant', choices=VARIANTS)
    ap.add_argument('--gpu', type=int, default=5)
    args = ap.parse_args()
    out = OUT / args.variant
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((BASE / 'sparse_pw2_d13/config_sparse.json').read_text())
    cfg['name'] = 'chair_sparse_fidelity_' + args.variant
    cfg['stages']['mesh'].update(VARIANTS[args.variant])
    cfg_path = out / 'config_sparse.json'
    cfg_path.write_text(json.dumps(cfg, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(cfg_path),
               '--target-dir', str(INPUT), '--out', str(out / 'generation')]
    with (out / 'run.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=generation_env(args.gpu),
                                stdout=log, stderr=subprocess.STDOUT)
    (out / 'run.json').write_text(json.dumps({'command': command,
        'exit_code': result.returncode}, indent=2) + '\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    check = audit(out / 'generation/mesh.obj', BC, BASE / 'prototype.npz')
    (out / 'bc_audit.json').write_text(json.dumps(check, indent=2) + '\n')
    print(json.dumps({'mesh': str(out / 'generation/mesh.obj'),
                      'bc_pass': check['bc_geometry_pass'],
                      'shape_pass': check['shape_gate_pass'],
                      'components': check['mesh_components']}, indent=2))


if __name__ == '__main__':
    main()
