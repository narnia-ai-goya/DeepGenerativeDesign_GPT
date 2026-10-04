#!/usr/bin/env python3
"""Direct3D-S2 dense backbone baseline on the existing open-arm chair image."""
from __future__ import annotations

import argparse
import json
import subprocess

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SOURCE = BASE / 'open_arm'
CONFIG = BASE / 'fresh_dense_no_prototype_2026-10-02/open_arm/image_strong/config_dense.json'
OUT = BASE / 'vanilla_direct3ds2_2026-10-02'
CASES = {'front': 'v00_front_lo', 'three_view': 'v00_front_lo,v02_right_lo,v_top'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--gpu', type=int, default=5)
    args = parser.parse_args()
    out = OUT / args.case
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text())
    cfg['name'] = 'chair_open_arm_direct3ds2_vanilla_' + args.case
    cfg['views'] = CASES[args.case]
    cfg['stages']['mesh'].update({
        'views': CASES[args.case], 'n_views': 1 if args.case == 'front' else 3,
        'cfg': 7.0, 'dense_steps': 50, 'skip_sparse': True,
        'load_dense_cache': None, 'save_dense_cache': str(out / 'dense_cache.npz'),
        'bc_w': 0.0, 'out_w': 0.0, 'dw': 0.0, 'vw': 0.0, 'pw': 0.0, 'cw': 0.0,
        'sw': 0.0, 'tw': 0.0, 'fea_w': 0.0, 'image_proj_w': 0.0,
        'shape_anchor_w': 0.0, 'shape_scaffold_w': 0.0, 'shape_qd_w': 0.0,
        'load_path_mask': None, 'shape_anchor_bank': None,
        'force_bc_solid': False, 'force_envelope_clip': False,
        'restrict_active_to_envelope': False, 'dense_keep_bc_components': False,
        'dense_keep_largest_mesh_component': False,
    })
    (out / 'config_dense.json').write_text(json.dumps(cfg, indent=2) + '\n')
    cmd = [str(PYTHON), str(GENERATOR), '--config', str(out / 'config_dense.json'),
           '--target-dir', str(SOURCE / 'input_lr162'), '--out', str(out / 'generation')]
    env = generation_env(args.gpu)
    env['VANILLA'] = '1'
    with (out / 'generation.log').open('w') as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (out / 'run.json').write_text(json.dumps({'command': cmd, 'env': {'VANILLA': '1'},
                                               'exit_code': result.returncode}, indent=2) + '\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    print(out / 'generation/mesh_dense_raw.obj')


if __name__ == '__main__':
    main()
