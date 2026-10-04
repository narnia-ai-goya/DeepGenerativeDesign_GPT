#!/usr/bin/env python3
"""Single-view Direct3D-S2 dense+sparse without engineering guidance."""
from __future__ import annotations

import argparse
import json
import subprocess

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SOURCE = BASE / 'open_arm'
REF = BASE / 'vanilla_direct3ds2_2026-10-02/front/config_dense.json'
OUT = BASE / 'minimal_direct3ds2_2026-10-02'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(REF.read_text())
    cfg['name'] = 'chair_open_arm_minimal_direct3ds2_dense_sparse'
    cfg['stages']['mesh'].update({
        'skip_sparse': False, 'sparse_steps': 30, 'sp_cfg': 7.0,
        'sp_guide_w': 0.0, 'sp_guide_w_peak': 0.0, 'sp_thick_w': 0.0,
        'sp_fea_w': 0.0, 'sp_image_proj_w': 0.0,
        'sp_bc_w': 0.0, 'sp_bc_buffer_w': 0.0, 'sp_out_w': 0.0,
        'sp_design_w': 0.0, 'sp_pool_anchor_w': 0.0,
        'sp_support_halo_vox': 0,
        'force_bc_solid': False, 'force_envelope_clip': False,
        'restrict_active_to_envelope': False,
        'save_dense_cache': str(OUT / 'dense_cache.npz'),
    })
    config = OUT / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(config),
               '--target-dir', str(SOURCE / 'input_lr162'), '--out', str(OUT / 'generation')]
    env = generation_env(args.gpu)
    env['VANILLA'] = '1'
    with (OUT / 'generation.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    (OUT / 'run.json').write_text(json.dumps({'command': command, 'env': {'VANILLA': '1'},
                                               'exit_code': result.returncode}, indent=2) + '\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    print(OUT / 'generation/mesh.obj')


if __name__ == '__main__':
    main()
