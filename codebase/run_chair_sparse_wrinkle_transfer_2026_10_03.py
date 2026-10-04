#!/usr/bin/env python3
"""Validate the selected SDF regularization on a second chair image."""
from __future__ import annotations

import json
import subprocess

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SOURCE = BASE / 'text_latent_qd_2026-10-03/mesh_cases/tapered__straight__standard'
CASE = BASE / 'sparse_wrinkle_pilot_2026-10-03/tapered_smooth3_match'


def main() -> None:
    CASE.mkdir(parents=True, exist_ok=True)
    config = json.loads((SOURCE / 'config.json').read_text())
    config['name'] = 'chair_sparse_wrinkle_tapered_smooth3_match'
    config['stages']['mesh'].update({
        'load_dense_cache': str(SOURCE / 'dense_cache.npz'),
        'save_dense_cache': None,
        'skip_sparse': False,
        'sp_guide_w': 0.0, 'sp_guide_w_peak': 0.0,
        'sp_sdf_smooth_sigma': 3.0,
        'sp_sdf_smooth_volume_match': True,
        'force_envelope_clip': False,
        'restrict_active_to_envelope': False,
    })
    cp = CASE / 'config.json'
    cp.write_text(json.dumps(config, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(cp),
               '--target-dir', str(SOURCE / 'input_lr162'),
               '--out', str(CASE / 'generation')]
    env = generation_env(4)
    env['VANILLA'] = '1'
    with (CASE / 'generation.log').open('w') as stream:
        process = subprocess.run(command, cwd=ROOT, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
    (CASE / 'run.json').write_text(json.dumps({'command': command,
        'exit_code': process.returncode}, indent=2) + '\n')
    print(process.returncode)
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
