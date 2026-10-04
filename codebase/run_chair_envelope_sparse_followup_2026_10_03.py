#!/usr/bin/env python3
"""Run unchanged sparse refinement from the envelope-guided dense cache."""
from __future__ import annotations

import json
import subprocess

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
CASE = BASE / 'envelope_excess_pilot_2026-10-03/faceted__curved__standard__eta10'
SOURCE = BASE / 'text_latent_qd_2026-10-03/mesh_cases/faceted__curved__standard'


def main() -> None:
    cfg = json.loads((CASE / 'config.json').read_text())
    cfg['name'] += '_sparse'
    cfg['stages']['mesh'].update({
        'skip_sparse': False,
        'load_dense_cache': str(CASE / 'dense_cache.npz'),
        'save_dense_cache': None,
        'sp_guide_w': 0.0,
        'sp_out_w': 0.0,
        'force_envelope_clip': False,
        'restrict_active_to_envelope': False,
    })
    config = CASE / 'config_sparse.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(config),
               '--target-dir', str(SOURCE / 'input_lr162'),
               '--out', str(CASE / 'sparse_generation')]
    env = generation_env(3)
    env['VANILLA'] = '0'
    with (CASE / 'sparse_generation.log').open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    (CASE / 'sparse_run.json').write_text(json.dumps({
        'command': command, 'exit_code': result.returncode,
        'mesh': str(CASE / 'sparse_generation/mesh.obj'),
    }, indent=2) + '\n')
    print('sparse exit', result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
