"""Actually activate dense thickness guidance (VANILLA=0) for chair ablation."""
from __future__ import annotations

import json
import os
import subprocess
import sys

from chair_qd_long_protocol_2026_10_03 import OUT as MAIN
from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

OUT = MAIN / 'dense_thickness_guided_2026-10-03'
SOURCE = MAIN / 'qd_01'


def run(name: str, weight: float, gpu: int, mode: str = 'hard') -> dict:
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    config = json.loads((SOURCE / 'config.json').read_text())
    config['name'] = 'chair_dense_guided_' + name
    mesh = config['stages']['mesh']
    mesh['tw_hard'] = weight if mode == 'hard' else 0.0
    mesh['tw_soft'] = 0.0
    mesh['tw_rmin'] = weight if mode == 'rmin' else 0.0
    if mode == 'rmin':
        mesh['rmin_radius'] = 1
        mesh['rmin_thresh'] = .45
        mesh['rmin_normalize'] = True
    mesh['load_dense_cache'] = None
    mesh['save_dense_cache'] = str(folder / 'dense_cache.npz')
    cfg = folder / 'config.json'
    cfg.write_text(json.dumps(config, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(cfg),
               '--target-dir', str(SOURCE / 'input_lr162'), '--out', str(folder / 'generation')]
    env = generation_env(gpu)
    env['VANILLA'] = '0'
    with (folder / 'generation.log').open('w') as log:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
    log = (folder / 'generation.log').read_text(errors='replace')
    if process.returncode == 0 and ('(vanilla)' in log or 'dense,' not in
                                    (folder / 'generation/loss_log.csv').read_text(errors='replace')):
        raise RuntimeError('Dense guidance did not activate; refusing to label this thickness-guided')
    row = {'id': name, 'mode': mode, 'weight': weight, 'gpu': gpu, 'VANILLA': '0',
           'exit_code': process.returncode, 'dense_guidance_logged': 'dense,' in
           (folder / 'generation/loss_log.csv').read_text(errors='replace') if
           (folder / 'generation/loss_log.csv').exists() else False,
           'config': str(cfg), 'source_image': str(MAIN / 'qd_01.png'),
           'dense_mesh': str(folder / 'generation/mesh_dense_raw.obj'),
           'sparse_mesh': str(folder / 'generation/mesh.obj')}
    (folder / 'run.json').write_text(json.dumps(row, indent=2) + '\n')
    print(name, 'exit', process.returncode, 'guidance logged', row['dense_guidance_logged'], flush=True)
    return row


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    name = sys.argv[1] if len(sys.argv) > 1 else 'hard_0p01'
    weight = float(sys.argv[2]) if len(sys.argv) > 2 else .01
    gpu = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    mode = sys.argv[4] if len(sys.argv) > 4 else 'hard'
    result = run(name, weight, gpu, mode)
    if result['exit_code']:
        raise SystemExit(result['exit_code'])
