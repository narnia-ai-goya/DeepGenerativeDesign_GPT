#!/usr/bin/env python3
"""Run matched tapered image controls with changed Direct3D-S2 seeds."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SOURCE = BASE / 'text_latent_qd_2026-10-03/mesh_cases/tapered__straight__standard'
OUT = BASE / 'tapered_examples_2026-10-03'


def run(seed: int, gpu: int) -> dict:
    name = f'tapered_seed{seed}'
    case = OUT / name
    case.mkdir(exist_ok=True)
    config = json.loads((SOURCE / 'config.json').read_text())
    config['name'] = name
    config['seed'] = seed
    config['stages']['mesh'].update({
        'load_dense_cache': None,
        'save_dense_cache': str(case / 'dense_cache.npz'),
        'sp_sdf_smooth_sigma': 3.0,
        'sp_sdf_smooth_volume_match': True,
    })
    config_path = case / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(config_path),
               '--target-dir', str(SOURCE / 'input_lr162'),
               '--out', str(case / 'generation')]
    env = generation_env(gpu)
    env['VANILLA'] = '1'
    with (case / 'generation.log').open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    row = {'name': name, 'seed': seed, 'gpu': gpu,
           'input': str(BASE / 'text_latent_qd_2026-10-03/images/tapered__straight__standard.png'),
           'mesh': str(case / 'generation/mesh.obj'),
           'exit_code': result.returncode, 'command': command}
    (case / 'run.json').write_text(json.dumps(row, indent=2) + '\n')
    print(name, result.returncode, flush=True)
    return row


def main() -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda item: run(*item), ((43, 0), (44, 1))))
    (OUT / 'seed_control_runs.json').write_text(json.dumps(rows, indent=2) + '\n')
    if any(row['exit_code'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
