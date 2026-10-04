#!/usr/bin/env python3
"""Matched, dense-only test of one-sided envelope guidance on existing chair images."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SOURCE = BASE / 'text_latent_qd_2026-10-03/mesh_cases'
OUT = BASE / 'envelope_excess_pilot_2026-10-03'
CASES = (
    ('tapered__straight__standard', 3.0, 1),
    ('faceted__curved__standard', 3.0, 2),
    ('faceted__curved__standard', 10.0, 3),
)


def run(item: tuple[str, float, int]) -> dict:
    name, eta, gpu = item
    source = SOURCE / name
    case = OUT / f'{name}__eta{eta:g}'
    case.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((source / 'config.json').read_text())
    cfg['name'] = f'chair_envelope_excess_{name}_eta{eta:g}'
    mesh = cfg['stages']['mesh']
    mesh.update({
        'skip_sparse': True,
        'load_dense_cache': None,
        'save_dense_cache': str(case / 'dense_cache.npz'),
        'eta': eta,
        'bc_w': 0.0, 'out_w': 0.0, 'dw': 0.0,
        'env_excess_w': 1.0, 'env_excess_tol': 0.002,
        'dense_token_policy': 'raw',
        'cw': 0.0, 'tw': 0.0, 'vw': 0.0, 'pw': 0.0,
        'fea_w': 0.0, 'image_proj_w': 0.0,
        'force_bc_solid': False, 'force_envelope_clip': False,
        'restrict_active_to_envelope': False,
    })
    config = case / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(config),
               '--target-dir', str(source / 'input_lr162'),
               '--out', str(case / 'generation')]
    env = generation_env(gpu)
    env['VANILLA'] = '0'
    with (case / 'generation.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    record = {'name': name, 'eta': eta, 'gpu': gpu, 'exit_code': result.returncode,
              'source': str(source), 'output': str(case), 'command': command,
              'environment': {'VANILLA': '0'}}
    (case / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    print(name, eta, 'exit', result.returncode, flush=True)
    return record


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(CASES)) as pool:
        records = list(pool.map(run, CASES))
    (OUT / 'runs.json').write_text(json.dumps(records, indent=2) + '\n')
    if any(row['exit_code'] for row in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
