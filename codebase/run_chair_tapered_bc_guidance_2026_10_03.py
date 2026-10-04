#!/usr/bin/env python3
"""Tapered candidate BC-guidance ablation with unchanged images and seed."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28/tapered_examples_2026-10-03'
OUT = BASE / 'bc_guidance_2026-10-03'
CASES = (
    ('tapered_curved', 'curved_dense_bc5', 5.0, 0.0, 0),
    ('tapered_curved', 'curved_dense_bc5_sparse_bc2', 5.0, 2.0, 1),
    ('tapered_seed44', 'seed44_dense_bc5_sparse_bc2', 5.0, 2.0, 2),
    ('tapered_diagonal', 'diagonal_dense_bc10_sparse_bc2', 10.0, 2.0, 3),
)


def run(item: tuple[str, str, float, float, int]) -> dict:
    source_name, name, dense_bc, sparse_bc, gpu = item
    source = BASE / source_name
    case = OUT / name
    case.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / 'config.json').read_text())
    config['name'] = 'chair_tapered_' + name
    mesh = config['stages']['mesh']
    mesh.update({
        'bc_w': dense_bc, 'out_w': 1.0,
        'sp_guide_w': 1.0 if sparse_bc else 0.0,
        'sp_guide_w_peak': 1.0 if sparse_bc else 0.0,
        'sp_bc_w': sparse_bc,
        'sp_bc_buffer_w': sparse_bc / 2 if sparse_bc else 0.0,
        'sp_out_w': 0.5 if sparse_bc else 0.0,
        'sp_guide_lr': 0.001,
        'sp_n_inner_late': 2,
        'save_dense_cache': str(case / 'dense_cache.npz'),
        'load_dense_cache': None,
    })
    cp = case / 'config.json'
    cp.write_text(json.dumps(config, indent=2) + '\n')
    target = (BASE / source_name / 'input_lr162') if not source_name.startswith('tapered_seed') else (
        ROOT / 'experiments/chair/sofa_style_2026-09-28/text_latent_qd_2026-10-03/mesh_cases/tapered__straight__standard/input_lr162')
    command = [str(PYTHON), str(GENERATOR), '--config', str(cp),
               '--target-dir', str(target), '--out', str(case / 'generation')]
    env = generation_env(gpu)
    env['VANILLA'] = '1'
    with (case / 'generation.log').open('w') as stream:
        process = subprocess.run(command, cwd=ROOT, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
    row = {'name': name, 'source': source_name, 'dense_bc_w': dense_bc,
           'sparse_bc_w': sparse_bc, 'gpu': gpu, 'command': command,
           'exit_code': process.returncode, 'mesh': str(case / 'generation/mesh.obj')}
    (case / 'run.json').write_text(json.dumps(row, indent=2) + '\n')
    print(name, process.returncode, flush=True)
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(run, CASES))
    (OUT / 'runs.json').write_text(json.dumps(rows, indent=2) + '\n')
    if any(row['exit_code'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
