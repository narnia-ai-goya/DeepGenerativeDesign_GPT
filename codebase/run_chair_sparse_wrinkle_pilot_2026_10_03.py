#!/usr/bin/env python3
"""Sparse-only chair surface study from the same unmodified dense tokens."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SOURCE = BASE / 'text_latent_qd_2026-10-03/mesh_cases/faceted__curved__standard'
OUT = BASE / 'sparse_wrinkle_pilot_2026-10-03'
CASES = (
    ('cfg4', 1, {'sp_cfg': 4.0}),
    ('steps50', 2, {'sparse_steps': 50}),
    ('cfg4_steps50', 3, {'sp_cfg': 4.0, 'sparse_steps': 50}),
    ('iso040', 4, {'mc_threshold': 0.4}),
    ('smooth3_match', 5, {'sp_sdf_smooth_sigma': 3.0,
                          'sp_sdf_smooth_volume_match': True}),
)


def run(case_info: tuple[str, int, dict]) -> dict:
    name, gpu, changes = case_info
    case = OUT / name
    case.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SOURCE / 'config.json').read_text())
    cfg['name'] = 'chair_sparse_wrinkle_' + name
    cfg['stages']['mesh'].update({
        'skip_sparse': False,
        'load_dense_cache': str(SOURCE / 'dense_cache.npz'),
        'save_dense_cache': None,
        'sp_guide_w': 0.0, 'sp_guide_w_peak': 0.0,
        'sp_lap_w': 0.0, 'sp_pool_anchor_w': 0.0,
        'force_envelope_clip': False,
        'restrict_active_to_envelope': False,
        **changes,
    })
    config = case / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(config),
               '--target-dir', str(SOURCE / 'input_lr162'),
               '--out', str(case / 'generation')]
    env = generation_env(gpu)
    env['VANILLA'] = '1'
    with (case / 'generation.log').open('w') as stream:
        process = subprocess.run(command, cwd=ROOT, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
    record = {'name': name, 'changes': changes, 'gpu': gpu,
              'exit_code': process.returncode, 'mesh': str(case / 'generation/mesh.obj'),
              'command': command}
    (case / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    print(name, 'exit', process.returncode, flush=True)
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cases', nargs='*', choices=[row[0] for row in CASES])
    args = ap.parse_args()
    selected = [row for row in CASES if not args.cases or row[0] in args.cases]
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        rows = list(pool.map(run, selected))
    (OUT / 'runs.json').write_text(json.dumps(rows, indent=2) + '\n')
    if any(row['exit_code'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
