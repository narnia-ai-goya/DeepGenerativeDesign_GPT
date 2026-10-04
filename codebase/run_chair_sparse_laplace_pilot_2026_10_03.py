#!/usr/bin/env python3
"""Sparse Laplace guidance ablation on fixed chair dense tokens."""
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
CASES = {
    'cfg7_lap0003': (1, 7.0, .0003),
    'cfg7_lap0010': (2, 7.0, .0010),
    'cfg4_lap0003': (3, 4.0, .0003),
    'cfg4_lap0010': (4, 4.0, .0010),
    'cfg4_lap0050': (1, 4.0, .0050),
    'cfg4_lap0100': (2, 4.0, .0100),
}


def run(name: str) -> dict:
    gpu, cfg_strength, lr = CASES[name]
    case = OUT / name
    case.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SOURCE / 'config.json').read_text())
    cfg['name'] = 'chair_sparse_laplace_' + name
    cfg['stages']['mesh'].update({
        'skip_sparse': False,
        'load_dense_cache': str(SOURCE / 'dense_cache.npz'),
        'save_dense_cache': None,
        'sp_cfg': cfg_strength,
        'sp_guide_w': 1.0, 'sp_guide_w_peak': 1.0,
        'sp_opt': 'adamw', 'sp_guide_lr': lr,
        'sp_lap_w': 1.0, 'sp_lap_sigma': .3,
        'sp_n_inner_late': 1,
        'sp_param_pool_schedule': '', 'sp_pool_anchor_w': 0.0,
        'sp_sdf_inside_low': True, 'sp_sdf_guidance_threshold': .6,
        'sp_rmin_w': 0.0, 'sp_thick_w': 0.0, 'sp_hole_w': 0.0,
        'sp_interior_w': 0.0, 'sp_normal_fd_w': 0.0,
        'sp_bc_w': 0.0, 'sp_out_w': 0.0, 'sp_design_w': 0.0,
        'sp_fea_w': 0.0, 'fea_w': 0.0,
        'force_envelope_clip': False,
        'restrict_active_to_envelope': False,
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
    result = {'name': name, 'gpu': gpu, 'sp_cfg': cfg_strength,
              'sp_guide_lr': lr, 'exit_code': process.returncode,
              'mesh': str(case / 'generation/mesh.obj'), 'command': command}
    (case / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
    print(name, 'exit', process.returncode, flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cases', nargs='*', choices=CASES, default=list(CASES))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(args.cases)) as pool:
        rows = list(pool.map(run, args.cases))
    (OUT / 'laplace_runs.json').write_text(json.dumps(rows, indent=2) + '\n')
    if any(row['exit_code'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
