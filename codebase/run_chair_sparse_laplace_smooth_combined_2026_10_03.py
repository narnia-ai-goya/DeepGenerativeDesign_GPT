#!/usr/bin/env python3
"""Validate gentle in-loop Laplacian plus volume-matched SDF regularization."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SOURCE = BASE / 'text_latent_qd_2026-10-03/mesh_cases'
OUT = BASE / 'sparse_wrinkle_pilot_2026-10-03'
CASES = (('faceted__curved__standard', 'combined_faceted', 1),
         ('tapered__straight__standard', 'combined_tapered', 2))


def run(item: tuple[str, str, int]) -> dict:
    source_name, name, gpu = item
    source = SOURCE / source_name
    case = OUT / name
    case.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((source / 'config.json').read_text())
    cfg['name'] = 'chair_sparse_laplace_smooth_' + source_name
    cfg['stages']['mesh'].update({
        'skip_sparse': False,
        'load_dense_cache': str(source / 'dense_cache.npz'),
        'save_dense_cache': None,
        'sp_guide_w': 1.0, 'sp_guide_w_peak': 1.0,
        'sp_opt': 'adamw', 'sp_guide_lr': .0003,
        'sp_lap_w': 1.0, 'sp_lap_sigma': .3,
        'sp_n_inner_late': 1, 'sp_param_pool_schedule': '',
        'sp_sdf_inside_low': True, 'sp_sdf_guidance_threshold': .6,
        'sp_rmin_w': 0.0, 'sp_thick_w': 0.0, 'sp_hole_w': 0.0,
        'sp_interior_w': 0.0, 'sp_normal_fd_w': 0.0,
        'sp_bc_w': 0.0, 'sp_out_w': 0.0, 'sp_design_w': 0.0,
        'sp_fea_w': 0.0, 'fea_w': 0.0,
        'sp_sdf_smooth_sigma': 3.0,
        'sp_sdf_smooth_volume_match': True,
        'force_envelope_clip': False,
        'restrict_active_to_envelope': False,
    })
    config = case / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(config),
               '--target-dir', str(source / 'input_lr162'),
               '--out', str(case / 'generation')]
    env = generation_env(gpu)
    env['VANILLA'] = '1'
    with (case / 'generation.log').open('w') as stream:
        process = subprocess.run(command, cwd=ROOT, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
    record = {'source': source_name, 'name': name, 'gpu': gpu,
              'exit_code': process.returncode, 'mesh': str(case / 'generation/mesh.obj'),
              'command': command}
    (case / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    print(name, 'exit', process.returncode, flush=True)
    return record


def main() -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(run, CASES))
    (OUT / 'combined_runs.json').write_text(json.dumps(rows, indent=2) + '\n')
    if any(row['exit_code'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
