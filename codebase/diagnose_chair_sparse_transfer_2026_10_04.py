"""Isolate Sparse FEA transfer through the refiner and SDF smoothing."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

from run_chair_round3_fea_weight_sweep_2026_10_04 import OUT as SWEEP, SOURCE
from run_chair_sparse_fea_loop_2026_10_03 import command
from run_connectivity_qd_sampling import generation_env
from make_chair_domain import ROOT


OUT = SWEEP / 'smoothing_transfer_diagnostic'


def one(label: str, gpu: int) -> dict:
    case = OUT / label
    case.mkdir(parents=True, exist_ok=True)
    previous = SWEEP / 'round_01' / label / 'config.json'
    cfg = json.loads(previous.read_text())
    cfg['name'] = f'chair_sparse_transfer_{label}_sigma0'
    cfg['stages']['mesh']['sp_sdf_smooth_sigma'] = 0.0
    cfg['stages']['mesh']['sp_sdf_smooth_volume_match'] = False
    config = case / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    result = case / 'run.json'
    output = case / 'generation'
    if result.exists() and (output / 'mesh_pre_refiner.obj').exists():
        saved = json.loads(result.read_text())
        if saved['exit_code'] == 0:
            print(label, 'reuse', flush=True)
            return saved
    env = generation_env(gpu)
    env.update({'VANILLA': '1', 'FEA_LOAD_MAGNITUDE': '800',
                'FEA_MAX_NODE_MAP_DISTANCE': '0.04',
                'BC_SURFACE_DIST': '1', 'BC_DIST': '0.025',
                'D3DS2_SAVE_PRE_REFINER': '1'})
    log = case / 'generation.log'
    with log.open('w') as f:
        process = subprocess.run(command(config, SOURCE / 'input_lr162', output),
                                 cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    saved = {'label': label, 'gpu': gpu, 'exit_code': process.returncode,
             'config': str(config), 'log': str(log),
             'pre_refiner_mesh': str(output / 'mesh_pre_refiner.obj'),
             'final_mesh': str(output / 'mesh.obj')}
    result.write_text(json.dumps(saved, indent=2) + '\n')
    print(label, 'exit', process.returncode, flush=True)
    return saved


def main() -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda args: one(*args), [('off', 6), ('w10000', 7)]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'runs.json').write_text(json.dumps(rows, indent=2) + '\n')
    if any(row['exit_code'] for row in rows):
        raise SystemExit('diagnostic generation failed')


if __name__ == '__main__':
    main()
