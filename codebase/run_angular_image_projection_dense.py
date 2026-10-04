#!/usr/bin/env python3
"""One FEA-on, dense-only image-projection guidance study case."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from run_connectivity_qd_sampling import GENERATOR, PYTHON, ROOT, generation_env


EXP = ROOT / 'experiments/bracket/angular_image_qd_loop_2026-09-24'
OUT = EXP / 'image_projection_guidance_2026-09-24'
SOURCES = {
    'triangular_truss': EXP / 'round_00/triangular_truss',
    'staggered_chevron': EXP / 'round_01/staggered_chevron',
}


def run(name: str, weight: float, gpu: int) -> dict:
    source = SOURCES[name]
    case = OUT / name / f'w{weight:g}'
    case.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((source / 'config_dense_top.json').read_text())
    cfg['name'] = f'angular_imgproj_{name}_w{weight:g}'
    mesh = cfg['stages']['mesh']
    mesh.update({'image_proj_target': str(OUT / f'{name}_target.npz'),
                 'image_proj_w': weight, 'image_proj_warmup': 0.35,
                 'load_dense_cache': None,
                 'save_dense_cache': str(case / 'dense_cache.npz'),
                 'skip_sparse': True})
    config = case / 'config.json'
    config.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(config),
               '--target-dir', str(source / 'input'), '--out', str(case / 'generation')]
    env = generation_env(gpu)
    env['FEA_MAX_NODE_MAP_DISTANCE'] = '0.02'
    with (case / 'generation.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    row = {'source': str(source), 'case': str(case), 'weight': weight,
           'gpu': gpu, 'exit_code': result.returncode,
           'config': str(config), 'dense_cache': str(case / 'dense_cache.npz'),
           'dense_mesh': str(case / 'generation/mesh_dense.obj')}
    (case / 'run.json').write_text(json.dumps(row, indent=2) + '\n')
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=SOURCES, required=True)
    parser.add_argument('--weight', type=float, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.case, args.weight, args.gpu), indent=2), flush=True)


if __name__ == '__main__':
    main()
