#!/usr/bin/env python3
"""Continue a guided angular dense sample through sparse FEA and post validation."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from run_angular_image_projection_dense import EXP, OUT, SOURCES
from run_bracket_multiview_case_study import render_side
from run_connectivity_qd_sampling import GENERATOR, PYTHON, ROOT, generation_env


def run(name: str, weight: float, gpu: int) -> dict:
    source = SOURCES[name]
    dense = OUT / name / f'w{weight:g}'
    cache = dense / 'dense_cache.npz'
    if not cache.exists():
        raise FileNotFoundError(cache)
    case = dense / 'sparse_fea_on'
    input_dir = case / 'input'
    input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / 'input/그림1.png', input_dir / '그림1.png')
    mesh_path = dense / 'dense_cache_mesh.obj'
    for sign, label in ((-1, 'side_xminus'), (1, 'side_xplus')):
        render_side(mesh_path, input_dir / f'{label}.png', sign)
    cfg = json.loads((source / 'config_full.json').read_text())
    cfg['name'] = f'angular_imgproj_{name}_w{weight:g}_sparse_fea_on'
    cfg['views'] = '그림1,side_xminus,side_xplus'
    cfg['stages']['mesh'].update({'n_views': 3, 'load_dense_cache': str(cache),
                                  'save_dense_cache': None, 'skip_sparse': False})
    config = case / 'config_full.json'
    config.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + '\n')
    cmd = [str(PYTHON), str(GENERATOR), '--config', str(config),
           '--target-dir', str(input_dir), '--out', str(case / 'generation')]
    env = generation_env(gpu)
    env['FEA_MAX_NODE_MAP_DISTANCE'] = '0.02'
    with (case / 'generation.log').open('w') as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    row = {'case': str(case), 'source': str(source), 'dense_cache': str(cache),
           'weight': weight, 'gpu': gpu, 'exit_code': result.returncode,
           'sparse_mesh': str(case / 'generation/mesh.obj')}
    (case / 'run.json').write_text(json.dumps(row, indent=2) + '\n')
    if result.returncode:
        return row
    with (case / 'post_runner.log').open('w') as log:
        post = subprocess.run([str(PYTHON), str(ROOT / 'codebase/run_angular_qd_candidate_post.py'),
                               str(case)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    row['post_exit_code'] = post.returncode
    if (case / 'metrics.json').exists():
        row['metrics'] = json.loads((case / 'metrics.json').read_text())
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
