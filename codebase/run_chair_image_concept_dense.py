#!/usr/bin/env python3
"""Run the same chair dense recipe on three image-generated concepts."""
from __future__ import annotations

import argparse
import json
import subprocess

from run_connectivity_qd_sampling import GENERATOR, PYTHON, ROOT, generation_env


BASE = ROOT / 'experiments/chair/image_concepts_2026-09-27'
SOURCE = ROOT / 'experiments/chair/aesthetic_reference_2026-09-27/archpath_cfg9/config_dense.json'
NAMES = ('arched_ladder', 'triangular_back', 'diagonal_braced')


def prepare() -> None:
    source = json.loads(SOURCE.read_text())
    for name in NAMES:
        case = BASE / name
        cfg = json.loads(json.dumps(source))
        cfg['name'] = f'chair_image_concept_{name}_lr162_dense'
        cfg['views'] = ''
        cfg['stages']['mesh'].update(
            n_views=1, views='', skip_sparse=True,
            save_dense_cache=str(case / 'dense_cache.npz'),
            load_dense_cache=None)
        (case / 'config_dense.json').write_text(json.dumps(cfg, indent=2) + '\n')
    print(BASE)


def run(name: str, gpu: int) -> None:
    if name not in NAMES:
        raise ValueError(name)
    case = BASE / name
    config = case / 'config_dense.json'
    if not config.exists():
        raise FileNotFoundError(config)
    command = [str(PYTHON), str(GENERATOR), '--config', str(config),
               '--image', str(case / 'input_162.png'), '--out', str(case / 'dense')]
    with (case / 'dense.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=generation_env(gpu),
                                stdout=log, stderr=subprocess.STDOUT)
    row = {'name': name, 'image': str(case / 'input_162.png'),
           'config': str(config), 'gpu': gpu, 'exit_code': result.returncode,
           'dense_mesh': str(case / 'dense/mesh_dense.obj'),
           'dense_cache': str(case / 'dense_cache.npz')}
    (case / 'run.json').write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row, indent=2), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare')
    one = commands.add_parser('run')
    one.add_argument('name', choices=NAMES)
    one.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    else:
        run(args.name, args.gpu)


if __name__ == '__main__':
    main()
