#!/usr/bin/env python3
"""Small dense-stage eta × support-weight search for the aesthetic chair."""
from __future__ import annotations

import argparse
import json
import subprocess

from run_connectivity_qd_sampling import GENERATOR, PYTHON, ROOT, generation_env


BASE = ROOT / 'experiments/chair/aesthetic_reference_2026-09-27'
GRID = BASE / 'eta_pw_grid'
SETTINGS = [(eta, pw) for eta in (150., 225., 300.)
            for pw in (1.5, 2.5, 3.5)
            if (eta, pw) != (300., 3.5)]


def case_dir(eta: float, pw: float):
    return GRID / f'eta{eta:g}_pw{pw:g}'


def prepare() -> None:
    GRID.mkdir(parents=True, exist_ok=True)
    source = json.loads((BASE / 'archpath_cfg9/config_dense.json').read_text())
    records = []
    for eta, pw in SETTINGS:
        case = case_dir(eta, pw)
        case.mkdir(exist_ok=True)
        cfg = json.loads(json.dumps(source))
        cfg['name'] = f'chair_aesthetic_eta{eta:g}_pw{pw:g}'
        cfg['stages']['mesh'].update(eta=eta, pw=pw,
                                     save_dense_cache=str(case / 'dense_cache.npz'))
        config = case / 'config_dense.json'
        config.write_text(json.dumps(cfg, indent=2) + '\n')
        records.append({'eta': eta, 'pw': pw, 'case': str(case), 'config': str(config)})
    (GRID / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    print(GRID / 'manifest.json')


def run(eta: float, pw: float, gpu: int) -> None:
    case = case_dir(eta, pw)
    config = case / 'config_dense.json'
    if not config.exists():
        raise FileNotFoundError(config)
    cmd = [str(PYTHON), str(GENERATOR), '--config', str(config),
           '--target-dir', str(BASE / 'input'), '--out', str(case / 'dense')]
    with (case / 'dense.log').open('w') as log:
        result = subprocess.run(cmd, cwd=ROOT, env=generation_env(gpu),
                                stdout=log, stderr=subprocess.STDOUT)
    (case / 'run.json').write_text(json.dumps({'eta': eta, 'pw': pw,
        'gpu': gpu, 'exit_code': result.returncode}, indent=2) + '\n')
    if result.returncode:
        raise SystemExit(f'{case}: exit {result.returncode}')
    print(case / 'dense/mesh_dense.obj', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    one = sub.add_parser('run')
    one.add_argument('--eta', type=float, required=True)
    one.add_argument('--pw', type=float, required=True)
    one.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    else:
        run(args.eta, args.pw, args.gpu)


if __name__ == '__main__':
    main()
