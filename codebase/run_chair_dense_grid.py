#!/usr/bin/env python3
"""Run one chair dense-grid cell (support-corridor weight × image CFG)."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from run_connectivity_qd_sampling import GENERATOR, PYTHON, ROOT, generation_env


BASE = ROOT / 'experiments/chair/pilot_2026-09-25'
GRID = BASE / 'dense_grid_pw_cfg'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pw', type=float, required=True)
    parser.add_argument('--cfg', type=float, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    case = GRID / f'pw{args.pw:g}_cfg{args.cfg:g}'
    case.mkdir(parents=True, exist_ok=True)
    source = BASE / f'support_pw{args.pw:g}/config_dense.json'
    cfg = json.loads(source.read_text())
    cfg['name'] = f'chair_grid_pw{args.pw:g}_cfg{args.cfg:g}'
    cfg['stages']['mesh'].update(cfg=args.cfg,
                                 save_dense_cache=str(case / 'dense_cache.npz'))
    config = case / 'config_dense.json'
    config.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + '\n')
    env = generation_env(args.gpu)
    command = [str(PYTHON), str(GENERATOR), '--config', str(config),
               '--target-dir', str(BASE / 'input'), '--out', str(case / 'dense')]
    with (case / 'dense.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
    row = {'case': str(case), 'pw': args.pw, 'cfg': args.cfg,
           'gpu': args.gpu, 'exit_code': result.returncode,
           'dense_mesh': str(case / 'dense/mesh_dense.obj'),
           'dense_cache': str(case / 'dense_cache.npz')}
    (case / 'run.json').write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps(row, indent=2), flush=True)


if __name__ == '__main__':
    main()
