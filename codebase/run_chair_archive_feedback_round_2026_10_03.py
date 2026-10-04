#!/usr/bin/env python3
"""Generate the frozen archive-feedback chair pair with matched 3D settings."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'archive_feedback_round_02_2026-10-03'
TEMPLATE = BASE / 'text_latent_qd_2026-10-03/mesh_cases/faceted__curved__standard/config.json'


def run(item: tuple[dict, int]) -> dict:
    selected, gpu = item
    name = selected['id']
    case = OUT / name
    target = case / 'input_lr162/v00_front_lo.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(case / 'input.png') as im:
        im.convert('RGB').resize((162, 162), Image.Resampling.LANCZOS).save(target)
    config = json.loads(TEMPLATE.read_text())
    config['name'] = 'chair_archive_feedback_' + name
    config['stages']['mesh']['save_dense_cache'] = str(case / 'dense_cache.npz')
    config['stages']['mesh']['load_dense_cache'] = None
    cp = case / 'config.json'
    cp.write_text(json.dumps(config, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(cp),
               '--target-dir', str(target.parent), '--out', str(case / 'generation')]
    env = generation_env(gpu)
    env['VANILLA'] = '1'
    with (case / 'generation.log').open('w') as stream:
        process = subprocess.run(command, cwd=ROOT, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
    row = {'id': name, 'method': selected['method'], 'gpu': gpu,
           'command': command, 'exit_code': process.returncode,
           'image': str(case / 'input.png'),
           'mesh': str(case / 'generation/mesh.obj')}
    (case / 'run.json').write_text(json.dumps(row, indent=2) + '\n')
    print(name, process.returncode, flush=True)
    return row


def main() -> None:
    selected = json.loads((OUT / 'selection.json').read_text())['selected']
    assert len(selected) == 2
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(run, zip(selected, (4, 5))))
    (OUT / 'runs.json').write_text(json.dumps(rows, indent=2) + '\n')
    if any(row['exit_code'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--round-dir', type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.round_dir.resolve()
    main()
