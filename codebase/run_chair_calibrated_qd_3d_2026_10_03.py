"""Run frozen image-selected chair candidates with identical Direct3D-S2 settings."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess

from PIL import Image

from chair_qd_long_protocol_2026_10_03 import BASE
from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env


OUT = BASE / 'text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03'
TEMPLATE = BASE / 'single_view_qd_2026-10-03/image_angular_seed42/config.json'


def run(row: dict, folder, gpu: int) -> dict:
    name = row['id']
    case = folder / name
    case.mkdir(parents=True, exist_ok=True)
    target = case / 'input_lr162/v00_front_lo.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(row['image']) as im:
        im.convert('RGB').resize((162, 162), Image.Resampling.LANCZOS).save(target)
    config = json.loads(TEMPLATE.read_text())
    config['name'] = f'chair_calibrated_qd_{folder.name}_{name}'
    config['stages']['mesh']['save_dense_cache'] = str(case / 'dense_cache.npz')
    config['stages']['mesh']['load_dense_cache'] = None
    cfg = case / 'config.json'
    cfg.write_text(json.dumps(config, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(cfg),
               '--target-dir', str(target.parent), '--out', str(case / 'generation')]
    env = generation_env(gpu)
    env['VANILLA'] = '1'
    with (case / 'generation.log').open('w') as log:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    result = {'id': name, 'method': row['method'], 'gpu': gpu, 'source_image': row['image'],
              'lowres_input': str(target), 'config': str(cfg),
              'mesh': str(case / 'generation/mesh.obj'), 'exit_code': proc.returncode,
              'command': command}
    (case / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
    print(name, 'generation exit', proc.returncode, flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    folder = OUT / f'round_{args.round:02d}'
    selection = json.loads((folder / 'selection.json').read_text())
    gpus = (6, 7, 6, 7)
    rows = []
    for start in (0, 2):
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows.extend(pool.map(lambda pair: run(pair[0], folder, pair[1]),
                                 zip(selection['selected'][start:start + 2],
                                     gpus[start:start + 2])))
    (folder / 'generation_results.json').write_text(json.dumps(rows, indent=2) + '\n')
    if any(row['exit_code'] != 0 for row in rows):
        raise SystemExit('At least one generation failed; see per-case log')


if __name__ == '__main__':
    main()
