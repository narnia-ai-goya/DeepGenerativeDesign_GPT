#!/usr/bin/env python3
"""Generate tapered chair examples with matched Direct3D-S2 settings."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SOURCE = BASE / 'text_latent_qd_2026-10-03/mesh_cases/tapered__straight__standard'
OUT = BASE / 'tapered_examples_2026-10-03'
IMAGES = (
    ('tapered_curved', '/home/goya/.codex/generated_images/01a098b3-02c6-72d0-9729-ce061f818654/exec-ee9877cc-2a60-437c-8452-391e3feab888.png'),
    ('tapered_diagonal', '/home/goya/.codex/generated_images/01a098b3-02c6-72d0-9729-ce061f818654/exec-3d1dc3a8-44b0-4b08-87cb-8eaefae59c8a.png'),
    ('tapered_tall', '/home/goya/.codex/generated_images/01a098b3-02c6-72d0-9729-ce061f818654/exec-752f7293-5253-48e3-a830-5155f4655720.png'),
)


def run(item: tuple[int, tuple[str, str]]) -> dict:
    gpu, (name, image_path) = item
    case = OUT / name
    case.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, case / 'input.png')
    target = case / 'input_lr162/v00_front_lo.png'
    target.parent.mkdir(exist_ok=True)
    with Image.open(case / 'input.png') as image:
        image.convert('RGB').resize((162, 162), Image.Resampling.LANCZOS).save(target)
    config = json.loads((SOURCE / 'config.json').read_text())
    config['name'] = 'chair_' + name + '_sdf_smooth3'
    config['stages']['mesh'].update({
        'save_dense_cache': str(case / 'dense_cache.npz'),
        'load_dense_cache': None,
        'sp_sdf_smooth_sigma': 3.0,
        'sp_sdf_smooth_volume_match': True,
    })
    config_path = case / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(config_path),
               '--target-dir', str(target.parent), '--out', str(case / 'generation')]
    env = generation_env(gpu)
    env['VANILLA'] = '1'
    with (case / 'generation.log').open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    row = {'name': name, 'gpu': gpu, 'input': str(case / 'input.png'),
           'lowres_input': str(target), 'mesh': str(case / 'generation/mesh.obj'),
           'exit_code': result.returncode, 'command': command}
    (case / 'run.json').write_text(json.dumps(row, indent=2) + '\n')
    print(name, result.returncode, flush=True)
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(run, zip((0, 1, 2), IMAGES)))
    (OUT / 'runs.json').write_text(json.dumps(rows, indent=2) + '\n')
    if any(row['exit_code'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
