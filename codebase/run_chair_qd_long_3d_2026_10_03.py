"""Run the frozen eight-image chair comparison with identical Direct3D-S2 settings."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
from PIL import Image

from chair_qd_long_protocol_2026_10_03 import BASE, OUT, CASES
from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env
from build_chair_image_qd_2026_10_03 import normalized, metrics
from prepare_chair_text_reasoned_pilot_2026_10_03 import front_aperture

TEMPLATE = BASE / 'single_view_qd_2026-10-03/image_angular_seed42/config.json'
GATE_REF = BASE / 'single_view_image_qd_2026-10-03/images/baseline.png'


def run(case: dict, gpu: int) -> dict:
    name = case['id']
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    source = OUT / f'{name}.png'
    target = folder / 'input_lr162/v00_front_lo.png'
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        im.convert('RGB').resize((162, 162), Image.Resampling.LANCZOS).save(target)
    config = json.loads(TEMPLATE.read_text())
    config['name'] = 'chair_qd_long_' + name
    config['stages']['mesh']['save_dense_cache'] = str(folder / 'dense_cache.npz')
    config['stages']['mesh']['load_dense_cache'] = None
    cfg = folder / 'config.json'
    cfg.write_text(json.dumps(config, indent=2) + '\n')
    command = [str(PYTHON), str(GENERATOR), '--config', str(cfg),
               '--target-dir', str(target.parent), '--out', str(folder / 'generation')]
    env = generation_env(gpu)
    env['VANILLA'] = '1'
    with (folder / 'generation.log').open('w') as log:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    result = {'id': name, 'gpu': gpu, 'source_image': str(source),
              'lowres_input': str(target), 'config': str(cfg),
              'generation_mesh': str(folder / 'generation/mesh.obj'),
              'exit_code': proc.returncode, 'command': command}
    (folder / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
    print(name, 'generation exit', proc.returncode, flush=True)
    return result


def main() -> None:
    _, reference = normalized(GATE_REF)
    images = []
    for case in CASES:
        source = OUT / f"{case['id']}.png"
        _, mask = normalized(source)
        image_metrics = metrics(mask, reference)
        image_metrics['front_arm_aperture_fraction'] = front_aperture(mask)
        row = {'id': case['id'], 'method': 'targeted' if case['id'].startswith('qd_') else 'control',
               'image': str(source), 'metrics': image_metrics,
               'projected_interface_gate': image_metrics['projected_interface_retention'] >= .9,
               'projected_back_load_gate': image_metrics['projected_back_load_retention'] >= .8}
        images.append(row)
        print(case['id'], 'image BC', round(image_metrics['projected_interface_retention'], 3),
              round(image_metrics['projected_back_load_retention'], 3), flush=True)
    (OUT / 'image_metrics.json').write_text(json.dumps(images, indent=2) + '\n')
    results = []
    for start in (0, 4):
        batch = CASES[start:start + 4]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results.extend(pool.map(lambda pair: run(*pair), zip(batch, (0, 1, 2, 3))))
        (OUT / 'generation_results.json').write_text(json.dumps(results, indent=2) + '\n')
    if any(row['exit_code'] for row in results):
        raise SystemExit('At least one 3D generation failed; see generation_results.json')


if __name__ == '__main__':
    main()
