#!/usr/bin/env python3
"""Freeze image prompts, then ingest built-in image outputs and check 2D gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import cv2
from PIL import Image

from make_chair_domain import ROOT
from build_chair_image_qd_2026_10_03 import normalized, metrics, image_cell

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
LOOP = BASE / 'archive_feedback_loop_2026-10-03'
REFERENCE = BASE / 'text_latent_qd_2026-10-03/images/faceted__curved__standard.png'
IMAGE_GATE_REFERENCE = BASE / 'single_view_image_qd_2026-10-03/images/baseline.png'
TEMPLATE = (
    'Edit this exact single black-metal chair front-view studio image for a controlled '
    'structural design experiment. Preserve the same square canvas, pure white background, '
    'camera, full chair scale and position, four floor-contact pads, four leg positions, '
    'seat width and height, black metallic material, solid central backrest load-bearing '
    'region, and open gap between backrest and seat. Change only the upper backrest and '
    'arm geometry. Shape change: {shape_text} Ensure the seat surface and central backrest '
    'remain visibly broad and fully solid; no holes, no detached pieces, no extra legs. '
    'Photorealistic shaded metal, clear silhouette. Single front orthographic view only, '
    'no collage, no text.'
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('round_index', type=int)
    parser.add_argument('--qd-image', type=Path)
    parser.add_argument('--random-image', type=Path)
    args = parser.parse_args()
    folder = LOOP / f'round_{args.round_index:02d}'
    selected = json.loads((folder / 'selection.json').read_text())['selected']
    prompts = [{'method': r['method'], 'id': r['id'],
                'reference_image': str(REFERENCE),
                'prompt': TEMPLATE.format(shape_text=r['shape_text']),
                'tool': 'built-in imagegen', 'model_id': 'not exposed by built-in tool'}
               for r in selected]
    (folder / 'image_prompts.json').write_text(json.dumps(prompts, indent=2) + '\n')
    if args.qd_image is None or args.random_image is None:
        for r in prompts:
            print(r['method'], r['id'], '\n', r['prompt'], '\n')
        return
    paths = {'archive_qd': args.qd_image, 'uniform_random': args.random_image}
    _, reference = normalized(IMAGE_GATE_REFERENCE)
    results = []
    for r in prompts:
        path = paths[r['method']]
        assert path.is_file(), path
        case = folder / r['id']
        case.mkdir(exist_ok=True)
        output = case / 'input.png'
        shutil.copy2(path, output)
        norm, mask = normalized(output)
        cv2.imwrite(str(case / 'input_normalized.png'), norm)
        measure = metrics(mask, reference)
        cell = image_cell(measure)
        gate = bool(cell is not None and measure['projected_interface_retention'] >= .9 and
                    measure['projected_back_load_retention'] >= .8)
        results.append({'id': r['id'], 'method': r['method'], 'image': str(output),
                        'original_generated_file': str(path), 'metrics': measure,
                        'image_cell': cell, 'image_gate': gate})
        print(r['id'], 'image_gate', gate, 'cell', cell)
    (folder / 'image_gate.json').write_text(json.dumps(results, indent=2) + '\n')
    (folder / 'image_generation.json').write_text(json.dumps(
        [{**p, 'original_generated_file': str(paths[p['method']]),
          'project_copy': str(folder / p['id'] / 'input.png')} for p in prompts], indent=2) + '\n')


if __name__ == '__main__':
    main()
