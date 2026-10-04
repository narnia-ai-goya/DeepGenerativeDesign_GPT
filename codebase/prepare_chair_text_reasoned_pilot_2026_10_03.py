"""Record an AI-authored 2x2 prompt intervention and prepare matched 3D runs."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from make_chair_domain import ROOT
from build_chair_image_qd_2026_10_03 import normalized, metrics, image_cell

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'text_reasoned_front_axes_2026-10-03'
REF = BASE / 'text_latent_qd_2026-10-03/images/faceted__curved__standard.png'
GATE_REF = BASE / 'single_view_image_qd_2026-10-03/images/baseline.png'
GENERATED = Path('/home/goya/.codex/generated_images/01a098b3-02c6-72d0-9729-ce061f818654')
BASE_PROMPT = (
    'Edit the attached exact chair image as a single front orthographic studio render. '
    'Keep the entire seat, its front lip, all four legs, under-seat braces, four foot pads, '
    'camera, scale, black metal shading, and pure white background visually unchanged. '
    'Keep a broad solid central backrest where the seated person\'s back presses. '
    'Change only the backrest outline and the left/right arm-to-back openings. '
    'Make the requested design difference obvious in the FRONT silhouette. '
    'No extra legs, floating pieces, or text. '
)
SHAPES = [
    ('narrow_open', 'exec-7d04aca0-87cc-4f28-9d99-6c40d1142730.png',
     'The backrest is strongly tapered: its top is about 70 percent as wide as its bottom, with a flat narrow top and two clearly slanted sides. Each slender arm rail bends outward, leaving a large clearly visible triangular white opening between the arm rail and backrest support. Keep the structural backrest solid.'),
    ('wide_open', 'exec-65c6b78a-783a-48f3-9eee-f1935174269a.png',
     'The backrest is strongly flared: its top is about 120 percent as wide as its bottom but stays inside the original overall chair width, with a broad flat top and two clearly slanted sides. Each slender angular arm rail leaves a large clearly visible triangular white opening between the arm rail and backrest support. Keep the structural backrest solid.'),
    ('narrow_closed', 'exec-b59e43e6-9b16-404c-891f-c26954789dab.png',
     'The backrest is strongly tapered: its top is about 70 percent as wide as its bottom, with a flat narrow top and two clearly slanted sides. Add broad continuous metal arm gussets on both sides so the front-view white openings between arm and backrest are small but still visibly open. Keep the structural backrest solid and the seat unchanged.'),
    ('wide_closed', 'exec-d156455e-efed-421f-9858-e200fef228e7.png',
     'The backrest is strongly flared: its top is about 120 percent as wide as its bottom but stays inside the original overall chair width, with a broad flat top and two clearly slanted sides. Add broad continuous metal arm gussets on both sides so the front-view white openings between arm and backrest are small but still visibly open. Keep the structural backrest solid and the seat unchanged.'),
]


def front_aperture(mask: np.ndarray) -> float:
    y0, y1 = int(.23 * 512), int(.38 * 512)
    left = mask[y0:y1, int(.24 * 512):int(.37 * 512)]
    right = mask[y0:y1, int(.63 * 512):int(.76 * 512)]
    return float(1 - (left.mean() + right.mean()) / 2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _, reference = normalized(GATE_REF)
    records = []
    for index, (name, file, clause) in enumerate(SHAPES):
        pair = OUT / f'pair_{index // 2 + 1:02d}'
        case = pair / name
        case.mkdir(parents=True, exist_ok=True)
        input_path = case / 'input.png'
        source = GENERATED / file
        shutil.copy2(source, input_path)
        norm, mask = normalized(input_path)
        cv2.imwrite(str(case / 'input_normalized.png'), norm)
        values = metrics(mask, reference)
        values['front_arm_aperture_fraction'] = front_aperture(mask)
        cell = image_cell(values)
        gate = bool(values['projected_interface_retention'] >= .9 and
                    values['projected_back_load_retention'] >= .8)
        item = {'id': name, 'method': 'text_reasoned', 'back': name.split('_')[0],
                'arm': name.split('_')[1], 'shape_text': clause,
                'full_prompt': BASE_PROMPT + clause,
                'reference_image': str(REF), 'source_image': str(source),
                'image': str(input_path), 'metrics': values,
                'image_cell': cell, 'image_gate': gate}
        records.append(item)
        print(name, 'front_aperture', round(values['front_arm_aperture_fraction'], 3),
              'upper_span', round(values['upper_span_ratio'], 3), 'gate', gate)
    for pair_number in (1, 2):
        rows = records[(pair_number - 1) * 2:pair_number * 2]
        pair = OUT / f'pair_{pair_number:02d}'
        (pair / 'selection.json').write_text(json.dumps({'round': pair_number, 'selected': rows}, indent=2) + '\n')
        (pair / 'image_gate.json').write_text(json.dumps(rows, indent=2) + '\n')
    (OUT / 'protocol.json').write_text(json.dumps({
        'status': 'exploratory text controllability pilot',
        'reference_image': str(REF), 'generation': 'built-in imagegen; model ID not exposed',
        'image_to_3d': 'same Direct3D-S2 config/seed 42, 162px front view',
        'axes': ['front_arm_aperture_fraction', 'upper_span_ratio'],
        'front_aperture_roi': 'normalized 512px, x=[.24,.37] and [.63,.76], y=[.23,.38]',
        'variation': 'narrow/wide backrest crossed with open/closed arm openings',
        'BC': 'same chair specification; check raw seat/back coverage before FEM',
        'candidates': records,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
