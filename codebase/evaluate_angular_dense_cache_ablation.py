#!/usr/bin/env python3
"""Compare dense-cache reuse with the original FEA-on angular runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from diagnose_angular_stage_fidelity import (CASES, EXP, N, OUT, PIXELS_PER_M,
                                            bc_centers, render_top, summary_for_pair)


ABL = OUT / 'dense_cache_reuse_ablation'


def main() -> None:
    source_metrics = json.loads((OUT / 'metrics.json').read_text())
    rows = {}
    for name in ('triangular_truss', 'staggered_chevron'):
        original = CASES[name]
        case = ABL / name
        config = json.loads((case / 'config_full.json').read_text())
        output = case / 'post/final.obj'
        if not output.exists():
            raise FileNotFoundError(output)
        mask = render_top(output, case / 'reuse_final_top.png')
        yy, xx = np.indices((N, N))
        bc = np.zeros((N, N), bool)
        for i, (cx, cy) in enumerate(bc_centers(config)):
            radius = (10 if i < 4 else 8) * PIXELS_PER_M / 1000
            bc |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        design = ~bc
        source = source_metrics['cases'][name]
        original_m = json.loads((original / 'metrics.json').read_text())
        reuse_m = json.loads((case / 'metrics.json').read_text())
        result = {'original_mesh': str(original / 'post/final.obj'),
                  'cache_reuse_mesh': str(output),
                  'baseline_geometry_valid': original_m['geometry_valid'],
                  'cache_reuse_geometry_valid': reuse_m['geometry_valid'],
                  'baseline_compliance_J': original_m.get('compliance_J'),
                  'cache_reuse_compliance_J': reuse_m.get('compliance_J'),
                  'baseline_volume_cm3': original_m['volume_cm3'],
                  'cache_reuse_volume_cm3': reuse_m['volume_cm3'],
                  'cache_reuse_bc_containment': reuse_m['bc_containment']}
        for mode in ('affine', 'tps'):
            rgb = np.asarray(Image.open(OUT / name / f'input_registered_{mode}.png').convert('RGB'))
            reference = rgb.min(axis=2) < 230
            candidate = summary_for_pair(reference, mask, design, bc)
            result[f'cache_reuse_image_to_final_{mode}'] = candidate
            result[f'baseline_image_to_final_{mode}'] = source['image_to_stage'][mode]['final']
        result['baseline_final_void_recall'] = source['image_to_stage']['affine']['final']['void_recall']
        result['cache_reuse_final_void_recall'] = result['cache_reuse_image_to_final_affine']['void_recall']
        # A diagnostic sheet with the same camera and image registration.
        paths = [OUT / name / 'input_registered_affine.png',
                 OUT / name / 'top_dense_top.png',
                 OUT / name / 'final_top.png',
                 case / 'reuse_final_top.png']
        images = [Image.open(path).convert('RGB') for path in paths]
        sheet = Image.new('RGB', (N * len(images), N + 42), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, (label, image) in enumerate(zip(('input', 'top dense', 'original final', 'reuse final'), images)):
            sheet.paste(image, (i * N, 42))
            draw.text((i * N + 12, 10), f'{name}: {label}', fill='black')
        sheet.save(case / 'comparison.png')
        rows[name] = result
    result = {'question': 'Does keeping the FEA-on top-only dense occupancy during multiview sparse preserve input openings?',
              'cases': rows}
    (ABL / 'ablation_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(ABL / 'ablation_metrics.json')


if __name__ == '__main__':
    main()
