#!/usr/bin/env python3
"""Compare the image target with old and camera-guided chair final meshes."""
from __future__ import annotations

import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from make_chair_domain import ROOT

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402


BASE = ROOT / 'experiments/chair/image_concepts_2026-09-27/diagonal_braced'
VIEWS = (('hero', 28, 40), ('front', 15, 0), ('right', 15, 90), ('top', 85, 0))
TARGET = {
    'hero': BASE / 'input_512.png',
    'front': BASE / 'multiview_registered/input/v00_front_lo.png',
    'right': BASE / 'multiview_registered/input/v02_right_lo.png',
    'top': BASE / 'multiview_registered/input/v_top.png',
}
MESHES = {
    'previous_final': BASE / 'no_mid_vw50/post/final.obj',
    'camera_guided_5mm_final': BASE / 'camera_proj_w50/post_edge5/final.obj',
}


def main() -> None:
    env = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents))*1.5
    scale = float(env.extents.max())/2
    sheet = Image.new('RGB', (512*4, 548*3), 'white')
    draw = ImageDraw.Draw(sheet)
    for j, (view, _, _) in enumerate(VIEWS):
        target = Image.open(TARGET[view]).convert('RGB')
        sheet.paste(target, (j*512, 36))
        draw.text((j*512+12, 10), f'input · {view}', fill='#20252b')
    summary = {}
    for i, (name, path) in enumerate(MESHES.items(), start=1):
        mesh = trimesh.load(path, force='mesh')
        scores = {}
        for j, (view, elev, azim) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rendered = Image.fromarray(render_lit(
                mesh, eye, center, up, size=512, fit_extent=scale,
                margin=1.15, color=(.42, .46, .49))).convert('RGB')
            sheet.paste(rendered, (j*512, i*548+36))
            draw.text((j*512+12, i*548+10), f'{name} · {view}', fill='#20252b')
            if view != 'hero':
                generated = np.asarray(rendered).min(axis=2) < 210
                target = np.asarray(Image.open(TARGET[view]).convert('RGB')).min(axis=2) < 210
                scores[view] = {
                    'iou': round(float((generated & target).sum()
                                       / max((generated | target).sum(), 1)), 4),
                    'false_positive_fraction': round(float((generated & ~target).sum()
                                                     / max(generated.sum(), 1)), 4),
                    'false_negative_fraction': round(float((target & ~generated).sum()
                                                     / max(target.sum(), 1)), 4),
                }
        summary[name] = {'mesh': str(path), 'volume_litres': round(abs(float(mesh.volume))*1000, 4),
                         'silhouettes': scores}
    destination = BASE / 'fidelity_comparison.png'
    sheet.save(destination)
    (BASE / 'fidelity_comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
