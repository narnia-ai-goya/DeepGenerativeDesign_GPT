#!/usr/bin/env python3
"""Build a neutral openwork chair reference and three aligned conditioning views."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from make_chair_domain import ROOT, box, foot

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit


OUT = ROOT / 'experiments/chair/pilot_2026-09-25'


def main() -> None:
    inp = OUT / 'input'
    inp.mkdir(parents=True, exist_ok=True)
    legs = [box(x-.042, x+.042, y-.045, y+.045, .025, .455)
            for y in (-.18, .18) for x in (-.195, .195)]
    feet = [foot(x, y) for y in (-.18, .18) for x in (-.195, .195)]
    parts = [box(-.265, .265, -.255, .255, .430, .500),
             box(-.245, -.165, .17, .255, .480, .870),
             box(.165, .245, .17, .255, .480, .870),
             box(-.245, .245, .17, .255, .805, .870),
             box(-.245, .245, .17, .255, .635, .685),
             *legs, *feet]
    reference = trimesh.boolean.union(parts, engine='manifold')
    if not reference.is_watertight or len(reference.split()) != 1:
        raise RuntimeError('chair reference must be watertight and connected')
    reference.export(OUT / 'neutral_reference.obj')
    envelope = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = envelope.bounds.mean(axis=0)
    extent = envelope.extents
    radius = float(np.linalg.norm(extent)) * 1.5
    scale = float(extent.max()) / 2
    views = [('v00_front_lo', 15, 0), ('v02_right_lo', 15, 90),
             ('v04_back_lo', 15, 180), ('v06_left_lo', 15, 270),
             ('v_top', 85, 0), ('v_bottom', -85, 0)]
    sheet = Image.new('RGB', (512 * len(views), 548), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (name, elev, azim) in enumerate(views):
        eye, up = camera_from_elev_azim(center, radius, elev, azim)
        rgb = render_lit(reference, eye, center, up, size=512, fit_extent=scale,
                         margin=1.15, color=(.25, .28, .31))
        image = Image.fromarray(rgb).convert('RGB')
        image.save(inp / f'{name}.png')
        sheet.paste(image, (i * 512, 36))
        draw.text((i * 512 + 10, 10), name, fill='#20252b')
    sheet.save(OUT / 'input_contact.png')
    print(OUT / 'input_contact.png')


if __name__ == '__main__':
    main()
