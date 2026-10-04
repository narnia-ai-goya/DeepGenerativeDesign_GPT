"""Render the actual Sparse triangles, without 64^3 occupancy conversion."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

from make_chair_domain import ROOT

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402


OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03/sparse_fea_loop_2026-10-03'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def preview(row: dict) -> Path:
    path = Path(row['aligned_mesh']).with_name('sparse_preview.png')
    if path.exists():
        return path
    mesh = trimesh.load(row['aligned_mesh'], force='mesh', process=False)
    page = Image.new('RGB', (870, 460), '#f6f8fa')
    draw = ImageDraw.Draw(page)
    center = np.array([0., .01, .46])
    for column, (label, azim) in enumerate((('FRONT', 0), ('RIGHT', 90))):
        eye, up = camera_from_elev_azim(center, 2., 15, azim)
        rgb = render_lit(mesh, eye, center, up, size=420, fit_extent=.56,
                         color=(.64, .69, .73))
        page.paste(Image.fromarray(rgb).convert('RGB'), (column * 435 + 7, 27))
        draw.text((column * 435 + 18, 7), label, font=ImageFont.truetype(BOLD, 16), fill='#263c4a')
    draw.text((10, 445), 'Original Sparse triangles · physical-frame alignment only',
              font=ImageFont.truetype(FONT, 12), fill='#506674')
    page.save(path)
    return path


def main() -> None:
    rows = json.loads((OUT / 'result.json').read_text())['selected']
    for row in rows:
        image = preview(row)
        print(row['id'], image, flush=True)


if __name__ == '__main__':
    main()
