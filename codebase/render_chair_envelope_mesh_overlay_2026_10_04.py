"""Render the actual physical chair mesh inside its physical envelope STL."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
STUDY = (BASE / 'text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03'
         / 'llm_text_proposal_round_03_2026-10-03'
         / 'simultaneous_load_dense_sparse_2026-10-04')
ENVELOPE = (BASE / 'text_reasoned_front_axes_2026-10-03/long_qd_fea_2026-10-03'
            / 'envelope_plus16_spec_2026-10-03/envelope.stl')
CHAIR = STUDY / 'sparse_evaluation/round_01/mesh_cases/sparse_d0_s1/aligned_main.obj'
OUT = STUDY / 'diagnostics/actual_envelope_chair_overlay_2026-10-04.png'


def render(mesh: pv.PolyData, envelope: pv.PolyData, view: str, azim: float,
           elev: float, center: np.ndarray, radius: float, scale: float) -> Image.Image:
    size = 640
    pl = pv.Plotter(off_screen=True, window_size=(size, size))
    pl.set_background('#fafcff')
    pl.enable_anti_aliasing('msaa')
    eye = center + radius * np.array([
        np.cos(np.deg2rad(elev)) * np.sin(np.deg2rad(azim)),
        -np.cos(np.deg2rad(elev)) * np.cos(np.deg2rad(azim)),
        np.sin(np.deg2rad(elev)),
    ])
    pl.add_mesh(mesh, color='#777e89', smooth_shading=True, ambient=.31,
                diffuse=.62, specular=.25, specular_power=28)
    # Same physical STL, drawn as a translucent shell and an explicit edge cage.
    pl.add_mesh(envelope, color='#38a4c9', opacity=.10,
                ambient=.7, diffuse=.3)
    # Only geometric creases: triangle diagonals are tessellation artifacts.
    creases = envelope.extract_feature_edges(feature_angle=25,
                                              boundary_edges=False,
                                              non_manifold_edges=False,
                                              feature_edges=True,
                                              manifold_edges=False)
    pl.add_mesh(creases, color='#037da6', line_width=2.0,
                render_lines_as_tubes=True)
    pl.add_light(pv.Light(position=tuple(eye + [0, 0, .5]),
                          focal_point=tuple(center), intensity=.8))
    pl.add_light(pv.Light(position=tuple(center + [-1, 1, .7]),
                          focal_point=tuple(center), intensity=.5))
    pl.camera_position = [eye.tolist(), center.tolist(), [0, 0, 1]]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = scale
    array = pl.screenshot(return_img=True)
    pl.close()
    return Image.fromarray(array).convert('RGB')


def main() -> None:
    envelope = pv.read(ENVELOPE).triangulate()
    chair = pv.read(CHAIR).triangulate()
    center = np.array(envelope.center)
    radius = 2.0 * np.linalg.norm(np.array(envelope.bounds)[[1, 3, 5]] -
                                  np.array(envelope.bounds)[[0, 2, 4]])
    scale = 0.59
    views = [('Front', 0, 12), ('Right side', 90, 10), ('Three-quarter', 42, 24)]
    tile = 640
    header = 110
    sheet = Image.new('RGB', (tile * 3, tile + header), '#fafcff')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    draw.text((24, 15), 'Actual generated chair + physical envelope',
              fill='#172a3a', font=font)
    draw.rectangle((25, 62, 45, 82), fill='#777e89')
    draw.text((55, 61), 'generated sparse chair', fill='#334452', font=small)
    draw.rectangle((330, 62, 350, 82), fill='#38a4c9', outline='#037da6', width=2)
    draw.text((360, 61), 'physical design envelope STL', fill='#334452', font=small)
    for i, (name, azim, elev) in enumerate(views):
        image = render(chair, envelope, name, azim, elev, center, radius, scale)
        sheet.paste(image, (i * tile, header))
        draw.text((i * tile + 24, header + 17), name,
                  fill='#172a3a', font=small)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT)
    print(OUT)
    print('chair', CHAIR)
    print('envelope', ENVELOPE)
    print('chair bounds', chair.bounds)
    print('envelope bounds', envelope.bounds)


if __name__ == '__main__':
    main()
