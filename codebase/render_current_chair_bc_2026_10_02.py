#!/usr/bin/env python3
"""Render the exact voxel envelope and BC masks used by current chair dense runs."""
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import marching_cubes

ROOT = Path('/home/goya/SDL/3d_qd')
SOURCE = ROOT / 'experiments/chair/sofa_style_2026-09-28/consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/current_bc_visualization_2026-10-02'
COLORS = {'bracket': '#aab4bf', 'fix': '#c94c40', 'load': '#e9a228', 'back_load': '#9b4fc2'}
VIEWS = (('3D oblique', 23, 35), ('front', 8, 0), ('right side', 8, 90), ('top', 82, 0))


def surface(mask, origin, pitch):
    vertices, faces, _, _ = marching_cubes(mask.astype(np.uint8), level=0.5, spacing=tuple(pitch))
    vertices += origin + pitch / 2
    cells = np.column_stack((np.full(len(faces), 3), faces)).ravel()
    return pv.PolyData(vertices, cells)


def render(meshes, center, distance, elev, azim):
    import math
    e, a = math.radians(elev), math.radians(azim)
    eye = center + distance * np.array([math.cos(e) * math.sin(a), -math.cos(e) * math.cos(a), math.sin(e)])
    pl = pv.Plotter(off_screen=True, window_size=(850, 720))
    pl.set_background('white')
    pl.enable_anti_aliasing('ssaa')
    pl.add_mesh(meshes['bracket'], color=COLORS['bracket'], opacity=0.13, show_edges=True,
                edge_color='#8d99a6', line_width=0.35, lighting=False)
    for name in ('fix', 'load', 'back_load'):
        pl.add_mesh(meshes[name], color=COLORS[name], opacity=1, smooth_shading=False,
                    ambient=0.35, diffuse=0.65, specular=0.12)
    pl.camera_position = [eye.tolist(), center.tolist(), [0, 0, 1]]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = 0.61
    image = pl.screenshot(return_img=True)
    pl.close()
    return Image.fromarray(image).convert('RGB')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(SOURCE)
    origin = np.asarray(data['origin'], dtype=float)
    pitch = np.asarray(data['pitch_xyz'], dtype=float)
    meshes = {name: surface(data[name], origin, pitch) for name in COLORS}
    center = np.array([0.0, 0.01, 0.46])
    canvas = Image.new('RGB', (1700, 1550), '#f7f8fa')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 23)
    title_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 34)
    draw.text((30, 15), 'Current chair envelope and boundary conditions', font=title_font, fill='#243040')
    for idx, (name, elev, azim) in enumerate(VIEWS):
        x, y = (idx % 2) * 850, 70 + (idx // 2) * 720
        canvas.paste(render(meshes, center, 2.2, elev, azim), (x, y))
        draw.text((x + 20, y + 12), name, font=font, fill='#243040')
    legend = [('envelope', 'bracket'), ('fix: 4 feet', 'fix'),
              ('load: seat', 'load'), ('load: backrest', 'back_load')]
    for idx, (label, key) in enumerate(legend):
        x = 45 + idx * 410
        draw.rectangle((x, 1510, x + 28, 1538), fill=COLORS[key])
        draw.text((x + 38, 1508), label, font=font, fill='#243040')
    canvas.save(OUT / 'envelope_load_fix.png')
    print(OUT / 'envelope_load_fix.png')


if __name__ == '__main__':
    main()
