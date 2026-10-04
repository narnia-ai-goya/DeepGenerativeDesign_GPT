"""Render the actual cached chair-envelope tetrahedral FEM mesh."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')

import meshio
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SPEC = (BASE / 'text_reasoned_front_axes_2026-10-03/long_qd_fea_2026-10-03'
        / 'envelope_plus16_spec_2026-10-03')
MESH = SPEC / 'fea_domain/chair_035.msh'
OUT = (BASE / 'text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03'
       / 'llm_text_proposal_round_03_2026-10-03'
       / 'simultaneous_load_dense_sparse_2026-10-04/diagnostics'
       / 'chair_035_msh_preview_2026-10-04.png')


def camera(center: np.ndarray, angle: float, elevation: float):
    a, e = np.deg2rad([angle, elevation])
    eye = center + 2.1 * np.array([np.cos(e)*np.sin(a),
                                   -np.cos(e)*np.cos(a), np.sin(e)])
    return [eye.tolist(), center.tolist(), [0, 0, 1]]


def snapshot(grid: pv.UnstructuredGrid, mode: str, angle: float, elevation: float,
             center: np.ndarray) -> Image.Image:
    pl = pv.Plotter(off_screen=True, window_size=(610, 610))
    pl.set_background('#fbfdff')
    pl.enable_anti_aliasing('msaa')
    surface = grid.extract_surface()
    if mode == 'surface':
        pl.add_mesh(surface, color='#aeb8c4', smooth_shading=False,
                    ambient=.34, diffuse=.65, show_edges=True,
                    edge_color='#365f78', line_width=.6)
    elif mode == 'side':
        pl.add_mesh(surface, color='#b7c2cc', smooth_shading=False,
                    ambient=.32, diffuse=.65, show_edges=True,
                    edge_color='#38647a', line_width=.65)
    else:
        pl.add_mesh(surface, color='#9dcce1', opacity=.09,
                    show_edges=False)
        section = grid.slice(normal=(1, 0, 0), origin=(0, center[1], center[2]))
        pl.add_mesh(section, color='#e4ae76', opacity=.96,
                    show_edges=True, edge_color='#6d3c25', line_width=.8,
                    lighting=False)
        pl.add_mesh(section.extract_feature_edges(feature_angle=1),
                    color='#6d3c25', line_width=.9)
    pl.camera_position = camera(center, angle, elevation)
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = .59
    result = Image.fromarray(pl.screenshot(return_img=True)).convert('RGB')
    pl.close()
    return result


def main() -> None:
    m = meshio.read(MESH)
    tets = np.vstack([block.data for block in m.cells if block.type == 'tetra'])
    cells = np.column_stack([np.full(len(tets), 4, dtype=np.int64), tets]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(tets), pv.CellType.TETRA, np.uint8),
                               m.points)
    center = np.array(grid.center)
    views = [('Isometric surface', 'surface', 38, 22),
             ('Right-side surface', 'side', 90, 12),
             ('Midplane tetra cut, X=0', 'section', 90, 4)]
    tile, header = 610, 108
    image = Image.new('RGB', (tile*3, tile+header), '#fbfdff')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    draw.text((24, 16), 'Actual FEA tetrahedral mesh · chair_035.msh',
              font=font, fill='#203040')
    draw.text((24, 57), f'{len(m.points):,} nodes  ·  {len(tets):,} tetrahedra  ·  nominal mesh size 35 mm',
              font=small, fill='#456070')
    for i, (name, mode, angle, elevation) in enumerate(views):
        tile_image = snapshot(grid, mode, angle, elevation, center)
        image.paste(tile_image, (i*tile, header))
        draw.text((i*tile+22, header+16), name, font=small, fill='#203040')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT)
    print(OUT)
    print(MESH)


if __name__ == '__main__':
    main()
