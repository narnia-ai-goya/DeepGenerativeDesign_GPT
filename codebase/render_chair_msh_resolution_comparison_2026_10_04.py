"""Compare actual tetra cross-sections at 35, 20, and 15 mm mesh settings."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')

import meshio
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SPEC = (ROOT / 'experiments/chair/sofa_style_2026-09-28'
        / 'text_reasoned_front_axes_2026-10-03/long_qd_fea_2026-10-03'
        / 'envelope_plus16_spec_2026-10-03/fea_domain')
OUT = (ROOT / 'experiments/chair/sofa_style_2026-09-28'
       / 'text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03'
       / 'llm_text_proposal_round_03_2026-10-03'
       / 'simultaneous_load_dense_sparse_2026-10-04/diagnostics'
       / 'msh_resolution_comparison_2026-10-04.png')


def panel(mesh_path: Path) -> tuple[Image.Image, int, int, float]:
    m = meshio.read(mesh_path)
    tet = np.vstack([b.data for b in m.cells if b.type == 'tetra'])
    q = m.points[tet]
    edge = np.concatenate([np.linalg.norm(q[:, i]-q[:, j], axis=1)
                           for i, j in ((0, 1), (0, 2), (0, 3),
                                        (1, 2), (1, 3), (2, 3))])
    cells = np.column_stack([np.full(len(tet), 4, dtype=np.int64), tet]).ravel()
    grid = pv.UnstructuredGrid(cells,
                               np.full(len(tet), pv.CellType.TETRA, np.uint8), m.points)
    center = np.array(grid.center)
    cut = grid.slice(normal=(1, 0, 0), origin=(0, center[1], center[2]))
    pl = pv.Plotter(off_screen=True, window_size=(560, 600))
    pl.set_background('#fcfdff')
    pl.add_mesh(cut, color='#e0b184', show_edges=True,
                edge_color='#693f30', line_width=.55, lighting=False)
    pl.camera_position = [[2, center[1], center[2]], center.tolist(), [0, 0, 1]]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = .55
    image = Image.fromarray(pl.screenshot(return_img=True)).convert('RGB')
    pl.close()
    return image, len(m.points), len(tet), float(np.median(edge)*1000)


def main() -> None:
    tile, header = 560, 115
    sheet = Image.new('RGB', (3*tile, header+600), '#fcfdff')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    draw.text((22, 12), 'Same chair envelope · tetrahedral midplane X=0',
              font=font, fill='#25374a')
    for i, size in enumerate((35, 20, 15)):
        image, nodes, tets, med = panel(SPEC / f'chair_{size:03d}.msh')
        sheet.paste(image, (i*tile, header))
        draw.text((i*tile+20, 57),
                  f'{size} mm setting: {tets:,} tets · median edge {med:.1f} mm',
                  font=small, fill='#365065')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT)
    print(OUT)


if __name__ == '__main__':
    main()
