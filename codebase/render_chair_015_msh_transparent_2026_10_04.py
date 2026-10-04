"""Transparent renders of the exact 15 mm tetrahedral mesh used in FEA."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import meshio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv

from run_chair_existing_fea_on_015_2026_10_04 import OUT, SPEC


MESH = SPEC / "fea_domain/chair_015.msh"
DEST = OUT / "diagnostics/fea_log_search_2026-10-04"


def capture(grid: pv.UnstructuredGrid, mode: str) -> Image.Image:
    surface = grid.extract_surface()
    center = np.asarray(grid.center)
    plotter = pv.Plotter(off_screen=True, window_size=(850, 850))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("msaa")
    if mode == "surface":
        plotter.add_mesh(surface, color="#a8bac8", show_edges=True,
                         edge_color="#45667d", line_width=.45,
                         smooth_shading=False, ambient=.35, diffuse=.65)
        eye = center + np.array([.85, -1.2, .8])
    else:
        plotter.add_mesh(surface, color="#a8bac8", opacity=.12,
                         show_edges=False, smooth_shading=False)
        section = grid.slice(normal=(1, 0, 0), origin=center)
        plotter.add_mesh(section, color="#e7a268", show_edges=True,
                         edge_color="#814e32", line_width=.7,
                         ambient=.4, diffuse=.6)
        eye = center + np.array([1.3, -.65, .45])
    plotter.camera_position = [eye.tolist(), center.tolist(), [0, 0, 1]]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = .78
    rgba = Image.fromarray(plotter.screenshot(return_img=True,
                                              transparent_background=True)).convert("RGBA")
    plotter.close()
    return rgba


def main() -> None:
    mesh = meshio.read(MESH)
    tets = np.vstack([block.data for block in mesh.cells if block.type == "tetra"])
    cells = np.column_stack([np.full(len(tets), 4, np.int64), tets]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(tets), pv.CellType.TETRA, np.uint8),
                               mesh.points)
    DEST.mkdir(parents=True, exist_ok=True)
    images = []
    for mode in ("surface", "section"):
        image = capture(grid, mode)
        path = DEST / f"chair_015_msh_{mode}_transparent.png"
        image.save(path)
        images.append(image)
        print(path)
    canvas = Image.new("RGBA", (1700, 910), (0, 0, 0, 0))
    for col, image in enumerate(images):
        canvas.alpha_composite(image, (850 * col, 60))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    draw.text((28, 14), "FEA mesh | 15 mm | exterior tetra surface", fill="#263c4a", font=font)
    draw.text((878, 14), "Midplane cut | tetrahedral cells", fill="#263c4a", font=font)
    output = DEST / "chair_015_msh_transparent.png"
    canvas.save(output)
    print(output)
    print(f"source={MESH} nodes={len(mesh.points)} tetrahedra={len(tets)}")


if __name__ == "__main__":
    main()
