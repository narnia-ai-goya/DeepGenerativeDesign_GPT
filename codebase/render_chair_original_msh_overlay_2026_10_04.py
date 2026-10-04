"""Overlay the original tapered chair OBJ on the 15 mm FEM domain in one view."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import meshio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv
import trimesh

from make_chair_domain import ROOT
from run_chair_existing_fea_on_015_2026_10_04 import OUT, SPEC


MSH = SPEC / "fea_domain/chair_015.msh"
CHAIR = ROOT / "experiments/chair/sofa_style_2026-09-28/tapered_examples_2026-10-03/final_meshes/tapered_original.obj"
DEST = OUT / "diagnostics/fea_log_search_2026-10-04"


def main() -> None:
    m = meshio.read(MSH)
    tets = np.vstack([block.data for block in m.cells if block.type == "tetra"])
    cells = np.column_stack([np.full(len(tets), 4, np.int64), tets]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(tets), pv.CellType.TETRA, np.uint8), m.points)
    domain = grid.extract_surface()
    chair = trimesh.load(CHAIR, force="mesh", process=False)
    # This source chair is already in the FEM domain frame.  Applying the
    # registration used for generated pipeline OBJ files would misalign it.
    lo, hi = m.points.min(axis=0), m.points.max(axis=0)
    inside_bbox = np.all((chair.vertices >= lo) & (chair.vertices <= hi), axis=1)
    if not inside_bbox.all():
        raise ValueError("chair OBJ does not fit the FEM frame")
    chair_pv = pv.wrap(chair).compute_normals(point_normals=True, cell_normals=False,
                                             auto_orient_normals=True)
    center = np.asarray(grid.center)
    eye = center + np.array([.85, -1.2, .8])
    plotter = pv.Plotter(off_screen=True, window_size=(1100, 1100))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("msaa")
    plotter.enable_depth_peeling(number_of_peels=8)
    plotter.add_mesh(domain, color="#8ab0c4", opacity=.12,
                     smooth_shading=False, show_edges=False)
    edge = domain.extract_feature_edges(feature_angle=30, boundary_edges=True,
                                        manifold_edges=False)
    plotter.add_mesh(edge, color="#53778d", opacity=.75, line_width=2)
    plotter.add_mesh(chair_pv, color="#727d88", smooth_shading=True,
                     ambient=.28, diffuse=.65, specular=.18, specular_power=24)
    plotter.camera_position = [eye.tolist(), center.tolist(), [0, 0, 1]]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = .78
    image = Image.fromarray(plotter.screenshot(return_img=True,
                                               transparent_background=True)).convert("RGBA")
    plotter.close()
    DEST.mkdir(parents=True, exist_ok=True)
    output = DEST / "chair_original_overlaid_on_015_msh_transparent.png"
    image.save(output)
    print(output)
    chair_plotter = pv.Plotter(off_screen=True, window_size=(1100, 1100))
    chair_plotter.set_background("white")
    chair_plotter.enable_anti_aliasing("msaa")
    chair_plotter.add_mesh(chair_pv, color="#727d88", smooth_shading=True,
                           ambient=.28, diffuse=.65, specular=.18, specular_power=24)
    chair_plotter.camera_position = [eye.tolist(), center.tolist(), [0, 0, 1]]
    chair_plotter.camera.parallel_projection = True
    chair_plotter.camera.parallel_scale = .78
    chair_image = Image.fromarray(chair_plotter.screenshot(
        return_img=True, transparent_background=True)).convert("RGBA")
    chair_plotter.close()
    chair_output = DEST / "chair_original_same_view_transparent.png"
    chair_image.save(chair_output)
    print(chair_output)
    domain_plotter = pv.Plotter(off_screen=True, window_size=(1100, 1100))
    domain_plotter.set_background("white")
    domain_plotter.enable_anti_aliasing("msaa")
    domain_plotter.add_mesh(domain, color="#a8bac8", show_edges=True,
                            edge_color="#45667d", line_width=.45,
                            smooth_shading=False, ambient=.35, diffuse=.65)
    domain_plotter.camera_position = [eye.tolist(), center.tolist(), [0, 0, 1]]
    domain_plotter.camera.parallel_projection = True
    domain_plotter.camera.parallel_scale = .78
    domain_image = Image.fromarray(domain_plotter.screenshot(
        return_img=True, transparent_background=True)).convert("RGBA")
    domain_plotter.close()
    domain_output = DEST / "chair_015_msh_surface_matched_1100_transparent.png"
    domain_image.save(domain_output)
    print(domain_output)
    print(f"source_obj={CHAIR}")
    print(f"source_msh={MSH}")
    print(f"chair_vertices_inside_FEM_bbox={inside_bbox.mean():.6f}")


if __name__ == "__main__":
    main()
