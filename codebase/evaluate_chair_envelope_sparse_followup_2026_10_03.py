#!/usr/bin/env python3
"""Compare uncut sparse chair meshes in the registered physical envelope."""
from __future__ import annotations

import json
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label
from scipy.spatial.transform import Rotation
import trimesh

from make_chair_domain import ROOT

sys.path.insert(0, str(ROOT / 'codebase/code'))
sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from make_bc_proper_pysdf import voxel_centers_inside
from cond_render_pv import camera_from_elev_azim, render_lit

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'envelope_excess_pilot_2026-10-03'
SPEC = BASE / 'single_view_spec_2026-10-03'
SOURCE = BASE / 'text_latent_qd_2026-10-03/mesh_cases/faceted__curved__standard'
GUIDED = OUT / 'faceted__curved__standard__eta10'


def main() -> None:
    data = np.load(SPEC / 'voxel.npz')
    env = data['bracket'].astype(bool)
    tr = json.loads((SPEC / 'specification.json').read_text())['native_frame_registration']
    R = Rotation.from_euler('x', tr['rotation_x_degrees'], degrees=True).as_matrix()
    src = np.asarray(tr['source_center_m'])
    dst = np.asarray(tr['physical_center_m'])
    structure = np.zeros((3, 3, 3), bool)
    structure[1, 1, :] = True
    structure[1, :, 1] = True
    structure[:, 1, 1] = True
    mesh_old = trimesh.load(SOURCE / 'aligned_main.obj', force='mesh', process=False)
    mesh_new = trimesh.load(GUIDED / 'sparse_generation/mesh.obj', force='mesh', process=False)
    mesh_new.vertices = ((mesh_new.vertices - src) @ R.T) * tr['uniform_scale'] + dst
    mesh_new = max(mesh_new.split(only_watertight=False), key=lambda part: abs(part.volume))
    mesh_new.export(GUIDED / 'aligned_sparse.obj')
    mesh_new.export(GUIDED / 'aligned_sparse.glb')
    rows = []
    occupancies = []
    canvas = Image.new('RGB', (720, 2 * 394), 'white')
    draw = ImageDraw.Draw(canvas)
    center = np.asarray([0., 0.01, .46])
    for i, (name, mesh) in enumerate((('baseline', mesh_old), ('guided eta10', mesh_new))):
        occ = voxel_centers_inside(mesh, 64, data['origin'], data['pitch_xyz']).astype(bool)
        occupancies.append(occ)
        _, components = label(occ, structure=structure)
        rows.append({
            'name': name, 'mesh': str(SOURCE / 'aligned_main.obj' if i == 0 else GUIDED / 'aligned_sparse.obj'),
            'occupied_voxels': int(occ.sum()),
            'outside_voxels': int((occ & ~env).sum()),
            'outside_fraction': float((occ & ~env).sum() / max(1, int(occ.sum()))),
            'seat_bc_coverage': float((occ & data['load']).sum() / data['load'].sum()),
            'back_bc_coverage': float((occ & data['back_load']).sum() / data['back_load'].sum()),
            'six_connected_components': int(components),
            'watertight': bool(mesh.is_watertight),
        })
        for col, (elev, azim) in enumerate(((15, 0), (15, 90))):
            eye, up = camera_from_elev_azim(center, 2.0, elev, azim)
            image = render_lit(mesh, eye, center, up, size=360,
                               fit_extent=.56, color=(.62, .66, .7))
            canvas.paste(Image.fromarray(image).convert('RGB'), (col * 360, i * 394 + 34))
            draw.text((col * 360 + 8, i * 394 + 8),
                      f'{name} / {"front" if col == 0 else "side"}', fill='#17212b')
    a, b = occupancies
    rows[1]['voxel_iou_to_baseline'] = float((a & b).sum() / max(1, int((a | b).sum())))
    (OUT / 'sparse_metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    canvas.save(OUT / 'sparse_comparison.png')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
