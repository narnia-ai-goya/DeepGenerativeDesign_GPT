#!/usr/bin/env python3
"""Check smooth sparse SDF on the second chair against its own baseline."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import trimesh

from make_chair_domain import ROOT
from report_chair_sparse_wrinkle_pilot_2026_10_03 import surface_metrics

sys.path.insert(0, str(ROOT / 'codebase/code'))
sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from make_bc_proper_pysdf import voxel_centers_inside
from cond_render_pv import camera_from_elev_azim, render_lit

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SOURCE = BASE / 'text_latent_qd_2026-10-03/mesh_cases/tapered__straight__standard'
OUT = BASE / 'sparse_wrinkle_pilot_2026-10-03'
SPEC = BASE / 'single_view_spec_2026-10-03'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='tapered_smooth3_match')
    parser.add_argument('--stem', default='transfer')
    args = parser.parse_args()
    case = OUT / args.case
    data = np.load(SPEC / 'voxel.npz')
    env = data['bracket'].astype(bool)
    reg = json.loads((SPEC / 'specification.json').read_text())['native_frame_registration']
    R = Rotation.from_euler('x', reg['rotation_x_degrees'], degrees=True).as_matrix()
    src = np.asarray(reg['source_center_m'])
    dst = np.asarray(reg['physical_center_m'])
    old = trimesh.load(SOURCE / 'aligned_main.obj', force='mesh', process=False)
    new = trimesh.load(case / 'generation/mesh.obj', force='mesh', process=False)
    components = new.split(only_watertight=False)
    largest_fraction = max(abs(part.volume) for part in components) / sum(abs(part.volume) for part in components)
    new.vertices = ((new.vertices - src) @ R.T) * reg['uniform_scale'] + dst
    new = max(new.split(only_watertight=False), key=lambda part: abs(part.volume))
    new.export(case / 'aligned_main.obj')
    new.export(case / 'aligned_main.glb')
    rows = []
    canvas = Image.new('RGB', (990, 368 * 2), 'white')
    draw = ImageDraw.Draw(canvas)
    center = np.asarray([0., .01, .46])
    occ0 = None
    for i, (name, mesh) in enumerate((('baseline', old), (args.case, new))):
        occ = voxel_centers_inside(mesh, 64, data['origin'], data['pitch_xyz']).astype(bool)
        if occ0 is None: occ0 = occ.copy()
        row = {'name': name, 'mesh': str(SOURCE / 'aligned_main.obj' if i == 0 else case / 'aligned_main.obj'),
               'occupied_voxels': int(occ.sum()),
               'voxel_iou_to_baseline': float((occ & occ0).sum() / max(1, int((occ | occ0).sum()))),
               'outside_fraction': float((occ & ~env).sum() / max(1, int(occ.sum()))),
               'seat_bc_coverage': float((occ & data['load']).sum() / data['load'].sum()),
               'surface': surface_metrics(mesh)}
        if i == 1:
            row['raw_mesh_components'] = len(components)
            row['largest_component_volume_fraction'] = float(largest_fraction)
        rows.append(row)
        for col, (elev, azim, target, extent) in enumerate((
            (15, 0, center, .56), (15, 90, center, .56), (10, 15, np.asarray([0., .17, .77]), .20))):
            eye, up = camera_from_elev_azim(target, 2.0, elev, azim)
            rendered = render_lit(mesh, eye, target, up, size=330, fit_extent=extent,
                                  color=(.62, .66, .7))
            canvas.paste(Image.fromarray(rendered).convert('RGB'), (col * 330, i * 368 + 38))
            draw.text((col * 330 + 8, i * 368 + 9),
                      f'{name} / {("front", "side", "backrest detail")[col]}', fill='#17212b')
    (OUT / f'{args.stem}_metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    canvas.save(OUT / f'{args.stem}_comparison.png')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
