#!/usr/bin/env python3
"""Compare sparse-only surface changes with fixed camera and geometric metrics."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import trimesh

from make_chair_domain import ROOT

sys.path.insert(0, str(ROOT / 'codebase/code'))
sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from make_bc_proper_pysdf import voxel_centers_inside
from cond_render_pv import camera_from_elev_azim, render_lit

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SOURCE = BASE / 'text_latent_qd_2026-10-03/mesh_cases/faceted__curved__standard'
OUT = BASE / 'sparse_wrinkle_pilot_2026-10-03'
SPEC = BASE / 'single_view_spec_2026-10-03'
NAMES = ('baseline', 'cfg4', 'steps50', 'cfg4_steps50', 'iso040')


def surface_metrics(mesh: trimesh.Trimesh) -> dict:
    center = mesh.triangles_center
    regions = {
        'backrest': (center[:, 1] > .14) & (center[:, 2] > .68) & (abs(center[:, 0]) < .24),
        'armrests': (center[:, 2] > .58) & (center[:, 2] < .8) & (abs(center[:, 0]) > .16),
    }
    adj = mesh.face_adjacency
    angle = mesh.face_adjacency_angles
    result = {}
    for name, mask in regions.items():
        selected = angle[mask[adj[:, 0]] & mask[adj[:, 1]]]
        result[name] = {'faces': int(mask.sum()),
                        'mean_dihedral_radians': float(selected.mean()),
                        'p90_dihedral_radians': float(np.percentile(selected, 90))}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--names', nargs='*', default=list(NAMES))
    parser.add_argument('--stem', default='')
    args = parser.parse_args()
    names = ('baseline',) + tuple(name for name in args.names if name != 'baseline')
    spec = np.load(SPEC / 'voxel.npz')
    env = spec['bracket'].astype(bool)
    registration = json.loads((SPEC / 'specification.json').read_text())['native_frame_registration']
    R = Rotation.from_euler('x', registration['rotation_x_degrees'], degrees=True).as_matrix()
    src = np.asarray(registration['source_center_m'])
    dst = np.asarray(registration['physical_center_m'])
    canvas = Image.new('RGB', (330 * 3, len(names) * 368), 'white')
    draw = ImageDraw.Draw(canvas)
    center = np.asarray([0., .01, .46])
    rows = []
    reference_occ = None
    for i, name in enumerate(names):
        if name == 'baseline':
            mesh_path = SOURCE / 'aligned_main.obj'
            mesh = trimesh.load(mesh_path, force='mesh', process=False)
            raw_components = len(mesh.split(only_watertight=False))
        else:
            case = OUT / name
            raw = trimesh.load(case / 'generation/mesh.obj', force='mesh', process=False)
            raw_components = len(raw.split(only_watertight=False))
            raw.vertices = ((raw.vertices - src) @ R.T) * registration['uniform_scale'] + dst
            mesh = max(raw.split(only_watertight=False), key=lambda part: abs(part.volume))
            mesh_path = case / 'aligned_main.obj'
            mesh.export(mesh_path)
            mesh.export(case / 'aligned_main.glb')
        occ = voxel_centers_inside(mesh, 64, spec['origin'], spec['pitch_xyz']).astype(bool)
        if reference_occ is None:
            reference_occ = occ.copy()
        row = {'name': name, 'mesh': str(mesh_path),
               'raw_mesh_components': raw_components,
               'occupied_voxels': int(occ.sum()),
               'outside_fraction': float((occ & ~env).sum() / max(1, int(occ.sum()))),
               'seat_bc_coverage': float((occ & spec['load']).sum() / spec['load'].sum()),
               'voxel_iou_to_baseline': float((occ & reference_occ).sum() / max(1, int((occ | reference_occ).sum()))),
               'surface': surface_metrics(mesh)}
        rows.append(row)
        for col, (elev, azim, view_center, extent) in enumerate((
            (15, 0, center, .56), (15, 90, center, .56), (10, 15, np.asarray([0., .17, .77]), .20),
        )):
            eye, up = camera_from_elev_azim(view_center, 2.0, elev, azim)
            image = render_lit(mesh, eye, view_center, up, size=330,
                               fit_extent=extent, color=(.62, .66, .7))
            canvas.paste(Image.fromarray(image).convert('RGB'),
                         (col * 330, i * 368 + 38))
            draw.text((col * 330 + 8, i * 368 + 9),
                      f'{name} / {("front", "side", "backrest detail")[col]}', fill='#17212b')
        print(name, 'back p90', row['surface']['backrest']['p90_dihedral_radians'],
              'arm p90', row['surface']['armrests']['p90_dihedral_radians'], flush=True)
    prefix = f'{args.stem}_' if args.stem else ''
    (OUT / f'{prefix}metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    canvas.save(OUT / f'{prefix}comparison.png')


if __name__ == '__main__':
    main()
