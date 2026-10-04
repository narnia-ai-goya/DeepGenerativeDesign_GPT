#!/usr/bin/env python3
"""Evaluate raw dense chair tokens and render them in the shared physical frame."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label
from scipy.spatial.transform import Rotation
import trimesh

from make_chair_domain import ROOT
sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SOURCE = BASE / 'text_latent_qd_2026-10-03/mesh_cases'
OUT = BASE / 'envelope_excess_pilot_2026-10-03'
SPEC = BASE / 'single_view_spec_2026-10-03'


def occupancy(cache: Path) -> np.ndarray:
    idx = np.load(cache)['latent_index'][:, 1:]
    arr = np.zeros((64, 64, 64), bool)
    arr[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return arr


def main() -> None:
    native = np.load(SPEC / 'native_frame_spec.npz')
    env = native['bracket'].astype(bool)
    transform = json.loads((SPEC / 'specification.json').read_text())['native_frame_registration']
    R = Rotation.from_euler('x', transform['rotation_x_degrees'], degrees=True).as_matrix()
    src = np.asarray(transform['source_center_m'])
    dst = np.asarray(transform['physical_center_m'])
    scale = transform['uniform_scale']
    structure = np.zeros((3, 3, 3), bool)
    structure[1, 1, :] = True
    structure[1, :, 1] = True
    structure[:, 1, 1] = True
    rows = []
    cases = [('feasible baseline', SOURCE / 'tapered__straight__standard'),
             ('feasible eta3', OUT / 'tapered__straight__standard__eta3'),
             ('outside baseline', SOURCE / 'faceted__curved__standard'),
             ('outside eta3', OUT / 'faceted__curved__standard__eta3'),
             ('outside eta10', OUT / 'faceted__curved__standard__eta10')]
    tile = 340
    canvas = Image.new('RGB', (tile * 2, len(cases) * (tile + 34)), 'white')
    draw = ImageDraw.Draw(canvas)
    center = np.asarray([0., 0.01, .46])
    radius = 2.0
    for row_i, (name, path) in enumerate(cases):
        occ = occupancy(path / 'dense_cache.npz')
        baseline_path = SOURCE / ('tapered__straight__standard' if name.startswith('feasible') else 'faceted__curved__standard')
        base = occupancy(baseline_path / 'dense_cache.npz')
        outside = occ & ~env
        _, components = label(occ, structure=structure)
        mesh = trimesh.load(path / 'generation/mesh_dense_raw.obj', force='mesh', process=False)
        mesh.vertices = ((mesh.vertices - src) @ R.T) * scale + dst
        row = {
            'name': name, 'raw_dense_mesh': str(path / 'generation/mesh_dense_raw.obj'),
            'active_tokens': int(occ.sum()), 'outside_tokens': int(outside.sum()),
            'outside_fraction': float(outside.sum() / max(1, int(occ.sum()))),
            'dense_token_iou_to_same_image_baseline': float((occ & base).sum() / max(1, (occ | base).sum())),
            'six_connected_components': int(components),
        }
        rows.append(row)
        for col, (elev, azim) in enumerate(((15, 0), (15, 90))):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rgba = render_lit(mesh, eye, center, up, size=tile,
                              fit_extent=.56, color=(.62, .66, .7))
            canvas.paste(Image.fromarray(rgba).convert('RGB'),
                         (col * tile, row_i * (tile + 34) + 34))
            draw.text((col * tile + 8, row_i * (tile + 34) + 8),
                      f'{name} / {"front" if col == 0 else "side"}', fill='#17212b')
    (OUT / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    canvas.save(OUT / 'raw_dense_comparison.png')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
