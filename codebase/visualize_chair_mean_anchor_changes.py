#!/usr/bin/env python3
"""Show exactly which projected pixels change under sparse SDF mean pooling."""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image, ImageDraw
from pysdf import SDF

from make_chair_domain import ROOT
from report_chair_joint_focus_projection import BC

OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/sparse_mean_anchor_2026-10-01'
FOCUS = ROOT / 'experiments/chair/sofa_style_2026-09-28/joint_focus_projection_2026-09-30/joint_focus_targets.npz'
SOURCE = ROOT / 'experiments/chair/sofa_style_2026-09-28/backrest_bc_multi_examples_2026-09-29/diagonal_braced'
CASES = (
    ('baseline', SOURCE / 'sparse_pw2_d13/generation/mesh.obj'),
    ('mean anchor w10', ROOT / 'experiments/chair/sofa_style_2026-09-28/sparse_pooling_2026-10-01/anchor_long/generation/mesh.obj'),
    ('mean anchor w50', OUT / 'anchor_w50_k4/generation/mesh.obj'),
    ('mean anchor w200', OUT / 'anchor_w200_k4/generation/mesh.obj'),
    ('mean anchor w100 K8', OUT / 'anchor_w100_k8/generation/mesh.obj'),
)


def project(mesh_path, query, target):
    mesh = trimesh.load(mesh_path, force='mesh')
    sdf = SDF(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    occ = (sdf(query) > 0).reshape(64, 64, 64).astype(np.float32)
    result = {}
    for name in ('front', 'right'):
        camera = torch.from_numpy(target[f'camera_grid_{name}'].astype(np.float32))[None]
        projected = F.grid_sample(torch.from_numpy(occ)[None, None], camera,
                                  mode='bilinear', align_corners=True)[0, 0].max(0).values.numpy()
        result[name] = projected > .2
    return occ.astype(bool), result


def main():
    bc = np.load(BC)
    query = (bc['origin'] + (np.indices((64, 64, 64)).reshape(3, -1).T + .5)
             * bc['pitch_xyz']).astype(np.float32)
    target = np.load(FOCUS)
    projected = {}
    occupancy = {}
    for name, path in CASES:
        occupancy[name], projected[name] = project(path, query, target)

    tile, header = 384, 56
    variants = CASES[1:]
    sheet = Image.new('RGB', (tile * 2, (tile + header) * len(variants)), 'white')
    draw = ImageDraw.Draw(sheet)
    records = []
    for row, (name, _) in enumerate(variants):
        change_voxels = occupancy['baseline'] != occupancy[name]
        row_rec = {'name': name, 'changed_64cube_voxels': int(change_voxels.sum()),
                   'changed_64cube_fraction': float(change_voxels.mean()), 'views': {}}
        for col, view in enumerate(('front', 'right')):
            base = projected['baseline'][view]
            candidate = projected[name][view]
            add = ~base & candidate
            remove = base & ~candidate
            focus = target[f'focus_{view}'].astype(bool)
            arr = np.full((64, 64, 3), 255, np.uint8)
            arr[base] = (117, 126, 134)
            arr[add] = (235, 74, 61)       # red: added solid
            arr[remove] = (42, 169, 187)   # cyan: removed solid
            arr[focus & ~base & ~add] = (255, 223, 168)
            tile_image = Image.fromarray(arr).resize((tile, tile), Image.Resampling.NEAREST)
            x, y = col * tile, row * (tile + header)
            sheet.paste(tile_image, (x, y + header))
            draw.text((x + 8, y + 7), f'{name} | {view}', fill='#18232d')
            draw.text((x + 8, y + 27), f'removed {remove.sum()} px, added {add.sum()} px; focus changed {((remove | add) & focus).sum()} px', fill='#34495e')
            row_rec['views'][view] = {
                'removed_pixels': int(remove.sum()), 'added_pixels': int(add.sum()),
                'changed_focused_pixels': int(((remove | add) & focus).sum()),
                'total_pixels': 4096,
            }
        records.append(row_rec)
    sheet.save(OUT / 'projection_change_maps.png')
    (OUT / 'change_metrics.json').write_text(json.dumps(records, indent=2) + '\n')
    index = OUT / 'index.html'
    page = index.read_text()
    section = '''<h2>Projected differences from baseline</h2><p>Gray: shared baseline material. Cyan: removed material. Red: added material. Pale orange: designer-marked empty focus pixels. Projection computed at 64×64, enlarged without interpolation.</p><a href="projection_change_maps.png"><img src="projection_change_maps.png"></a>'''
    page = page.replace('</html>', section + '</html>')
    index.write_text(page)
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
