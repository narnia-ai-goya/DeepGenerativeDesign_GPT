#!/usr/bin/env python3
"""Evaluate whether BC guidance promotes tapered variants to the QD archive."""
from __future__ import annotations

import html
import json
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label
from scipy.spatial.transform import Rotation
import trimesh

from make_chair_domain import ROOT
from build_chair_single_view_qd_archive_2026_10_03 import features, cell_for

sys.path.insert(0, str(ROOT / 'codebase/code'))
sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
NEW = BASE / 'tapered_examples_2026-10-03'
OUT = NEW / 'bc_guidance_2026-10-03'
SPEC = BASE / 'single_view_spec_2026-10-03'


def main() -> None:
    runs = json.loads((OUT / 'runs.json').read_text())
    spec = np.load(SPEC / 'voxel.npz')
    env = spec['bracket'].astype(bool)
    bc = spec['bc'].astype(bool)
    reg = json.loads((SPEC / 'specification.json').read_text())['native_frame_registration']
    rotation = Rotation.from_euler('x', reg['rotation_x_degrees'], degrees=True).as_matrix()
    source_center = np.asarray(reg['source_center_m'])
    dest_center = np.asarray(reg['physical_center_m'])
    six = np.zeros((3, 3, 3), bool)
    six[1, 1, 1] = True
    for axis in range(3):
        for direction in (-1, 1):
            index = [1, 1, 1]
            index[axis] += direction
            six[tuple(index)] = True
    baseline = {r['name']: r for r in json.loads((NEW / 'metrics.json').read_text())}
    rows = []
    cards = []
    for run in runs:
        name = run['name']
        case = OUT / name
        raw = trimesh.load(case / 'generation/mesh.obj', force='mesh', process=False)
        pieces = raw.split(only_watertight=False)
        mesh = max(pieces, key=lambda part: abs(part.volume))
        mesh.vertices = ((mesh.vertices - source_center) @ rotation.T) * reg['uniform_scale'] + dest_center
        mesh_path = case / 'aligned_main.obj'
        mesh.export(mesh_path)
        mesh.export(case / 'aligned_main.glb')
        occ = voxel_centers_inside(mesh, 64, spec['origin'], spec['pitch_xyz']).astype(bool)
        realized = (occ & env) | bc
        repair = int((realized & ~occ).sum() + (occ & ~realized).sum()) / max(1, int(realized.sum()))
        desc = features(realized, spec)
        seat = float((occ & spec['load']).sum() / spec['load'].sum())
        back = float((occ & spec['back_load']).sum() / spec['back_load'].sum())
        outside = float((occ & ~env).sum() / max(1, occ.sum()))
        components = int(label(realized, structure=six)[1])
        geometry_gate = (seat >= .5 and back >= .9 and repair <= .05 and outside <= .01
                         and components == 1 and mesh.is_watertight and cell_for(desc) is not None)
        source_name = run['source']
        old = trimesh.load(NEW / 'final_meshes' / f'{source_name}.obj', force='mesh', process=False)
        old_occ = voxel_centers_inside(old, 64, spec['origin'], spec['pitch_xyz']).astype(bool)
        iou = float((occ & old_occ).sum() / max(1, (occ | old_occ).sum()))
        row = {'name': name, 'source': source_name, 'mesh': str(mesh_path),
               'raw_components': len(pieces), 'components_after_repair': components,
               'watertight': bool(mesh.is_watertight), 'seat_bc': seat, 'back_bc': back,
               'outside_fraction': outside, 'repair_fraction': repair,
               'descriptor': {'side_open_fraction': desc[0], 'backrest_taper_ratio': desc[1]},
               'cell': cell_for(desc), 'occupied_voxels_64': int(occ.sum()),
               'mass_liters': float(realized.sum() * np.prod(spec['pitch_xyz']) * 1000),
               'iou_to_unguided': iou, 'geometry_gate': bool(geometry_gate)}
        rows.append(row)
        canvas = Image.new('RGB', (720, 350), '#F4F7F8')
        draw = ImageDraw.Draw(canvas)
        center = np.asarray([0., .01, .46])
        for i, (elev, azim) in enumerate(((15, 0), (15, 90))):
            eye, up = camera_from_elev_azim(center, 2.0, elev, azim)
            rgb = render_lit(mesh, eye, center, up, size=350, fit_extent=.56,
                             color=(.62, .66, .7))
            canvas.paste(Image.fromarray(rgb).convert('RGB'), (i * 360, 0))
            draw.text((i * 360 + 10, 327), ('front', 'side')[i], fill='#1E303A')
        canvas.save(case / 'preview.png')
        cards.append(f'''<article><h2>{html.escape(name)}</h2><img src="{name}/preview.png">
          <p>seat/back BC {seat:.0%}/{back:.0%} · repair {repair:.1%} · envelope outside {outside:.1%}
          · geometry gate {'PASS' if geometry_gate else 'FAIL'} · IoU to unguided {iou:.3f}</p>
          <a href="{name}/aligned_main.obj">OBJ</a> · <a href="{name}/aligned_main.glb">GLB</a></article>''')
        print(name, 'BC', round(seat, 3), round(back, 3), 'repair', round(repair, 3),
              'gate', geometry_gate, flush=True)
    (OUT / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    (OUT / 'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8">
    <title>Tapered chair BC-guidance ablation</title><style>body{font-family:system-ui,sans-serif;
    max-width:1300px;margin:auto;padding:24px;background:#edf2f4;color:#182a34}article{background:white;
    padding:16px;margin:18px 0;border-radius:12px}img{max-width:100%;display:block}a{color:#1c6688}</style>
    <h1>Tapered chair BC-guidance ablation</h1><p>Same input image and seed per pair;
    dense/sparse BC weights are varied. FEA is only run on geometry-gate passes.</p>'''
    + ''.join(cards) + '</html>')
    lines = ['# Tapered chair BC-guidance ablation', '',
             '| Run | Seat/back BC | Repair | Outside | Cell | Gate | IoU to unguided |',
             '|---|---:|---:|---:|---|---|---:|']
    for row in rows:
        lines.append(f"| {row['name']} | {row['seat_bc']:.0%}/{row['back_bc']:.0%} | {row['repair_fraction']:.1%} | {row['outside_fraction']:.1%} | {row['cell']} | {row['geometry_gate']} | {row['iou_to_unguided']:.3f} |")
    lines.extend(['', f'Viewer: {OUT / "index.html"}', f'Metrics: {OUT / "metrics.json"}'])
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
