#!/usr/bin/env python3
"""Render and measure matched tapered chair generations."""
from __future__ import annotations

import html
import json
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import trimesh

from make_chair_domain import ROOT
from report_chair_sparse_wrinkle_pilot_2026_10_03 import surface_metrics

sys.path.insert(0, str(ROOT / 'codebase/code'))
sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'tapered_examples_2026-10-03'
SPEC = BASE / 'single_view_spec_2026-10-03'
PREVIOUS = BASE / 'sparse_wrinkle_pilot_2026-10-03'
NAMES = ('tapered_original', 'tapered_curved', 'tapered_diagonal', 'tapered_tall',
         'tapered_seed43', 'tapered_seed44')


def main() -> None:
    spec = np.load(SPEC / 'voxel.npz')
    envelope = spec['bracket'].astype(bool)
    registration = json.loads((SPEC / 'specification.json').read_text())['native_frame_registration']
    rotation = Rotation.from_euler('x', registration['rotation_x_degrees'], degrees=True).as_matrix()
    source_center = np.asarray(registration['source_center_m'])
    dest_center = np.asarray(registration['physical_center_m'])
    final = OUT / 'final_meshes'
    final.mkdir(parents=True, exist_ok=True)
    rows = []
    occupancies = {}
    image_cards = []
    for name in NAMES:
        case = OUT / name
        case.mkdir(parents=True, exist_ok=True)
        if name == 'tapered_original':
            input_image = BASE / 'text_latent_qd_2026-10-03/images/tapered__straight__standard.png'
            mesh = trimesh.load(PREVIOUS / 'final_meshes/tapered_straight.obj',
                                force='mesh', process=False)
            components = 1
            shutil.copy2(input_image, case / 'input.png')
        else:
            if name.startswith('tapered_seed'):
                shutil.copy2(BASE / 'text_latent_qd_2026-10-03/images/tapered__straight__standard.png',
                             case / 'input.png')
            input_image = case / 'input.png'
            raw = trimesh.load(case / 'generation/mesh.obj', force='mesh', process=False)
            pieces = raw.split(only_watertight=False)
            components = len(pieces)
            mesh = max(pieces, key=lambda p: abs(p.volume))
            mesh.vertices = ((mesh.vertices - source_center) @ rotation.T) * registration['uniform_scale'] + dest_center
        mesh_path = final / f'{name}.obj'
        mesh.export(mesh_path)
        mesh.export(final / f'{name}.glb')
        occ = voxel_centers_inside(mesh, 64, spec['origin'], spec['pitch_xyz']).astype(bool)
        occupancies[name] = occ
        seat_xy = np.any(spec['load'], axis=2)
        load_indices = np.argwhere(spec['load'])
        seat_candidates = np.argwhere(occ & seat_xy[:, :, None] &
                                      (np.arange(occ.shape[2])[None, None, :] < load_indices[:, 2].min()))
        seat_top_z = float((spec['origin'][2] + (seat_candidates[:, 2].max() + .5) * spec['pitch_xyz'][2]))
        load_bottom_z = float((spec['origin'][2] + (load_indices[:, 2].min() + .5) * spec['pitch_xyz'][2]))
        row = {
            'name': name, 'input_image': str(input_image), 'mesh': str(mesh_path),
            'raw_components': components, 'selected_components': len(mesh.split(only_watertight=False)),
            'watertight': bool(mesh.is_watertight), 'occupied_voxels_64': int(occ.sum()),
            'outside_envelope_fraction': float((occ & ~envelope).sum() / max(1, occ.sum())),
            'seat_bc_coverage': float((occ & spec['load']).sum() / spec['load'].sum()),
            'back_bc_coverage': float((occ & spec['back_load']).sum() / spec['back_load'].sum()),
            'seat_vertical_gap_mm': (0.0 if (occ & spec['load']).any() else
                                     max(0., 1000 * (load_bottom_z - seat_top_z))),
            'surface': surface_metrics(mesh),
        }
        rows.append(row)
        canvas = Image.new('RGB', (1080, 370), '#f5f7fa')
        draw = ImageDraw.Draw(canvas)
        with Image.open(input_image) as source_image:
            source_image.convert('RGB').resize((310, 310), Image.Resampling.LANCZOS).save(case / 'input_preview.png')
        center = np.asarray([0., .01, .46])
        for i, (elev, azim, target, extent) in enumerate((
            (15, 0, center, .56), (15, 90, center, .56),
            (10, 15, np.asarray([0., .17, .77]), .20))):
            eye, up = camera_from_elev_azim(target, 2.0, elev, azim)
            rgb = render_lit(mesh, eye, target, up, size=350, fit_extent=extent,
                             color=(.62, .66, .7))
            canvas.paste(Image.fromarray(rgb).convert('RGB'), (i * 360, 18))
            draw.text((i * 360 + 8, 348), ('front', 'side', 'back detail')[i], fill='#17212b')
        canvas.save(case / 'mesh_preview.png')
        image_cards.append((name, case, row))
        print(name, row['watertight'], row['occupied_voxels_64'], flush=True)
    baseline = occupancies['tapered_original']
    for row in rows:
        occ = occupancies[row['name']]
        row['iou_to_original'] = float((occ & baseline).sum() / max(1, (occ | baseline).sum()))
    (OUT / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    cards = []
    for name, case, row in image_cards:
        cards.append(f'''<article><h2>{html.escape(name)}</h2><div class="visual">
        <img class="input" src="{name}/input_preview.png" alt="input image">
        <img class="mesh" src="{name}/mesh_preview.png" alt="mesh front side detail"></div>
        <p>64³ occupied {row['occupied_voxels_64']:,} · original IoU {row['iou_to_original']:.3f} ·
        seat/back BC {row['seat_bc_coverage']:.0%}/{row['back_bc_coverage']:.0%} ·
        seat vertical gap {row['seat_vertical_gap_mm']:.1f} mm ·
        envelope outside {row['outside_envelope_fraction']:.2%} ·
        raw components {row['raw_components']} · watertight {row['watertight']}</p>
        <p><a href="final_meshes/{name}.obj">OBJ</a> · <a href="final_meshes/{name}.glb">GLB</a> ·
        <a href="{name}/input.png">input</a></p></article>''')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8">
    <title>Tapered chair examples</title><style>
    body{{font-family:system-ui,sans-serif;color:#14212b;background:#e9edf1;max-width:1480px;margin:auto;padding:24px}}
    h1{{margin-bottom:0}}p{{line-height:1.5}}article{{background:white;padding:18px;margin:22px 0;border-radius:14px}}
    .visual{{display:flex;gap:12px;align-items:center}}img.input{{width:25%;object-fit:contain}}
    img.mesh{{width:72%;object-fit:contain}}a{{color:#195a8d}}@media(max-width:700px){{.visual{{display:block}}
    img.input,img.mesh{{width:100%}}}}</style><h1>Tapered chair examples</h1>
    <p>Six front-image-conditioned Direct3D-S2 chairs; same BC/envelope and dense/sparse settings.
    First four cases use seed 42 with different images; last two keep the original image and use seed 43/44.
    Sparse SDF Gaussian σ=3 with occupied-volume matching. FEA off. All meshes are displayed in physical coordinates.
    Input image and mesh are shown side by side; mesh views use fixed cameras.</p>
    <p><strong>BC check:</strong> the three new image variants miss the seat load surface entirely;
    the seed controls touch the seat load but lose much of the back load contact.
    Treat these as shape exploration, not validated QD elites.</p>
    {''.join(cards)}</html>'''
    (OUT / 'index.html').write_text(page)
    report = ['# Tapered chair examples (2026-10-03)', '',
              'Four input images including the previous tapered baseline, plus two seed controls using the original image. Same physical BC, envelope, and generation parameters; seed 42 for image variants, seeds 43 and 44 for controls. The new cases use dense+sparse Direct3D-S2 and sparse SDF σ=3 with volume matching. No FEA.', '',
              '| Case | 64³ voxels | IoU vs original | seat/back BC | seat vertical gap | outside envelope | components | watertight |',
              '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        report.append(f"| {row['name']} | {row['occupied_voxels_64']} | {row['iou_to_original']:.3f} | {row['seat_bc_coverage']:.0%}/{row['back_bc_coverage']:.0%} | {row['seat_vertical_gap_mm']:.1f} mm | {row['outside_envelope_fraction']:.2%} | {row['raw_components']} | {row['watertight']} |")
    report.extend(['', 'The three image variants change the visual design but their 3D seat is 16–32 mm below the prescribed seat load. All three image-variant meshes therefore fail the seat BC validity check; do not use them as FEA or QD archive elites without interface correction. The single front view constrains the projected leg endpoints, but did not preserve the inferred 3D seat elevation.', ''])
    report.extend(['The two original-image seed controls retain 50% seat contact, but back BC contact drops from 98% to 12% and 40%; changing only the stochastic seed is therefore also insufficient to preserve the structural interfaces.', ''])
    report.extend(['', 'Absolute results:', str(OUT / 'index.html'), str(final), str(OUT / 'metrics.json')])
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n')


if __name__ == '__main__':
    main()
