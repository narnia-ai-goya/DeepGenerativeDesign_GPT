#!/usr/bin/env python3
"""Compare focused negative-space guidance against the chair sparse baseline."""
from __future__ import annotations

import html
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image, ImageDraw
from pysdf import SDF

from make_chair_domain import ROOT
from report_chair_prefea_stages import silhouette_scores

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit

SOURCE = ROOT / 'experiments/chair/sofa_style_2026-09-28/backrest_bc_multi_examples_2026-09-29/diagonal_braced'
BC = ROOT / 'experiments/chair/sofa_style_2026-09-28/consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/joint_focus_projection_2026-09-30'
INPUT = ROOT / 'experiments/chair/image_concepts_2026-09-27/diagonal_braced/multiview_registered/input'
ROWS = (
    ('baseline', SOURCE / 'sparse_pw2_d13/generation/mesh.obj', SOURCE / 'sparse_pw2_d13/bc_audit.json'),
    ('focused void w5', OUT / 'focus_w5/generation/mesh.obj', OUT / 'focus_w5/audit.json'),
    ('focused void w20', OUT / 'focus_w20/generation/mesh.obj', OUT / 'focus_w20/audit.json'),
    ('corrected focus w5', OUT / 'corrected_focus_w5/generation/mesh.obj', OUT / 'corrected_focus_w5/audit.json'),
    ('corrected focus w20', OUT / 'corrected_focus_w20/generation/mesh.obj', OUT / 'corrected_focus_w20/audit.json'),
)
VIEWS = (('oblique', 25, 35), ('front', 15, 0), ('right', 15, 90))
TARGETS = {'front': INPUT / 'v00_front_lo.png', 'right': INPUT / 'v02_right_lo.png'}


def local_counts(mesh, target_data, query):
    sdf = SDF(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    occupancy = (sdf(query) > 0).reshape(64, 64, 64).astype(np.float32)
    counts = {}
    for name in ('front', 'right'):
        camera = torch.from_numpy(target_data[f'camera_grid_{name}'].astype(np.float32))[None]
        projected = F.grid_sample(torch.from_numpy(occupancy)[None, None], camera,
            mode='bilinear', align_corners=True)[0, 0].max(0).values.numpy()
        focus = target_data[f'focus_{name}'].astype(bool)
        counts[name] = {'excess_pixels': int(((projected > .2) & focus).sum()),
                        'focused_empty_pixels': int(focus.sum()),
                        'mean_predicted_occupancy': float(projected[focus].mean())}
    return counts


def main():
    env = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents)) * 1.5
    fit = float(env.extents.max()) / 2
    bc = np.load(BC)
    query = (bc['origin'] + (np.indices((64,64,64)).reshape(3,-1).T + .5) *
             bc['pitch_xyz']).astype(np.float32)
    targets = np.load(OUT / 'joint_focus_targets.npz')
    rows = [row for row in ROWS if row[1].exists()]
    size, header = 400, 34
    sheet = Image.new('RGB', (size * len(VIEWS), (size + header) * len(rows)), 'white')
    draw = ImageDraw.Draw(sheet)
    records = []
    for i, (name, path, audit_path) in enumerate(rows):
        mesh = trimesh.load(path, force='mesh')
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        rec = {'name': name, 'mesh': str(path),
               'bc_pass': audit.get('bc_geometry_pass'),
               'shape_pass': audit.get('shape_gate_pass'),
               'components': audit.get('mesh_components'),
               'local_empty_space': local_counts(mesh, targets, query),
               'silhouettes': {}}
        for j, (view, elev, azim) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            render = Image.fromarray(render_lit(mesh, eye, center, up,
                size=size, fit_extent=fit, margin=1.15, color=(.47, .52, .56))).convert('RGB')
            x, y = j * size, i * (size + header)
            sheet.paste(render, (x, y + header))
            draw.text((x + 8, y + 9), f'{name}: {view}', fill='#20252b')
            if view in TARGETS:
                target = Image.open(TARGETS[view]).convert('RGB').resize((size, size))
                rec['silhouettes'][view] = silhouette_scores(render, target)
        records.append(rec)
    sheet.save(OUT / 'comparison.png')
    (OUT / 'metrics.json').write_text(json.dumps(records, indent=2) + '\n')
    detail = Image.new('RGB', (1200, 265 * len(rows)), 'white')
    dd = ImageDraw.Draw(detail)
    for i, (name, _, _) in enumerate(rows):
        for j in range(3):
            y0 = i * (size + header) + header
            crop = sheet.crop((j * size + 50, y0 + 145, j * size + 350, y0 + 345))
            detail.paste(crop.resize((400, 250), Image.Resampling.LANCZOS),
                         (j * 400, i * 265 + 15))
            dd.text((j * 400 + 5, i * 265 + 2), f'{name}: {VIEWS[j][0]}', fill='#20252b')
    detail.save(OUT / 'joint_detail.png')
    table = ''.join('<tr><td>' + html.escape(r['name']) + '</td><td><a href="/' +
        r['mesh'].removeprefix(str(ROOT) + '/') + '">OBJ</a></td><td>' +
        str(r['bc_pass']) + '</td><td>' + str(r['components']) + '</td><td>' +
        '/'.join(f'{r["silhouettes"][v]["iou"]:.3f}' for v in ('front', 'right')) +
        '</td><td>' + '/'.join(str(r['local_empty_space'][v]['excess_pixels']) for v in ('front','right')) +
        '</td></tr>' for r in records)
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair local negative-space guidance</title>
<style>body{font:16px system-ui;margin:30px;background:#f4f6f8;color:#18232d}img{max-width:100%}table{border-collapse:collapse;background:white}td,th{border:1px solid #c8d0da;padding:8px}a{color:#075f9e}</style>
<h1>Chair A/B local negative-space guidance</h1><p>Same dense cache, seed, sparse thickness, BC and envelope. Only image projection guidance changes. Orange pixels mark selected image-empty regions in front/right registered 64px projection targets. FEA off.</p>
<a href="focus_pixels.png"><img src="focus_pixels.png" width="384"></a>
<h2>Full renders</h2><a href="comparison.png"><img src="comparison.png"></a><h2>Joint detail</h2><a href="joint_detail.png"><img src="joint_detail.png"></a>
<table><tr><th>Variant</th><th>OBJ</th><th>BC pass</th><th>Components</th><th>Front/right IoU</th><th>Excess focus pixels F/R</th></tr>''' + table + '</table></html>'
    (OUT / 'index.html').write_text(page)
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
