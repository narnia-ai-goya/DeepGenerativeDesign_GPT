#!/usr/bin/env python3
"""Render the same chair at dense, sparse-decoder, and refiner stages."""
from __future__ import annotations

import html
import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from make_chair_domain import ROOT
from report_chair_prefea_stages import silhouette_scores

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit

SOURCE = ROOT / 'experiments/chair/sofa_style_2026-09-28/backrest_bc_multi_examples_2026-09-29/diagonal_braced'
OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/joint_stage_diagnostic_2026-09-30'
INPUT = ROOT / 'experiments/chair/image_concepts_2026-09-27/diagonal_braced/multiview_registered/input'
ROWS = (
    ('dense input to sparse', SOURCE / 'dense_pw2/mesh_dense.obj'),
    ('guided: pre-refiner', OUT / 'guided/generation/mesh_pre_refiner_world.obj'),
    ('guided: final', OUT / 'guided/generation/mesh.obj'),
    ('unguided: pre-refiner', OUT / 'unguided/generation/mesh_pre_refiner_world.obj'),
    ('unguided: final', OUT / 'unguided/generation/mesh.obj'),
)
VIEWS = (('oblique', 25, 35), ('front', 15, 0), ('right', 15, 90))
TARGETS = {'front': INPUT / 'v00_front_lo.png', 'right': INPUT / 'v02_right_lo.png'}


def main():
    env = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents)) * 1.5
    fit = float(env.extents.max()) / 2
    rows = [row for row in ROWS if row[1].exists()]
    size, header = 400, 34
    sheet = Image.new('RGB', (size * len(VIEWS), (size + header) * len(rows)), 'white')
    draw = ImageDraw.Draw(sheet)
    records = []
    for i, (name, path) in enumerate(rows):
        mesh = trimesh.load(path, force='mesh')
        rec = {'name': name, 'mesh': str(path), 'bounds': mesh.bounds.tolist(), 'silhouettes': {}}
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
    sheet.save(OUT / 'stage_comparison.png')
    (OUT / 'metrics.json').write_text(json.dumps(records, indent=2) + '\n')
    detail = Image.new('RGB', (1200, 265 * len(rows)), 'white')
    dd = ImageDraw.Draw(detail)
    for i, (name, _) in enumerate(rows):
        for j in range(3):
            y0 = i * (size + header) + header
            crop = sheet.crop((j * size + 50, y0 + 145, j * size + 350, y0 + 345))
            detail.paste(crop.resize((400, 250), Image.Resampling.LANCZOS),
                         (j * 400, i * 265 + 15))
            dd.text((j * 400 + 5, i * 265 + 2),
                    f'{name}: {VIEWS[j][0]}', fill='#20252b')
    detail.save(OUT / 'joint_detail.png')
    table = ''.join('<tr><td>' + html.escape(r['name']) + '</td><td><a href="/' +
        r['mesh'].removeprefix(str(ROOT) + '/') + '">OBJ</a></td><td>' +
        '/'.join(f'{r["silhouettes"][v]["iou"]:.3f}' for v in ('front', 'right')) +
        '</td></tr>' for r in records)
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair joint stage diagnostic</title>
<style>body{font:16px system-ui;margin:30px;background:#f4f6f8;color:#18232d}img{max-width:100%}table{border-collapse:collapse;background:white}td,th{border:1px solid #c8d0da;padding:8px}a{color:#075f9e}</style>
<h1>Chair A/B joint stage diagnostic</h1><p>Same dense cache, seed, 30 sparse steps, 13 mm BC and envelope. Guided uses sp_guide_w=10; unguided uses zero. Both include the original BC hard constraint and use FEA off.</p>
<h2>Full renders</h2><a href="stage_comparison.png"><img src="stage_comparison.png"></a><h2>Seat joints</h2><a href="joint_detail.png"><img src="joint_detail.png"></a>
<table><tr><th>Stage</th><th>OBJ</th><th>Front/right IoU</th></tr>''' + table + '</table></html>'
    (OUT / 'index.html').write_text(page)
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
