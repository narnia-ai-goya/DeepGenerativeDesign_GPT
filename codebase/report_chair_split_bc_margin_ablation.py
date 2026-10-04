#!/usr/bin/env python3
"""Render sparse chair joints for foot/load BC margin separation."""
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
OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/split_bc_margin_2026-09-30'
INPUT = ROOT / 'experiments/chair/image_concepts_2026-09-27/diagonal_braced/multiview_registered/input'
ROWS = (
    ('fix 13 / load 13 mm', SOURCE / 'sparse_pw2_d13/generation/mesh.obj', SOURCE / 'sparse_pw2_d13/bc_audit.json'),
    ('fix 13 / load 6 mm', OUT / 'load_6mm/generation/mesh.obj', OUT / 'load_6mm/audit.json'),
    ('fix 13 / load 0 mm', OUT / 'load_0mm/generation/mesh.obj', OUT / 'load_0mm/audit.json'),
)
VIEWS = (('oblique', 25, 35), ('front', 15, 0), ('right', 15, 90), ('top', 85, 0))
TARGETS = {'front': INPUT / 'v00_front_lo.png',
           'right': INPUT / 'v02_right_lo.png', 'top': INPUT / 'v_top.png'}


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
    for i, (name, path, audit_path) in enumerate(rows):
        mesh = trimesh.load(path, force='mesh')
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
        record = {'name': name, 'mesh': str(path),
                  'bc_pass': audit.get('bc_geometry_pass'),
                  'shape_pass': audit.get('shape_gate_pass'),
                  'components': audit.get('mesh_components'),
                  'region_coverage': {k: v.get('coverage') for k, v in audit.get('regions', {}).items()},
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
                record['silhouettes'][view] = silhouette_scores(render, target)
        records.append(record)
    sheet.save(OUT / 'comparison.png')
    (OUT / 'metrics.json').write_text(json.dumps(records, indent=2) + '\n')
    detail = Image.new('RGB', (1200, 265 * len(rows)), 'white')
    dd = ImageDraw.Draw(detail)
    for i, (name, _, _) in enumerate(rows):
        for j, source_col in enumerate((0, 1, 2)):
            y0 = i * (size + header) + header
            crop = sheet.crop((source_col * size + 50, y0 + 145,
                               source_col * size + 350, y0 + 345))
            detail.paste(crop.resize((400, 250), Image.Resampling.LANCZOS),
                         (j * 400, i * 265 + 15))
            dd.text((j * 400 + 5, i * 265 + 2),
                    f'{name}: {("oblique", "front", "right")[j]}', fill='#20252b')
    detail.save(OUT / 'joint_detail.png')
    table = ''.join('<tr><td>' + html.escape(r['name']) + '</td><td><a href="/' +
        r['mesh'].removeprefix(str(ROOT) + '/') + '">OBJ</a></td><td>' +
        str(r['bc_pass']) + '</td><td>' + str(r['components']) + '</td><td>' +
        '/'.join(f'{r["silhouettes"][v]["iou"]:.3f}' for v in ('front', 'right', 'top')) +
        '</td></tr>' for r in records)
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair split BC margin study</title>
<style>body{font:16px system-ui;margin:30px;background:#f4f6f8;color:#18232d}img{max-width:100%}table{border-collapse:collapse;background:white}td,th{border:1px solid #c8d0da;padding:8px}a{color:#075f9e}</style>
<h1>Chair split BC margin study</h1><p>Same dense cache, seed, 30 sparse steps, BC geometry and envelope. Only the load-STL dilation changes; all four foot fixed regions retain 13 mm. FEA off.</p>
<h2>3D renders</h2><a href="comparison.png"><img src="comparison.png"></a><h2>Seat joints</h2><a href="joint_detail.png"><img src="joint_detail.png"></a>
<table><tr><th>Variant</th><th>OBJ</th><th>BC pass</th><th>Components</th><th>Front/right/top IoU</th></tr>''' + table + '</table></html>'
    (OUT / 'index.html').write_text(page)
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
