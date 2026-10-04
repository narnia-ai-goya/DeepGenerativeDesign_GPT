#!/usr/bin/env python3
"""Render a controlled chair image-swap comparison with the established recipe."""
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
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402

OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/alternative_image_2026-10-01'
NEW = OUT / 'complex_truss_armchair'
OLD = ROOT / 'experiments/chair/sofa_style_2026-09-28/backrest_bc_multi_examples_2026-09-29/diagonal_braced'
NEW_INPUT = ROOT / 'experiments/chair/image_concepts_2026-09-27/complex_truss_armchair/multiview_registered'
OLD_INPUT = ROOT / 'experiments/chair/image_concepts_2026-09-27/diagonal_braced/multiview_registered'
VIEWS = (('oblique', 25, 35), ('front', 15, 0), ('right', 15, 90), ('top', 85, 0))
STEMS = {'front': 'v00_front_lo', 'right': 'v02_right_lo', 'top': 'v_top'}


def run():
    cases = (
        ('Original image / dense', OLD / 'dense_pw2/mesh_dense.obj', OLD / 'dense_pw2_bc_audit.json', OLD_INPUT),
        ('Original image / sparse', OLD / 'sparse_pw2_d13/generation/mesh.obj', OLD / 'sparse_pw2_d13/bc_audit.json', OLD_INPUT),
        ('New truss image / prototype', NEW / 'prototype.obj', None, NEW_INPUT),
        ('New truss image / dense', NEW / 'dense_pw2/mesh_dense.obj', NEW / 'dense_pw2_bc_audit.json', NEW_INPUT),
        ('New truss image / sparse', NEW / 'sparse_pw2_d13/generation/mesh.obj', NEW / 'sparse_pw2_d13/bc_audit.json', NEW_INPUT),
    )
    cases = [item for item in cases if item[1].exists()]
    env = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents)) * 1.5
    scale = float(env.extents.max()) / 2
    size, header = 340, 35
    sheet = Image.new('RGB', (size * len(VIEWS), (size + header) * len(cases)), 'white')
    draw = ImageDraw.Draw(sheet)
    records = []
    for row, (label, path, audit_path, source) in enumerate(cases):
        mesh = trimesh.load(path, force='mesh')
        audit = json.loads(audit_path.read_text()) if audit_path and audit_path.exists() else {}
        record = {
            'stage': label, 'mesh': str(path),
            'bc_pass': audit.get('bc_geometry_pass'),
            'components': audit.get('mesh_components', len(mesh.split())),
            'largest_component_volume_fraction': audit.get('largest_component_volume_fraction'),
            'top_bar_coverage': audit.get('top_bar', {}).get('coverage'),
            'silhouette_iou': {},
        }
        for col, (view, elev, azim) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rendered = Image.fromarray(render_lit(mesh, eye, center, up, size=size,
                fit_extent=scale, margin=1.15, color=(.48, .52, .56))).convert('RGB')
            sheet.paste(rendered, (col * size, row * (size + header) + header))
            draw.text((col * size + 8, row * (size + header) + 9), f'{label} / {view}', fill='#20252b')
            if view in STEMS:
                target = Image.open(source / 'input' / f'{STEMS[view]}.png').convert('RGB')
                record['silhouette_iou'][view] = silhouette_scores(rendered, target.resize((size, size)))['iou']
        records.append(record)
    OUT.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT / 'comparison.png')
    (OUT / 'metrics.json').write_text(json.dumps(records, indent=2) + '\n')

    def rel(path):
        return '/' + str(path.relative_to(ROOT))

    table = []
    for row in records:
        value = lambda key: '—' if row[key] is None else str(row[key])
        iou = row['silhouette_iou']
        table.append('<tr><td><a href="' + rel(ROOT / row['mesh']) + '">' + html.escape(row['stage']) + '</a></td>'
            + ''.join(f'<td>{value(k)}</td>' for k in ('bc_pass', 'components', 'top_bar_coverage'))
            + ''.join(f'<td>{iou.get(v, 0):.3f}</td>' for v in STEMS) + '</tr>')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair image swap</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:2rem auto;background:#f4f6f8;color:#1d2730}}
article{{background:white;padding:1.2rem;margin:1rem 0;border-radius:12px}}img{{max-width:100%}}
table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:.45rem}}</style>
<article><h1>의자 입력 이미지 변경: 동일 BC와 dense→sparse 설정</h1>
<p>새 입력은 대각 보강재와 팔걸이가 분명한 complex-truss 의자다. 기존 diagonal-braced와 동일한
BC, envelope, dense pw2, sparse d13 레시피를 사용했다. FEA는 꺼져 있다.</p>
<h2>입력 이미지</h2><p>기존 diagonal-braced</p><img src="{rel(OLD_INPUT / 'input_contact.png')}">
<p>새 complex-truss</p><img src="{rel(NEW_INPUT / 'input_contact.png')}"></article>
<article><h2>실제 3D 메시 렌더</h2><img src="comparison.png"><p>각 행의 이름을 누르면 OBJ를 연다.</p></article>
<article><h2>형상 및 BC 수치</h2><table><tr><th>단계 / OBJ</th><th>BC 통과</th><th>성분 수</th>
<th>등받이 상부 커버리지</th><th>앞 IoU</th><th>오른쪽 IoU</th><th>위 IoU</th></tr>{''.join(table)}</table>
<p><a href="metrics.json">수치 JSON</a></p></article></html>'''
    (OUT / 'index.html').write_text(page)
    print(OUT / 'index.html')


if __name__ == '__main__':
    run()
