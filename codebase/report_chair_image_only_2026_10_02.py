#!/usr/bin/env python3
"""Compare the straight-corridor and image-only chair runs using 3D renders."""
from __future__ import annotations

import argparse
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

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OLD = BASE / 'alternative_image_2026-10-01/complex_truss_armchair'
INPUT = ROOT / 'experiments/chair/image_concepts_2026-09-27/complex_truss_armchair/multiview_registered'
VIEWS = (('oblique', 25, 35, None), ('front', 15, 0, 'v00_front_lo'),
         ('right', 15, 90, 'v02_right_lo'), ('top', 85, 0, 'v_top'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('image_only_bc_2026-10-02', 'front_support_only_2026-10-02'),
                        default='image_only_bc_2026-10-02')
    args = parser.parse_args()
    new = BASE / args.variant / 'complex_truss_armchair'
    out = new.parent
    label = 'No corridor' if args.variant.startswith('image_only') else 'Front support only'
    entries = (
        ('Old / dense', OLD / 'dense_pw2/mesh_dense.obj', OLD / 'dense_pw2_bc_audit.json'),
        ('Old / sparse', OLD / 'sparse_pw2_d13/generation/mesh.obj', OLD / 'sparse_pw2_d13/bc_audit.json'),
        (label + ' / prototype', new / 'prototype.obj', None),
        (label + ' / dense', new / 'dense_pw2/mesh_dense.obj', new / 'dense_pw2_bc_audit.json'),
        (label + ' / sparse', new / 'sparse_pw2_d13/generation/mesh.obj', new / 'sparse_pw2_d13/bc_audit.json'),
    )
    entries = [row for row in entries if row[1].exists()]
    env = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents)) * 1.5
    scale = float(env.extents.max()) / 2
    size, header = 340, 34
    sheet = Image.new('RGB', (size * len(VIEWS), (size + header) * len(entries)), 'white')
    draw = ImageDraw.Draw(sheet)
    metrics = []
    for row, (label, path, audit_path) in enumerate(entries):
        mesh = trimesh.load(path, force='mesh')
        audit = json.loads(audit_path.read_text()) if audit_path and audit_path.exists() else {}
        item = {'stage': label, 'mesh': str(path), 'components': audit.get('mesh_components', len(mesh.split())),
                'bc_pass': audit.get('bc_geometry_pass'),
                'largest_component_volume_fraction': audit.get('largest_component_volume_fraction'),
                'top_bar_coverage': audit.get('top_bar', {}).get('coverage'), 'iou': {}}
        for col, (view, elev, azim, stem) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rendered = Image.fromarray(render_lit(mesh, eye, center, up, size=size,
                fit_extent=scale, margin=1.15, color=(.48, .52, .56))).convert('RGB')
            sheet.paste(rendered, (col * size, row * (size + header) + header))
            draw.text((col * size + 8, row * (size + header) + 8), label + ' / ' + view, fill='#20252b')
            if stem:
                target = Image.open(INPUT / 'input' / (stem + '.png')).convert('RGB').resize((size, size))
                item['iou'][view] = silhouette_scores(rendered, target)['iou']
        metrics.append(item)
    out.mkdir(parents=True, exist_ok=True)
    sheet.save(out / 'dense_sparse_comparison.png')
    (out / 'dense_sparse_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')

    table = []
    for item in metrics:
        link = '/' + str((ROOT / item['mesh']).relative_to(ROOT))
        bar = item['top_bar_coverage']
        table.append('<tr><td><a href="' + link + '">' + html.escape(item['stage']) + '</a></td>'
                     + f'<td>{item["bc_pass"]}</td><td>{item["components"]}</td>'
                     + f'<td>{"—" if bar is None else format(bar, ".3f")}</td>'
                     + ''.join(f'<td>{item["iou"].get(v, 0):.3f}</td>' for v in ('front', 'right', 'top'))
                     + '</tr>')
    inp = '/' + str((INPUT / 'input_contact.png').relative_to(ROOT))
    (out / 'index.html').write_text(f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Chair: remove straight corridors</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:2rem auto;background:#f4f6f8;color:#1d2730}}
article{{background:white;padding:1.2rem;margin:1rem 0;border-radius:12px}}img{{max-width:100%}}
table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:.45rem}}</style>
<article><h1>의자 support corridor 제거 비교</h1><p>입력 이미지와 BC·envelope는 동일하다.
새 실험은 {html.escape(label)} 설정이다. FEA는 꺼져 있다.</p>
<img src="{inp}"></article><article><h2>3D 메시 렌더</h2><img src="dense_sparse_comparison.png"></article>
<article><h2>검증</h2><table><tr><th>단계 / OBJ</th><th>BC 통과</th><th>성분</th><th>등받이 위</th>
<th>앞 IoU</th><th>오른쪽 IoU</th><th>위 IoU</th></tr>{''.join(table)}</table>
<p><a href="dense_sparse_metrics.json">수치 JSON</a> · <a href="REPORT.md">설명</a></p></article></html>''')
    print(out / 'index.html')


if __name__ == '__main__':
    main()
