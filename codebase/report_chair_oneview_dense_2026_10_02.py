#!/usr/bin/env python3
"""Compare single-view and three-view dense chair generation."""
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

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SOURCE = BASE / 'open_arm'
STUDY = BASE / 'fresh_dense_no_prototype_2026-10-02'
OUT = STUDY / 'oneview_open_arm'
VIEWS = (('oblique', 25, 35, None), ('front', 15, 0, 'v00_front_lo'),
         ('right', 15, 90, 'v02_right_lo'), ('top', 85, 0, 'v_top'))


def main():
    entries = [
        ('three views', STUDY / 'open_arm/image_strong/generation/mesh_dense.obj', STUDY / 'open_arm/image_strong/bc_audit.json'),
        ('front only', OUT / 'front/generation/mesh_dense.obj', OUT / 'front/bc_audit.json'),
        ('right only', OUT / 'right/generation/mesh_dense.obj', OUT / 'right/bc_audit.json'),
        ('front image + 3-view loss', OUT / 'front_allproj/generation/mesh_dense.obj', OUT / 'front_allproj/bc_audit.json'),
    ]
    entries = [e for e in entries if e[1].exists() and e[2].exists()]
    env = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents)) * 1.5
    scale = float(env.extents.max()) / 2
    size, header = 350, 34
    sheet = Image.new('RGB', (size * len(VIEWS), (size + header) * len(entries)), 'white')
    draw = ImageDraw.Draw(sheet)
    metrics = []
    for row, (label, path, audit_path) in enumerate(entries):
        mesh = trimesh.load(path, force='mesh')
        audit = json.loads(audit_path.read_text())
        item = {'stage': label, 'mesh': str(path), 'bc_pass': audit['bc_geometry_pass'],
                'components': audit['mesh_components'], 'iou': {}}
        for col, (view, elev, azim, stem) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rendered = Image.fromarray(render_lit(mesh, eye, center, up, size=size,
                fit_extent=scale, margin=1.15, color=(.48, .52, .56))).convert('RGB')
            sheet.paste(rendered, (col * size, row * (size + header) + header))
            draw.text((col * size + 7, row * (size + header) + 7), label + ' / ' + view, fill='#20252b')
            if stem:
                target = Image.open(SOURCE / 'input' / (stem + '.png')).convert('RGB').resize((size, size))
                item['iou'][view] = silhouette_scores(rendered, target)['iou']
        metrics.append(item)
    OUT.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT / 'oneview_comparison.png')
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    rows = []
    for item in metrics:
        mesh = '/' + str((ROOT / item['mesh']).relative_to(ROOT))
        rows.append('<tr><td><a href="' + mesh + '">' + html.escape(item['stage']) + '</a></td>'
                    + f'<td>{item["bc_pass"]}</td><td>{item["components"]}</td>'
                    + ''.join(f'<td>{item["iou"].get(v, 0):.3f}</td>' for v in ('front', 'right', 'top'))
                    + '</tr>')
    contact = '/' + str((SOURCE / 'input_contact.png').relative_to(ROOT))
    (OUT / 'index.html').write_text(f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>One-view chair dense</title><style>body{{font:16px system-ui;max-width:1500px;margin:2rem auto;background:#f4f6f8;color:#1d2730}}
article{{background:white;padding:1.2rem;margin:1rem 0;border-radius:12px}}img{{max-width:100%}}table{{border-collapse:collapse}}
td,th{{border:1px solid #bbb;padding:.45rem}}</style>
<article><h1>의자 단일 뷰 vs 3뷰</h1><p>같은 BC·envelope·seed, prototype 없음, FEA off. front/right only는 조건 이미지와 투영 손실을 함께 1뷰로 줄였고, front image + 3-view loss는 조건 이미지만 1뷰로 줄였다.</p>
<p><strong>판정:</strong> 정면 한 장은 정면 형상만 개선했고 네 결과 모두 BC 연결 실패. sparse 단계는 실행하지 않았다.</p>
<img src="{contact}"></article><article><h2>조명 적용 3D 렌더</h2><img src="oneview_comparison.png"></article>
<article><h2>모든 뷰에서 검증</h2><table><tr><th>조건 / OBJ</th><th>BC 통과</th><th>성분</th><th>앞 IoU</th><th>오른쪽 IoU</th><th>위 IoU</th></tr>
{''.join(rows)}</table><p><a href="metrics.json">수치 JSON</a> · <a href="REPORT.md">해석</a></p></article></html>''')
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
