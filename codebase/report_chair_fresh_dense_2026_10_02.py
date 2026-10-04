#!/usr/bin/env python3
"""Compare fresh image-conditioned chair dense generation with the prior dense cache."""
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

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28/front_support_only_2026-10-02/complex_truss_armchair'
OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/fresh_dense_no_prototype_2026-10-02'
INPUT = ROOT / 'experiments/chair/image_concepts_2026-09-27/complex_truss_armchair/multiview_registered'
VIEWS = (('oblique', 25, 35, None), ('front', 15, 0, 'v00_front_lo'),
         ('right', 15, 90, 'v02_right_lo'), ('top', 85, 0, 'v_top'))


def main():
    entries = [
        ('prior dense with prototype', BASE / 'dense_pw2/mesh_dense.obj', BASE / 'dense_pw2_bc_audit.json'),
        ('fresh dense, no shape guidance', OUT / 'generation/mesh_dense.obj', OUT / 'bc_audit.json'),
        ('fresh dense, image w=50', OUT / 'image_guided/generation/mesh_dense.obj', OUT / 'image_guided/bc_audit.json'),
        ('fresh dense, image w=150', OUT / 'image_strong/generation/mesh_dense.obj', OUT / 'image_strong/bc_audit.json'),
    ]
    entries = [e for e in entries if e[1].exists() and e[2].exists()]
    env = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents)) * 1.5
    scale = float(env.extents.max()) / 2
    size, header = 360, 34
    sheet = Image.new('RGB', (size * len(VIEWS), (size + header) * len(entries)), 'white')
    draw = ImageDraw.Draw(sheet)
    metrics = []
    for row, (label, path, audit_path) in enumerate(entries):
        mesh = trimesh.load(path, force='mesh')
        audit = json.loads(audit_path.read_text())
        item = {'stage': label, 'mesh': str(path), 'components': audit['mesh_components'],
                'bc_pass': audit['bc_geometry_pass'], 'iou': {}}
        for col, (view, elev, azim, stem) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rendered = Image.fromarray(render_lit(mesh, eye, center, up, size=size,
                fit_extent=scale, margin=1.15, color=(.48, .52, .56))).convert('RGB')
            sheet.paste(rendered, (col * size, row * (size + header) + header))
            draw.text((col * size + 7, row * (size + header) + 7), label + ' / ' + view, fill='#20252b')
            if stem:
                target = Image.open(INPUT / 'input' / (stem + '.png')).convert('RGB').resize((size, size))
                item['iou'][view] = silhouette_scores(rendered, target)['iou']
        metrics.append(item)
    OUT.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT / 'dense_comparison.png')
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    rows = []
    for item in metrics:
        mesh = '/' + str((ROOT / item['mesh']).relative_to(ROOT))
        rows.append('<tr><td><a href="' + mesh + '">' + html.escape(item['stage']) + '</a></td>'
                    + f'<td>{item["bc_pass"]}</td><td>{item["components"]}</td>'
                    + ''.join(f'<td>{item["iou"].get(v, 0):.3f}</td>' for v in ('front', 'right', 'top'))
                    + '</tr>')
    contact = '/' + str((INPUT / 'input_contact.png').relative_to(ROOT))
    (OUT / 'index.html').write_text(f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Chair fresh dense</title><style>body{{font:16px system-ui;max-width:1540px;margin:2rem auto;background:#f4f6f8;color:#1d2730}}
article{{background:white;padding:1.2rem;margin:1rem 0;border-radius:12px}}img{{max-width:100%}}table{{border-collapse:collapse}}
td,th{{border:1px solid #bbb;padding:.45rem}}</style>
<article><h1>의자 dense 재생성</h1><p>같은 입력 이미지·BC·envelope·seed로 dense를 새로 생성했다. 새 실험 모두 prototype anchor와 FEA를 껐다. clean은 image projection과 support corridor도 끄고, w=50/150은 둘을 사용했다.</p>
<p><strong>시각 평가:</strong> 세 새 dense 모두 경사진 팔걸이·등받이 X자·좌석 아래 대각재가 부족하므로 sparse로 진행하지 않았다.</p>
<img src="{contact}"></article><article><h2>조명 적용 3D 렌더</h2><img src="dense_comparison.png"></article>
<article><h2>검증</h2><table><tr><th>단계 / OBJ</th><th>BC 통과</th><th>성분</th><th>앞 IoU</th><th>오른쪽 IoU</th><th>위 IoU</th></tr>
{''.join(rows)}</table><p><a href="metrics.json">수치 JSON</a> · <a href="REPORT.md">해석</a></p></article></html>''')
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
