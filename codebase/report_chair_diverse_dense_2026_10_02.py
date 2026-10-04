#!/usr/bin/env python3
"""Compare fresh dense results across registered chair styles."""
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
OUT = BASE / 'fresh_dense_no_prototype_2026-10-02'
INPUTS = {
    'complex_truss': ROOT / 'experiments/chair/image_concepts_2026-09-27/complex_truss_armchair/multiview_registered',
    'open_arm': BASE / 'open_arm',
    'solid_side': BASE / 'solid_side',
}
VIEWS = (('oblique', 25, 35, None), ('front', 15, 0, 'v00_front_lo'),
         ('right', 15, 90, 'v02_right_lo'), ('top', 85, 0, 'v_top'))


def main():
    env = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents)) * 1.5
    scale = float(env.extents.max()) / 2
    size, header = 350, 34
    sheet = Image.new('RGB', (size * len(VIEWS), (size + header) * len(INPUTS)), 'white')
    draw = ImageDraw.Draw(sheet)
    metrics = []
    for row, (case, source) in enumerate(INPUTS.items()):
        run = OUT / 'image_strong' if case == 'complex_truss' else OUT / case / 'image_strong'
        path = run / 'generation/mesh_dense.obj'
        if not path.exists():
            continue
        mesh = trimesh.load(path, force='mesh')
        audit = json.loads((run / 'bc_audit.json').read_text())
        item = {'case': case, 'mesh': str(path), 'bc_pass': audit['bc_geometry_pass'],
                'components': audit['mesh_components'], 'iou': {}}
        for col, (view, elev, azim, stem) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rendered = Image.fromarray(render_lit(mesh, eye, center, up, size=size,
                fit_extent=scale, margin=1.15, color=(.48, .52, .56))).convert('RGB')
            sheet.paste(rendered, (col * size, row * (size + header) + header))
            draw.text((col * size + 7, row * (size + header) + 7), case + ' / ' + view, fill='#20252b')
            if stem:
                target = Image.open(source / 'input' / (stem + '.png')).convert('RGB').resize((size, size))
                item['iou'][view] = silhouette_scores(rendered, target)['iou']
        metrics.append(item)
    sheet.save(OUT / 'diverse_dense_comparison.png')
    (OUT / 'diverse_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    cards = []
    for item in metrics:
        source = INPUTS[item['case']]
        contact = '/' + str((source / 'input_contact.png').relative_to(ROOT))
        mesh = '/' + str((ROOT / item['mesh']).relative_to(ROOT))
        cards.append(f'<article><h2>{html.escape(item["case"])}</h2>'
                     + f'<p><a href="{mesh}">dense OBJ</a> · BC {item["bc_pass"]} · components {item["components"]} · '
                     + f'IoU front/right/top {item["iou"].get("front",0):.3f}/{item["iou"].get("right",0):.3f}/{item["iou"].get("top",0):.3f}</p>'
                     + f'<img src="{contact}"></article>')
    (OUT / 'diverse_index.html').write_text(f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Thicker chair forms, fresh dense</title><style>body{{font:16px system-ui;max-width:1500px;margin:2rem auto;background:#f4f6f8;color:#1d2730}}
article{{background:white;padding:1.2rem;margin:1rem 0;border-radius:12px}}img{{max-width:100%}}</style>
<article><h1>두께감 있는 의자 형태 비교</h1><p>같은 BC·envelope·seed·dense 설정에서 입력 의자 형태만 변경했다. prototype anchor 및 FEA off.</p>
<p><strong>평가:</strong> 막힌 측면형은 두꺼워졌지만 판이 바닥까지 과장되었고, 열린 팔걸이형은 등받이가 분리되었다. 세 후보 모두 sparse 단계로 넘기지 않았다.</p>
<img src="diverse_dense_comparison.png"><p><a href="diverse_metrics.json">수치 JSON</a> · <a href="DIVERSE_REPORT.md">실험 해석</a></p></article>
{''.join(cards)}</html>''')
    print(OUT / 'diverse_index.html')


if __name__ == '__main__':
    main()
