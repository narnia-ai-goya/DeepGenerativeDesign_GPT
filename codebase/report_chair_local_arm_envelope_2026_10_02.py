#!/usr/bin/env python3
"""Publish controlled baseline vs local upper-junction envelope comparison."""
from __future__ import annotations

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
OLD = BASE / 'fresh_dense_no_prototype_2026-10-02/open_arm/image_strong'
OUT = BASE / 'local_arm_envelope_2026-10-02'
NEW = OUT / 'open_arm_image_strong'
INPUT = BASE / 'open_arm/input'
VIEWS = (('oblique', 25, 35, None), ('front', 15, 0, 'v00_front_lo'),
         ('right', 15, 90, 'v02_right_lo'), ('top', 85, 0, 'v_top'))


def main():
    env = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents)) * 1.5
    scale = float(env.extents.max()) / 2
    size, header = 390, 32
    sheet = Image.new('RGB', (size * len(VIEWS), (size + header) * 2), 'white')
    draw = ImageDraw.Draw(sheet)
    metrics = []
    for row, (label, folder) in enumerate((('original envelope', OLD), ('upper-junction extension', NEW))):
        path = folder / 'generation/mesh_dense.obj'
        mesh = trimesh.load(path, force='mesh')
        audit = json.loads((folder / 'bc_audit.json').read_text())
        item = {'label': label, 'mesh': str(path), 'components': audit['mesh_components'],
                'bc_pass': audit['bc_geometry_pass'], 'volume_litres': round(abs(mesh.volume) * 1000, 2),
                'silhouette_iou': {}}
        for col, (view, elev, azim, stem) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            image = Image.fromarray(render_lit(mesh, eye, center, up, size=size,
                fit_extent=scale, margin=1.15, color=(.48, .52, .56))).convert('RGB')
            sheet.paste(image, (col * size, row * (size + header) + header))
            draw.text((col * size + 8, row * (size + header) + 7), f'{label} / {view}', fill='#20252b')
            if stem:
                target = Image.open(INPUT / f'{stem}.png').convert('RGB').resize((size, size))
                item['silhouette_iou'][view] = round(silhouette_scores(image, target)['iou'], 4)
        metrics.append(item)
    sheet.save(OUT / 'dense_comparison.png')
    (OUT / 'dense_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    feas = json.loads((OUT / 'envelope_metrics.json').read_text())
    rows = ''.join('<tr><td><a href="/' + str((ROOT / item['mesh']).relative_to(ROOT)) + '">' + item['label'] +
                   '</a></td><td>' + str(item['bc_pass']) + '</td><td>' + str(item['components']) +
                   '</td><td>' + str(item['volume_litres']) + '</td>' +
                   ''.join(f'<td>{item["silhouette_iou"][v]:.3f}</td>' for v in ('front', 'right', 'top')) +
                   '</tr>' for item in metrics)
    (OUT / 'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair local envelope test</title>
<style>body{font:16px system-ui;max-width:1650px;margin:2rem auto;background:#f4f6f8;color:#1d2730}article{background:white;padding:1.2rem;margin:1rem 0;border-radius:12px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.45rem}</style>
<article><h1>의자 상부 접합부 envelope 확장 실험</h1><p>같은 open_arm 3뷰 입력, seed=42, BC, 50-step dense, FEA off. 후방 팔걸이와 등받이가 만나는 양쪽 상단에만 허용 영역을 추가했습니다.</p>
<p>추가 voxel: ''' + str(feas['added_voxels']) + '''. 오른쪽 뷰에서 envelope 밖 목표 픽셀: 16 → 6. 빨강은 현재 envelope에서 만들 수 없는 목표 윤곽입니다.</p><img src="envelope_feasibility_comparison.png"></article>
<article><h2>판정</h2><p>envelope만 넓혀서는 개선되지 않았습니다. 등받이는 여전히 분리되며, 재료량과 메시 성분 수가 늘어났습니다. 이 후보는 sparse로 진행하지 않았습니다.</p></article>
<article><h2>3D dense 메시</h2><img src="dense_comparison.png"></article><article><h2>비교 수치</h2><table><tr><th>조건 / OBJ</th><th>BC 통과</th><th>성분</th><th>체적 L</th><th>정면 IoU</th><th>오른쪽 IoU</th><th>상면 IoU</th></tr>''' + rows + '''</table><p><a href="REPORT.md">판정</a> · <a href="dense_metrics.json">수치 JSON</a></p></article></html>''')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
