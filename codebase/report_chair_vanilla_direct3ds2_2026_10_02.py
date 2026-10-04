#!/usr/bin/env python3
"""Render unfiltered Direct3D-S2 dense predictions against the guided chair baseline."""
from __future__ import annotations

import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from make_chair_domain import ROOT
from report_chair_prefea_stages import silhouette_scores
from validate_chair_bc_geometry import audit

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'vanilla_direct3ds2_2026-10-02'
INPUT = BASE / 'open_arm/input'
CASES = (
    ('Direct3D-S2 front only', OUT / 'front/generation/mesh_dense_raw.obj'),
    ('front only, rigid frame alignment', OUT / 'front/aligned_rot_x90.obj'),
    ('Direct3D-S2 3-view fusion', OUT / 'three_view/generation/mesh_dense_raw.obj'),
    ('guided 3-view dense', BASE / 'fresh_dense_no_prototype_2026-10-02/open_arm/image_strong/generation/mesh_dense.obj'),
)
VIEWS = (('oblique', 25, 35, None), ('front', 15, 0, 'v00_front_lo'),
         ('right', 15, 90, 'v02_right_lo'), ('top', 85, 0, 'v_top'))


def main():
    env = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents)) * 1.5
    # Same framing as the registered input views; the raw meshes still fit it.
    scale = float(env.extents.max()) / 2
    raw = trimesh.load(CASES[0][1], force='mesh')
    aligned = raw.copy()
    aligned.vertices = ((aligned.vertices - raw.bounds.mean(axis=0)) @
                        Rotation.from_euler('x', 90, degrees=True).as_matrix().T * 0.94 + center)
    aligned.export(CASES[1][1])
    bc = BASE / 'consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
    (OUT / 'front/aligned_bc_audit.json').write_text(json.dumps(audit(CASES[1][1], bc), indent=2) + '\n')
    size, header = 410, 34
    sheet = Image.new('RGB', (size * 4, (size + header) * len(CASES)), 'white')
    draw = ImageDraw.Draw(sheet)
    metrics = []
    for row, (label, path) in enumerate(CASES):
        mesh = trimesh.load(path, force='mesh')
        parts = mesh.split(only_watertight=False)
        item = {'label': label, 'mesh': str(path), 'components': len(parts),
                'largest_component_volume_fraction': round(max(abs(p.volume) for p in parts) /
                    max(sum(abs(p.volume) for p in parts), 1e-12), 4),
                'volume_litres': round(abs(mesh.volume) * 1000, 2),
                'bounds_m': mesh.bounds.tolist(), 'silhouette_iou': {}}
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
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    rows = ''.join('<tr><td><a href="/' + str(path.relative_to(ROOT)) + '">' + label +
                   '</a></td><td>' + str(item['components']) + '</td><td>' +
                   str(item['volume_litres']) + '</td>' +
                   ''.join(f'<td>{item["silhouette_iou"][v]:.3f}</td>' for v in ('front','right','top')) +
                   '</tr>' for (label,path),item in zip(CASES,metrics))
    (OUT / 'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Direct3D-S2 raw dense chair</title>
<style>body{font:16px system-ui;max-width:1700px;margin:2rem auto;background:#f4f6f8;color:#1d2730}article{background:white;padding:1.2rem;margin:1rem 0;border-radius:12px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.45rem}</style>
<article><h1>open_arm 입력: 순수 Direct3D-S2 dense</h1><p>첫째·셋째 줄은 VANILLA=1, CFG 7, 50 steps, seed 42. BC·envelope 필터·투영 손실 없이 나온 <code>mesh_dense_raw.obj</code>입니다. 정면 한 장은 원 모델에 가까운 입력이고, 3뷰 결합은 이 프로젝트의 토큰 결합 방식입니다. 둘째 줄은 정면 결과를 X축 +90° 회전하고 중심·크기만 정렬한 진단 결과이며 생성이나 물리 검증을 다시 한 것이 아닙니다. 넷째 줄은 기존 유도 dense 결과입니다.</p>
<img src="/experiments/chair/sofa_style_2026-09-28/open_arm/input_contact.png"></article>
<article><h2>3D dense 렌더</h2><img src="dense_comparison.png"></article>
<article><h2>비교</h2><table><tr><th>조건 / OBJ</th><th>성분</th><th>체적 L</th><th>앞 IoU</th><th>오른쪽 IoU</th><th>위 IoU</th></tr>''' + rows + '''</table><p><a href="metrics.json">수치 JSON</a> · <a href="REPORT.md">판정</a></p></article></html>''')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
