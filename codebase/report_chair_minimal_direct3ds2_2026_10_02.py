#!/usr/bin/env python3
"""Assess the image-only Direct3D-S2 chair after dense and sparse stages."""
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
OUT = BASE / 'minimal_direct3ds2_2026-10-02'
INPUT = BASE / 'open_arm/input'
BC = BASE / 'consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
VIEWS = (('oblique', 25, 35, None), ('front', 15, 0, 'v00_front_lo'),
         ('right', 15, 90, 'v02_right_lo'), ('top', 85, 0, 'v_top'))


def main():
    env = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents)) * 1.5
    scale = float(env.extents.max()) / 2
    dense = trimesh.load(OUT / 'generation/mesh_dense_raw.obj', force='mesh')
    sparse = trimesh.load(OUT / 'generation/mesh.obj', force='mesh')
    rotation = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    source_center = dense.bounds.mean(axis=0)
    for name, mesh in (('dense', dense), ('sparse', sparse)):
        aligned = mesh.copy()
        aligned.vertices = (aligned.vertices - source_center) @ rotation.T * 0.94 + center
        aligned.export(OUT / f'aligned_{name}.obj')
        check = audit(OUT / f'aligned_{name}.obj', BC)
        (OUT / f'aligned_{name}_bc_audit.json').write_text(json.dumps(check, indent=2) + '\n')
        if name == 'sparse':
            main_component = max(aligned.split(only_watertight=False), key=lambda part: abs(part.volume))
            main_component.export(OUT / 'aligned_sparse_main.obj')
    cases = (
        ('raw sparse', OUT / 'generation/mesh.obj'),
        ('aligned dense', OUT / 'aligned_dense.obj'),
        ('aligned sparse', OUT / 'aligned_sparse.obj'),
    )
    size, header = 410, 34
    sheet = Image.new('RGB', (size * 4, (size + header) * len(cases)), 'white')
    draw = ImageDraw.Draw(sheet)
    metrics = []
    for row, (label, path) in enumerate(cases):
        mesh = trimesh.load(path, force='mesh')
        parts = mesh.split(only_watertight=False)
        item = {'label': label, 'mesh': str(path), 'components': len(parts),
                'largest_component_volume_fraction': round(max(abs(p.volume) for p in parts) /
                    max(sum(abs(p.volume) for p in parts), 1e-12), 6),
                'volume_litres': round(abs(mesh.volume) * 1000, 3), 'silhouette_iou': {}}
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
    sheet.save(OUT / 'dense_sparse_comparison.png')
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    bc = json.loads((OUT / 'aligned_sparse_bc_audit.json').read_text())
    rows = ''.join('<tr><td><a href="/' + str(path.relative_to(ROOT)) + '">' + label +
                   '</a></td><td>' + str(item['components']) + '</td><td>' + str(item['volume_litres']) +
                   '</td>' + ''.join(f'<td>{item["silhouette_iou"][v]:.3f}</td>' for v in ('front','right','top')) +
                   '</tr>' for (label,path),item in zip(cases,metrics))
    (OUT / 'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Minimal Direct3D-S2 chair</title>
<style>body{font:16px system-ui;max-width:1700px;margin:2rem auto;background:#f4f6f8;color:#1d2730}article{background:white;padding:1.2rem;margin:1rem 0;border-radius:12px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.45rem}</style>
<article><h1>최소 구성: 정면 1뷰 → 순수 Direct3D-S2 dense+sparse → 좌표 정렬</h1>
<p>seed 42, CFG 7, dense 50/sparse 30 step. BC·envelope·실루엣·FEA guidance는 모두 껐습니다. 정렬은 모델 결과의 X축 +90° 회전과 균일 0.94배 및 중심 이동만 적용했습니다. 첫 줄은 정렬 전 sparse, 둘째·셋째 줄은 정렬한 dense/sparse입니다.</p>
<p>정렬 sparse의 BC 검증: ''' + ('통과' if bc['bc_geometry_pass'] else '실패') + '''. 생성 과정에 BC를 넣지 않았으므로 이 결과를 구조적으로 유효한 의자로 해석하면 안 됩니다.</p>
<img src="/experiments/chair/sofa_style_2026-09-28/open_arm/input_contact.png"></article>
<article><h2>동일 카메라 3D 비교</h2><img src="dense_sparse_comparison.png"></article>
<article><h2>수치</h2><table><tr><th>결과 / OBJ</th><th>성분</th><th>체적 L</th><th>앞 IoU</th><th>오른쪽 IoU</th><th>상면 IoU</th></tr>''' + rows + '''</table>
<p><a href="REPORT.md">판정</a> · <a href="metrics.json">수치 JSON</a> · <a href="aligned_sparse_bc_audit.json">BC 검증</a></p></article></html>''')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
