#!/usr/bin/env python3
"""Summarize controlled specification ablations in the common chair frame."""
from __future__ import annotations

import html
import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import label

from make_chair_domain import ROOT

sys.path.insert(0, str(ROOT / 'codebase/code'))
sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402


BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'spec_root_cause_2026-10-02'
BC = BASE / 'consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
CASES = [
    ('D3D-S2 raw', BASE / 'minimal_direct3ds2_2026-10-02/generation/mesh_dense_raw.obj'),
    ('D3D-S2 aligned', BASE / 'minimal_direct3ds2_2026-10-02/aligned_dense.obj'),
    ('out only raw', OUT / 'out_only/generation/mesh_dense_raw.obj'),
    ('BC only raw', OUT / 'bc_only/generation/mesh_dense_raw.obj'),
    ('out + BC raw', OUT / 'out_bc/generation/mesh_dense_raw.obj'),
    ('previous full guidance raw', BASE / 'fresh_dense_no_prototype_2026-10-02/oneview_open_arm/front/generation/mesh_dense_raw.obj'),
    ('previous full guidance final', BASE / 'fresh_dense_no_prototype_2026-10-02/oneview_open_arm/front/generation/mesh_dense.obj'),
]
VIEWS = [('front', 15, 0), ('right', 15, 90), ('top', 85, 0)]


def metrics(mesh: trimesh.Trimesh, data: np.lib.npyio.NpzFile) -> dict:
    occ = voxel_centers_inside(mesh, 64, data['origin'], data['pitch_xyz'])
    env, bc = data['bracket'].astype(bool), data['bc'].astype(bool)
    _, count = label(occ)
    original_count = count
    labs, count = label(occ & env)
    sizes = np.bincount(labs.ravel())[1:]
    return {
        'voxels': int(occ.sum()),
        'outside_envelope_fraction': round(float((occ & ~env).sum() / max(occ.sum(), 1)), 4),
        'bc_coverage': round(float((occ & bc).sum() / bc.sum()), 4),
        'raw_components': int(original_count),
        'clipped_components': int(count),
        'clipped_largest_fraction': round(float(sizes.max() / max((occ & env).sum(), 1)), 4) if len(sizes) else 0,
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = np.load(BC)
    envelope = trimesh.load(BASE.parent.parent.parent / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = envelope.bounds.mean(axis=0)
    radius = float(np.linalg.norm(envelope.extents)) * 1.5
    scale = float(envelope.extents.max()) / 2
    tile, header = 360, 28
    sheet = Image.new('RGB', (tile * len(VIEWS), (tile + header) * len(CASES)), 'white')
    draw = ImageDraw.Draw(sheet)
    results = []
    for row, (name, path) in enumerate(CASES):
        mesh = trimesh.load(path, force='mesh')
        item = {'case': name, 'mesh': str(path), **metrics(mesh, data)}
        results.append(item)
        for col, (view, elev, azim) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rendered = render_lit(mesh, eye, center, up, size=tile,
                                  fit_extent=scale, margin=1.15, color=(.54, .58, .62))
            sheet.paste(Image.fromarray(rendered).convert('RGB'),
                        (col * tile, row * (tile + header) + header))
            draw.text((col * tile + 8, row * (tile + header) + 5),
                      f'{name} / {view}', fill='#20252b')
    figure = OUT / 'raw_comparison.png'
    sheet.save(figure)
    (OUT / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    rows = ''.join('<tr><td><a href="/' + str(path.relative_to(ROOT)) + '">' +
                   html.escape(item['case']) + '</a></td><td>' + str(item['voxels']) +
                   '</td><td>' + f"{item['outside_envelope_fraction']:.1%}" +
                   '</td><td>' + f"{item['bc_coverage']:.1%}" +
                   '</td><td>' + str(item['raw_components']) +
                   '</td><td>' + str(item['clipped_components']) + '</td></tr>'
                   for (_, path), item in zip(CASES, results))
    (OUT / 'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair specification root cause</title>
<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;background:#f3f5f7;color:#17212b}article{background:white;border-radius:10px;padding:1.3rem;margin:1rem 0}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.4rem}</style>
<article><h1>의자 specification 충돌 진단</h1><p>정면 이미지 한 장, seed 42, CFG 7, dense 50 steps, FEA off. 실험 3개는 out/BC 손실만 바꾸고 원본 모델 출력(mesh_dense_raw.obj)을 비교합니다. 수치는 동일한 64³ 의자 격자에서 평가했습니다. 이전 full-guidance 사례는 CFG와 기타 손실이 달라 참고용입니다.</p>
<img src="/experiments/chair/sofa_style_2026-09-28/open_arm/input_contact.png"></article>
<article><h2>공통 카메라의 3D 메시</h2><img src="raw_comparison.png"></article>
<article><h2>원본 메시의 voxel 진단</h2><table><tr><th>조건/OBJ</th><th>재료 voxel</th><th>envelope 밖</th><th>BC 포함</th><th>원본 성분</th><th>clip 후 성분</th></tr>''' + rows +
'''</table><p><a href="metrics.json">수치 JSON</a> · <a href="REPORT.md">분석</a></p></article></html>''')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
