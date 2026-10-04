#!/usr/bin/env python3
"""Visual and BC audit for the corrected soft sparse chair with fix-only preservation."""
from __future__ import annotations

import html
import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from make_chair_domain import ROOT
from report_chair_prefea_stages import silhouette_scores
from validate_chair_bc_geometry import audit

sys.path.insert(0, str(ROOT / 'codebase/code'))
sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402


BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
STUDY = BASE / 'registered_spec_2026-10-02'
OUT = STUDY / 'soft_bc_update_2026-10-02'
BC = BASE / 'consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
CASES = [
    ('original soft BC', STUDY / 'strong_sparse_soft_bc/physical_mesh.obj'),
    ('sign-corrected + BC halo', STUDY / 'strong_sparse_soft_sign_bc_halo1/physical_mesh.obj'),
    ('fix only / current 64³ domain', STUDY / 'strong_sparse_soft_halo1_fix13/physical_mesh_current_mask_clipped.obj'),
    ('fix only / original STL domain', STUDY / 'strong_sparse_soft_halo1_fix13/physical_mesh_original_clipped.obj'),
    ('previous all-BC hard / current domain', STUDY / 'strong_sparse_hard_bc_fix13/physical_mesh_envelope_clipped.obj'),
]
VIEWS = [('front', 15, 0, 'v00_front_lo'),
         ('right', 15, 90, 'v02_right_lo'), ('top', 85, 0, 'v_top')]


def prepare_updated_meshes():
    directory = STUDY / 'strong_sparse_soft_halo1_fix13'
    calibration = json.loads((STUDY / 'calibration.json').read_text())['native_to_physical']
    source = np.asarray(calibration['source_center_m'])
    center = np.asarray(calibration['physical_center_m'])
    rotation = Rotation.from_euler('x', calibration['rotation_x_degrees'], degrees=True).as_matrix()
    mesh = trimesh.load(directory / 'generation/mesh.obj', force='mesh')
    mesh.vertices = (mesh.vertices - source) @ rotation.T * float(calibration['uniform_scale']) + center
    mesh.export(directory / 'physical_mesh.obj')
    main = max(mesh.split(only_watertight=False), key=lambda part: abs(part.volume))
    main.export(directory / 'physical_mesh_main.obj')
    for name, path in [('current_mask', BASE / 'coherent_proxy_dense/envelope.stl'),
                       ('original', ROOT / 'data_real/chair/original_DesignSpace.stl')]:
        envelope = trimesh.load(path, force='mesh')
        clipped = trimesh.boolean.intersection([main, envelope], engine='manifold')
        clipped.export(directory / f'physical_mesh_{name}_clipped.obj')


def main():
    OUT.mkdir(exist_ok=True)
    prepare_updated_meshes()
    bc = np.load(BC)
    original_domain = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    original_occ = voxel_centers_inside(original_domain, 64, bc['origin'], bc['pitch_xyz'])
    center = original_domain.bounds.mean(axis=0)
    radius = float(np.linalg.norm(original_domain.extents)) * 1.5
    fit_extent = float(original_domain.extents.max()) / 2
    tile, header = 410, 30
    sheet = Image.new('RGB', (tile * len(VIEWS), (tile + header) * len(CASES)), 'white')
    draw = ImageDraw.Draw(sheet)
    rows = []
    for row, (name, path) in enumerate(CASES):
        mesh = trimesh.load(path, force='mesh')
        occ = voxel_centers_inside(mesh, 64, bc['origin'], bc['pitch_xyz'])
        check = audit(path, BC)
        item = {'name': name, 'mesh': str(path), 'volume_liters': round(abs(mesh.volume) * 1000, 3),
                'watertight': bool(mesh.is_watertight), 'mesh_components': check['mesh_components'],
                'outside_64_envelope_voxels': int((occ & ~bc['bracket'].astype(bool)).sum()),
                'outside_original_stl_voxels': int((occ & ~original_occ).sum()),
                'bc_geometry_pass': check['bc_geometry_pass'],
                'bc_connected': check['all_bc_same_6_connected_component'],
                'bc_coverage': {k: v['coverage'] for k, v in check['regions'].items()},
                'silhouette_iou': {}}
        for col, (view, elev, azim, stem) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rgb = render_lit(mesh, eye, center, up, size=tile,
                             fit_extent=fit_extent, margin=1.15, color=(.54, .58, .62))
            image = Image.fromarray(rgb).convert('RGB')
            sheet.paste(image, (col * tile, row * (tile + header) + header))
            draw.text((col * tile + 8, row * (tile + header) + 6),
                      f'{name} / {view}', fill='#20252b')
            target = Image.open(BASE / 'open_arm/input' / f'{stem}.png').convert('RGB').resize((tile, tile))
            item['silhouette_iou'][view] = round(silhouette_scores(image, target)['iou'], 4)
        rows.append(item)
    sheet.save(OUT / 'comparison.png')
    (OUT / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    table = ''.join('<tr><td><a href="/' + str(path.relative_to(ROOT)) + '">' + html.escape(item['name']) +
                    '</a></td><td>' + f"{item['bc_coverage']['seat_load']:.1%}" +
                    '</td><td>' + f"{item['bc_coverage']['backrest_load']:.1%}" +
                    '</td><td>' + f"{min(item['bc_coverage'][k] for k in item['bc_coverage'] if k.startswith('foot')):.1%}" +
                    '</td><td>' + str(item['mesh_components']) +
                    '</td><td>' + str(item['outside_original_stl_voxels']) +
                    '</td><td>' + f"{item['silhouette_iou']['front']:.3f}" +
                    '</td><td>' + f"{item['silhouette_iou']['right']:.3f}" +
                    '</td><td>' + str(item['bc_geometry_pass']) + '</td></tr>'
                    for (_, path), item in zip(CASES, rows))
    (OUT / 'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair soft BC update</title>
<style>body{font:16px system-ui;max-width:1350px;margin:2rem auto;background:#f3f5f7;color:#17212b}article{background:white;border-radius:10px;padding:1.3rem;margin:1rem 0}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.4rem}</style>
<article><h1>의자 sparse soft BC 개선</h1><p>동일 정면 입력·seed 42·dense cache. 손실 부호 정렬, 1-voxel BC 활성 영역, 발 fix에만 high-resolution solid 보존을 적용했다. FEA off. 모든 메시는 물리 좌표계에서 원래 BC로 검사했다.</p><p><a href="REPORT.md">판정 보고서</a> · <a href="metrics.json">수치 JSON</a></p></article>
<article><h2>공통 카메라 3D 비교</h2><img src="comparison.png"></article>
<article><h2>원래 BC 및 이미지 실루엣</h2><table><tr><th>조건/OBJ</th><th>좌석</th><th>등받이</th><th>발 최소</th><th>성분</th><th>원본 STL 밖</th><th>정면 IoU</th><th>측면 IoU</th><th>BC 판정</th></tr>''' + table +
'''</table></article></html>''')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
