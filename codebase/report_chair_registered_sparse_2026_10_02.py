#!/usr/bin/env python3
"""Compare registered-spec chair sparse variants and produce final clipped mesh."""
from __future__ import annotations

import html
import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from make_chair_domain import ROOT
from validate_chair_bc_geometry import audit

sys.path.insert(0, str(ROOT / 'codebase/code'))
sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402


BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'registered_spec_2026-10-02'
BC = BASE / 'consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
VARIANTS = ('vanilla', 'soft_bc', 'hard_bc', 'hard_bc_fix13')
VIEWS = (('front', 15, 0), ('right', 15, 90), ('top', 85, 0))


def prepare():
    c = json.loads((OUT / 'calibration.json').read_text())['native_to_physical']
    source, target = np.asarray(c['source_center_m']), np.asarray(c['physical_center_m'])
    rotation = Rotation.from_euler('x', c['rotation_x_degrees'], degrees=True).as_matrix()
    scale = float(c['uniform_scale'])
    for variant in VARIANTS:
        directory = OUT / ('strong_sparse_' + variant)
        mesh = trimesh.load(directory / 'generation/mesh.obj', force='mesh')
        mesh.vertices = (mesh.vertices - source) @ rotation.T * scale + target
        mesh.export(directory / 'physical_mesh.obj')
    directory = OUT / 'strong_sparse_hard_bc_fix13'
    mesh = trimesh.load(directory / 'physical_mesh.obj', force='mesh')
    parts = mesh.split(only_watertight=False)
    main = max(parts, key=lambda part: abs(part.volume))
    main.export(directory / 'physical_mesh_main.obj')
    envelope = trimesh.load(BASE / 'coherent_proxy_dense/envelope.stl', force='mesh')
    clipped = trimesh.boolean.intersection([main, envelope], engine='manifold')
    clipped.export(directory / 'physical_mesh_envelope_clipped.obj')
    original_domain = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    final = trimesh.boolean.intersection([clipped, original_domain], engine='manifold')
    final.export(directory / 'physical_mesh_original_domain_clipped.obj')


def main():
    prepare()
    paths = [
        ('dense / strong', OUT / 'strong/physical_mesh_dense.obj'),
        ('sparse / no guidance', OUT / 'strong_sparse_vanilla/physical_mesh.obj'),
        ('sparse / soft BC', OUT / 'strong_sparse_soft_bc/physical_mesh.obj'),
        ('sparse / hard 6 mm', OUT / 'strong_sparse_hard_bc/physical_mesh.obj'),
        ('sparse / hard fix 13 mm', OUT / 'strong_sparse_hard_bc_fix13/physical_mesh.obj'),
        ('sparse / final envelope intersection', OUT / 'strong_sparse_hard_bc_fix13/physical_mesh_envelope_clipped.obj'),
        ('sparse / final original domain', OUT / 'strong_sparse_hard_bc_fix13/physical_mesh_original_domain_clipped.obj'),
    ]
    data = np.load(BC)
    envelope = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    original_occupancy = voxel_centers_inside(envelope, 64, data['origin'], data['pitch_xyz'])
    center = envelope.bounds.mean(axis=0)
    radius = float(np.linalg.norm(envelope.extents)) * 1.5
    fit_extent = float(envelope.extents.max()) / 2
    tile, header = 360, 28
    sheet = Image.new('RGB', (tile * len(VIEWS), (tile + header) * len(paths)), 'white')
    draw = ImageDraw.Draw(sheet)
    rows = []
    for row, (name, path) in enumerate(paths):
        mesh = trimesh.load(path, force='mesh')
        occupancy = voxel_centers_inside(mesh, 64, data['origin'], data['pitch_xyz'])
        check = audit(path, BC)
        item = {'name': name, 'mesh': str(path), 'volume_liters': round(abs(mesh.volume) * 1000, 3),
                'watertight': bool(mesh.is_watertight), 'mesh_components': check['mesh_components'],
                'largest_component_volume_fraction': check['largest_component_volume_fraction'],
                'outside_envelope_voxels': int((occupancy & ~data['bracket'].astype(bool)).sum()),
                'outside_original_domain_voxels': int((occupancy & ~original_occupancy).sum()),
                'occupied_voxels': int(occupancy.sum()),
                'bc_connected': check['all_bc_same_6_connected_component'],
                'bc_coverage': {k: v['coverage'] for k, v in check['regions'].items()},
                'bc_geometry_pass': check['bc_geometry_pass']}
        rows.append(item)
        for col, (view, elev, azim) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rgb = render_lit(mesh, eye, center, up, size=tile, fit_extent=fit_extent,
                             margin=1.15, color=(.54, .58, .62))
            sheet.paste(Image.fromarray(rgb).convert('RGB'),
                        (col * tile, row * (tile + header) + header))
            draw.text((col * tile + 8, row * (tile + header) + 5),
                      f'{name} / {view}', fill='#20252b')
    sheet.save(OUT / 'sparse_comparison.png')
    (OUT / 'sparse_metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    table = ''.join('<tr><td><a href="/' + str(path.relative_to(ROOT)) + '">' +
                    html.escape(item['name']) + '</a></td><td>' + str(item['volume_liters']) +
                    '</td><td>' + str(item['mesh_components']) +
                    '</td><td>' + str(item['outside_envelope_voxels']) +
                    '</td><td>' + str(item['outside_original_domain_voxels']) +
                    '</td><td>' + f"{item['bc_coverage']['seat_load']:.1%}" +
                    '</td><td>' + f"{item['bc_coverage']['backrest_load']:.1%}" +
                    '</td><td>' + f"{min(item['bc_coverage'][k] for k in item['bc_coverage'] if k.startswith('foot')):.1%}" +
                    '</td><td>' + str(item['bc_geometry_pass']) + '</td></tr>'
                    for (_, path), item in zip(paths, rows))
    (OUT / 'sparse_index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Registered chair sparse specification</title>
<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;background:#f3f5f7;color:#17212b}article{background:white;border-radius:10px;padding:1.3rem;margin:1rem 0}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.4rem}</style>
<article><h1>좌표 등록된 의자 specification: sparse 비교</h1><p>같은 정면 이미지·seed 42·등록된 dense strong cache로 sparse를 생성했다. 모든 결과는 원래 물리 좌표로 되돌려 기존 BC/envelope으로 평가했다. FEA off. 마지막 행은 발 BC 13 mm, 하중 BC 6 mm를 추출 중 강제하고 큰 성분만 남긴 뒤 실험용 envelope과 원본 설계 STL 모두에 교집합한 결과다.</p><p><a href="index.html">dense 비교</a> · <a href="REPORT.md">실험 판정</a></p></article>
<article><h2>정면·측면·상면 3D 메시</h2><img src="sparse_comparison.png"></article>
<article><h2>원래 물리 사양 판정</h2><table><tr><th>조건/OBJ</th><th>체적 L</th><th>메시 성분</th><th>64³ envelope 밖</th><th>원본 설계 STL 밖</th><th>좌석 BC</th><th>등받이 BC</th><th>발 최소 BC</th><th>BC 판정</th></tr>''' + table +
'''</table><p><a href="sparse_metrics.json">전체 수치 JSON</a></p></article></html>''')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
