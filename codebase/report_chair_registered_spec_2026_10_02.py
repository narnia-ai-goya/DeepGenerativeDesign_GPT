#!/usr/bin/env python3
"""Return native-frame guided chair meshes to the unchanged physical specification."""
from __future__ import annotations

import html
import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import label
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
CASES = [
    ('D3D-S2 aligned', BASE / 'minimal_direct3ds2_2026-10-02/aligned_dense.obj'),
    ('old guided', BASE / 'fresh_dense_no_prototype_2026-10-02/oneview_open_arm/front/generation/mesh_dense.obj'),
]
for name in ('gentle', 'medium', 'strong'):
    CASES.extend([
        (f'{name} raw', OUT / name / 'physical_mesh_dense_raw.obj'),
        (f'{name} final', OUT / name / 'physical_mesh_dense.obj'),
    ])
VIEWS = [('front', 15, 0), ('right', 15, 90), ('top', 85, 0)]


def physical_meshes() -> None:
    calibration = json.loads((OUT / 'calibration.json').read_text())['native_to_physical']
    center_a = np.asarray(calibration['source_center_m'])
    center_b = np.asarray(calibration['physical_center_m'])
    rotation = Rotation.from_euler('x', calibration['rotation_x_degrees'], degrees=True).as_matrix()
    scale = float(calibration['uniform_scale'])
    for name in ('gentle', 'medium', 'strong'):
        for kind in ('mesh_dense_raw', 'mesh_dense'):
            source = OUT / name / 'generation' / f'{kind}.obj'
            target = OUT / name / f'physical_{kind}.obj'
            mesh = trimesh.load(source, force='mesh')
            mesh.vertices = (mesh.vertices - center_a) @ rotation.T * scale + center_b
            mesh.export(target)


def measure(path, bc_data):
    mesh = trimesh.load(path, force='mesh')
    occ = voxel_centers_inside(mesh, 64, bc_data['origin'], bc_data['pitch_xyz'])
    env = bc_data['bracket'].astype(bool)
    labels, count = label(occ)
    sizes = np.bincount(labels.ravel())[1:]
    row = {'case': next(name for name, p in CASES if p == path),
           'mesh': str(path), 'voxels': int(occ.sum()),
           'outside_envelope_fraction': round(float((occ & ~env).sum() / max(occ.sum(), 1)), 4),
           'voxel_components': int(count),
           'largest_voxel_component_fraction': round(float(sizes.max() / occ.sum()), 4) if len(sizes) else 0}
    result = audit(path, BC)
    row['bc_pass'] = result['bc_geometry_pass']
    row['bc_coverage'] = {k: v['coverage'] for k, v in result['regions'].items()}
    row['bc_connected'] = result['all_bc_same_6_connected_component']
    row['largest_mesh_component_fraction'] = result['largest_component_volume_fraction']
    row['mesh_components'] = result['mesh_components']
    return mesh, row


def main():
    physical_meshes()
    data = np.load(BC)
    envelope = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = envelope.bounds.mean(axis=0)
    radius = float(np.linalg.norm(envelope.extents)) * 1.5
    scale = float(envelope.extents.max()) / 2
    tile, header = 360, 28
    sheet = Image.new('RGB', (tile * len(VIEWS), (tile + header) * len(CASES)), 'white')
    draw = ImageDraw.Draw(sheet)
    rows = []
    for row, (name, path) in enumerate(CASES):
        mesh, item = measure(path, data)
        rows.append(item)
        for col, (view, elev, azim) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rgb = render_lit(mesh, eye, center, up, size=tile, fit_extent=scale,
                             margin=1.15, color=(.54, .58, .62))
            sheet.paste(Image.fromarray(rgb).convert('RGB'),
                        (tile * col, row * (tile + header) + header))
            draw.text((tile * col + 8, row * (tile + header) + 5),
                      f'{name} / {view}', fill='#20252b')
    sheet.save(OUT / 'physical_comparison.png')
    (OUT / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    table = ''.join('<tr><td><a href="/' + str(path.relative_to(ROOT)) + '">' +
                    html.escape(item['case']) + '</a></td><td>' + str(item['voxels']) +
                    '</td><td>' + f"{item['outside_envelope_fraction']:.1%}" +
                    '</td><td>' + str(item['voxel_components']) +
                    '</td><td>' + f"{item['bc_coverage']['seat_load']:.0%}" +
                    '</td><td>' + str(item['bc_connected']) +
                    '</td><td>' + str(item['bc_pass']) + '</td></tr>'
                    for (_, path), item in zip(CASES, rows))
    (OUT / 'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Registered chair specification</title>
<style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;background:#f3f5f7;color:#17212b}article{background:white;border-radius:10px;padding:1.3rem;margin:1rem 0}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.4rem}</style>
<article><h1>고정된 물리 사양을 생성기 좌표에 등록</h1><p>동일 정면 입력, seed 42, CFG 7, dense 50 steps. 물리 사양은 그대로 두고 생성기에 주는 64³ BC·envelope만 역변환했다. 아래 모든 메시를 다시 물리 좌표계로 옮겨 원래 BC·envelope에 대해 평가했다. FEA는 사용하지 않았다.</p>
<img src="/experiments/chair/sofa_style_2026-09-28/open_arm/input_contact.png"></article>
<article><h2>3D 메시: 정면·측면·상면</h2><img src="physical_comparison.png"></article>
<article><h2>원래 물리 사양 기준</h2><table><tr><th>조건/OBJ</th><th>재료 voxel</th><th>envelope 밖</th><th>voxel 성분</th><th>좌석 BC</th><th>BC 한 몸체</th><th>BC 판정</th></tr>''' + table +
'''</table><p>최종 판정에는 여섯 BC 영역의 95% 이상 포함, 동일 6-connected 성분 및 최대 메시 성분 체적 99% 이상을 요구합니다. <a href="metrics.json">전체 수치</a> · <a href="calibration.json">좌표 변환</a> · <a href="REPORT.md">분석</a></p></article></html>''')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
