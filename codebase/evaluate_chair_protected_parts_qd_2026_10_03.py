"""Exploratory QD evaluation with shared BC-bearing chair geometry.

The lower seat/legs and central back-load patch are part of the specification,
not generated variations. All remaining voxels come from each image-conditioned
Direct3D-S2 mesh. This pilot does not retroactively change the frozen rounds 3–5.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import html
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import label
from scipy.spatial.transform import Rotation
from skimage.measure import marching_cubes
import trimesh

from make_chair_domain import ROOT
import evaluate_chair_text_latent_qd_3d_2026_10_03 as prior_fea

sys.path.insert(0, str(ROOT / 'codebase/code'))
sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SRC = BASE / 'text_reasoned_front_axes_2026-10-03'
OUT = SRC / 'protected_parts_qd'
SPEC = BASE / 'single_view_spec_2026-10-03'
ANCHOR = BASE / 'archive_feedback_loop_2026-10-03/round_04/round__straight__standard/aligned_main.obj'
RANGES = {'front_arm_aperture_fraction': [0.0, 0.45],
          'front_upper_span_ratio': [0.5, 1.2]}
CASES = [('narrow_open', 1), ('wide_open', 1),
         ('narrow_closed', 2), ('wide_closed', 2),
         ('tapered_triangular_perforations', 3), ('flared_oval_perforations', 3)]


def front_mask(occ: np.ndarray) -> np.ndarray:
    front = occ.any(axis=1).T[::-1, :].astype('uint8')
    yy, xx = np.where(front)
    crop = front[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    h, w = crop.shape
    scale = min(448 / w, 448 / h)
    resized = np.asarray(Image.fromarray(crop).resize(
        (round(w * scale), round(h * scale)), Image.Resampling.NEAREST)).astype(bool)
    mask = np.zeros((512, 512), bool)
    y0, x0 = (512 - resized.shape[0]) // 2, (512 - resized.shape[1]) // 2
    mask[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return mask


def descriptor(mask: np.ndarray) -> dict:
    top = np.where(mask[int(.09 * 512):int(.18 * 512), int(.1 * 512):int(.9 * 512)])[1]
    middle = np.where(mask[int(.37 * 512):int(.45 * 512), int(.1 * 512):int(.9 * 512)])[1]
    def span(xx):
        return float(np.percentile(xx, 95) - np.percentile(xx, 5)) if len(xx) > 10 else 0.
    width = span(top) / span(middle) if span(middle) else 0.
    y0, y1 = int(.23 * 512), int(.38 * 512)
    left = mask[y0:y1, int(.24 * 512):int(.37 * 512)]
    right = mask[y0:y1, int(.63 * 512):int(.76 * 512)]
    aperture = float(1 - (left.mean() + right.mean()) / 2)
    white = ~mask
    labels, n = label(white)
    holes = []
    for k in range(1, n + 1):
        yy, xx = np.where(labels == k)
        if not len(xx) or xx.min() == 0 or yy.min() == 0 or xx.max() == 511 or yy.max() == 511:
            continue
        cx, cy = xx.mean() / 512, yy.mean() / 512
        if .35 < cx < .65 and .05 < cy < .22 and len(xx) > 100:
            holes.append({'center': [float(cx), float(cy)], 'area_px': int(len(xx))})
    return {'front_arm_aperture_fraction': aperture,
            'front_upper_span_ratio': float(width),
            'upper_back_projected_holes': holes}


def cell_of(d: dict) -> list[int] | None:
    cell = []
    for key, (lo, hi) in RANGES.items():
        value = d[key]
        if not lo <= value <= hi:
            return None
        cell.append(min(2, int((value - lo) / (hi - lo) * 3)))
    return cell


def preview(mesh: trimesh.Trimesh, output: Path) -> None:
    canvas = Image.new('RGB', (730, 370), '#f6f8fa')
    for i, (elev, azim) in enumerate(((15, 0), (15, 90))):
        target = np.array([0., .01, .46])
        eye, up = camera_from_elev_azim(target, 2., elev, azim)
        rgb = render_lit(mesh, eye, target, up, size=350, fit_extent=.56,
                         color=(.64, .69, .73))
        canvas.paste(Image.fromarray(rgb).convert('RGB'), (i * 365, 10))
    canvas.save(output)


def aligned_mesh(name: str, pair: int, cal: dict) -> trimesh.Trimesh:
    case = SRC / f'pair_{pair:02d}' / name
    path = case / 'aligned_main.obj'
    if path.exists():
        return trimesh.load(path, force='mesh', process=False)
    raw = trimesh.load(case / 'generation/mesh.obj', force='mesh', process=False)
    mesh = max(raw.split(only_watertight=False), key=lambda part: abs(part.volume))
    rotation = Rotation.from_euler('x', cal['rotation_x_degrees'], degrees=True).as_matrix()
    mesh.vertices = ((mesh.vertices - np.asarray(cal['source_center_m'])) @ rotation.T *
                     cal['uniform_scale'] + np.asarray(cal['physical_center_m']))
    mesh.export(path)
    return mesh


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = np.load(SPEC / 'voxel.npz')
    env, bc = spec['bracket'].astype(bool), spec['bc'].astype(bool)
    origin, pitch = spec['origin'], spec['pitch_xyz']
    anchor_mesh = trimesh.load(ANCHOR, force='mesh', process=False)
    anchor_occ = voxel_centers_inside(anchor_mesh, 64, origin, pitch).astype(bool)
    z = origin[2] + (np.arange(64) + .5) * pitch[2]
    protected = (z <= .61)[None, None, :] | spec['back_load'].astype(bool)
    if float((anchor_occ & spec['load']).sum() / spec['load'].sum()) < .5:
        raise RuntimeError('Shared anchor does not cover the required seat load patch')
    if not np.all(anchor_occ[spec['back_load'].astype(bool)]):
        raise RuntimeError('Shared anchor does not cover the central back load patch')
    cal = json.loads((SPEC / 'specification.json').read_text())['native_frame_registration']
    six = np.zeros((3, 3, 3), bool)
    six[1, 1, 1] = True
    for axis in range(3):
        for sign in (-1, 1):
            ijk = [1, 1, 1]
            ijk[axis] += sign
            six[tuple(ijk)] = True
    rows = []
    for name, pair in CASES:
        original = SRC / f'pair_{pair:02d}' / name
        if not (original / 'generation/mesh.obj').exists():
            continue
        mesh = aligned_mesh(name, pair, cal)
        generated = voxel_centers_inside(mesh, 64, origin, pitch).astype(bool)
        candidate = (anchor_occ & protected) | (generated & ~protected)
        realized = (candidate & env) | bc
        added = int((realized & ~candidate).sum())
        removed = int((candidate & ~realized).sum())
        repair = (added + removed) / max(1, int(realized.sum()))
        components = int(label(realized, structure=six)[1])
        seat = float((candidate & spec['load']).sum() / spec['load'].sum())
        back = float((candidate & spec['back_load']).sum() / spec['back_load'].sum())
        outside = float((candidate & ~env).sum() / max(1, int(candidate.sum())))
        mask = front_mask(realized)
        d = descriptor(mask)
        cell = cell_of(d)
        reasons = []
        if seat < .5: reasons.append('seat BC')
        if back < .9: reasons.append('back BC')
        if outside > .01: reasons.append('outside envelope')
        if repair > .05: reasons.append('repair > 5%')
        if components != 1: reasons.append('disconnected')
        if not mesh.is_watertight: reasons.append('generated main mesh not watertight')
        if cell is None: reasons.append('new descriptor out of range')
        case = OUT / 'mesh_cases' / name
        case.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(case / 'realized_occupancy.npz', occupied=realized)
        np.savez_compressed(case / 'source_occupancies.npz', anchor=anchor_occ,
                            generated=generated, protected=protected)
        Image.fromarray((mask * 255).astype('uint8')).save(case / 'front_projection.png')
        vertices, faces, _, _ = marching_cubes(np.pad(realized, 1), .5)
        vertices = origin + (vertices - .5) * pitch
        outmesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        outmesh.export(case / 'combined_voxel.obj')
        preview(outmesh, case / 'combined_preview.png')
        row = {'id': name, 'pair': pair, 'source_image': str(original / 'input.png'),
               'generated_mesh': str(original / 'aligned_main.obj'),
               'combined_mesh': str(case / 'combined_voxel.obj'),
               'seat_bc': seat, 'back_bc': back, 'outside_fraction': outside,
               'repair_fraction': repair, 'components': components,
               'descriptor': d, 'cell': cell,
               'mass_liters': float(realized.sum() * np.prod(pitch) * 1000),
               'source_watertight': bool(mesh.is_watertight),
               'geometry_reasons': reasons, 'geometry_gate': not reasons}
        rows.append(row)
        print(name, 'cell', cell, 'BC', seat, back, 'repair', round(repair, 3),
              'holes', len(d['upper_back_projected_holes']), 'gate', not reasons, flush=True)
    indices = np.argwhere(env)
    nodes = origin + (indices + .5) * pitch
    prior_fea.OUT = OUT
    tasks = [(row, mode) for row in rows if row['geometry_gate'] for mode in ('seat', 'back')]
    with ThreadPoolExecutor(max_workers=3) as pool:
        fea = list(pool.map(lambda task: (task[0]['id'], task[1],
                                          prior_fea.fea(task[0], task[1], nodes, indices)), tasks))
    fea_by = {(name, mode): value for name, mode, value in fea}
    baseline = json.loads((BASE / 'single_view_qd_2026-10-03/summary.json').read_text())
    refs = baseline['baseline_compliance_proxy']
    archive = {}
    for row in rows:
        if row['geometry_gate']:
            a, b = fea_by[row['id'], 'seat'], fea_by[row['id'], 'back']
            row['fea_035'] = {'seat': a, 'back': b}
            row['fea_valid'] = bool(a['valid'] and b['valid'])
            if row['fea_valid']:
                row['worst_compliance_ratio'] = max(a['compliance_proxy'] / refs['seat'],
                                                    b['compliance_proxy'] / refs['back'])
                key = ','.join(map(str, row['cell']))
                archive.setdefault(key, []).append(row['id'])
        else:
            row['fea_valid'] = False
        (OUT / 'mesh_cases' / row['id'] / 'evaluation.json').write_text(json.dumps(row, indent=2) + '\n')
    result = {'status': 'exploratory protected-parts pilot; bins chosen after image calibration',
              'fixed_anchor': str(ANCHOR), 'cut_z_m': .61,
              'protected_region': 'anchor at z<=0.61 m and exact central back-load voxels',
              'descriptor_ranges': RANGES, 'archive_dims': [3, 3],
              'occupied_cells': len(archive), 'archive': archive, 'rows': rows,
              'fea_scope': '35 mm repaired voxel proxy, not raw-OBJ direct FEA'}
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    cards = []
    for row in rows:
        name = row['id']
        cards.append(f'''<article><h2>{html.escape(name)} · cell {row['cell']}</h2>
<div class="imgs"><img src="../pair_{row['pair']:02d}/{name}/input.png"><img src="mesh_cases/{name}/combined_preview.png"></div>
<p>seat/back BC {row['seat_bc']:.0%}/{row['back_bc']:.0%} · repair {row['repair_fraction']:.1%} · connected {row['components']==1} · FEA {'valid' if row['fea_valid'] else 'not run'}</p>
<p>front aperture {row['descriptor']['front_arm_aperture_fraction']:.3f} · upper span {row['descriptor']['front_upper_span_ratio']:.3f} · projected back holes {len(row['descriptor']['upper_back_projected_holes'])}</p>
<p>mass {row['mass_liters']:.2f} L · worst compliance ratio {row.get('worst_compliance_ratio', float('nan')):.3f}</p>
<p><a href="mesh_cases/{name}/combined_voxel.obj">combined OBJ</a> · <a href="mesh_cases/{name}/evaluation.json">data</a></p></article>''')
    (OUT / 'index.html').write_text(f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Protected-parts chair QD pilot</title>
<style>body{{font:16px/1.5 system-ui;max-width:1500px;margin:auto;padding:24px;background:#f3f6f8;color:#152332}}article{{background:white;border:1px solid #d4dfe6;border-radius:12px;padding:15px;margin:20px 0}}.imgs{{display:flex;gap:14px}}.imgs img{{width:49%;object-fit:contain;background:white}}a{{color:#155b91}}</style>
<h1>BC 보호 부위 + 생성 상부 형상: 탐색적 QD 실험</h1><p>좌면·다리(z≤0.61 m)와 중앙 등받이 하중 부위를 모든 후보에 동일하게 고정하고 나머지를 각 이미지의 Direct3D-S2 결과로 채웠다. 새로운 3×3 셀은 정면 팔걸이 개방률과 상단 폭 비율로 정의한다. 현재 유효한 3D 점유 셀: {len(archive)}/9.</p>
<p>이 실험의 범위와 bins는 앞선 이미지/3D 진단 후 정했으므로 성능 우월성의 검증 결과가 아니다. 3D FEA는 35 mm repaired voxel proxy다. <a href="result.json">전체 수치</a> · <a href="../index.html">이미지 실험</a></p>{''.join(cards)}</html>''')
    print('occupied', len(archive), 'of 9', 'FEA valid', sum(r['fea_valid'] for r in rows))


if __name__ == '__main__':
    main()
