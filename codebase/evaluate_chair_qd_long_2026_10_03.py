"""Prospective four-vs-four chair QD evaluation on frozen BC and FEM proxies."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import html
import json
import sys

import numpy as np
from scipy.ndimage import label
from scipy.spatial.transform import Rotation
from skimage.measure import marching_cubes
import trimesh

from chair_qd_long_protocol_2026_10_03 import BASE, OUT, CASES
from make_chair_domain import ROOT
from evaluate_chair_protected_parts_qd_2026_10_03 import (
    ANCHOR, SPEC, RANGES, cell_of, descriptor, front_mask, preview,
)
import evaluate_chair_text_latent_qd_3d_2026_10_03 as fea_module

sys.path.insert(0, str(ROOT / 'codebase/code'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402


def main() -> None:
    protocol = json.loads((OUT / 'protocol.json').read_text())
    image_by = {r['id']: r for r in json.loads((OUT / 'image_metrics.json').read_text())}
    spec = np.load(SPEC / 'voxel.npz')
    env, bc = spec['bracket'].astype(bool), spec['bc'].astype(bool)
    origin, pitch = spec['origin'], spec['pitch_xyz']
    anchor_mesh = trimesh.load(ANCHOR, force='mesh', process=False)
    anchor_occ = voxel_centers_inside(anchor_mesh, 64, origin, pitch).astype(bool)
    z = origin[2] + (np.arange(64) + .5) * pitch[2]
    protected = (z <= .61)[None, None, :] | spec['back_load'].astype(bool)
    calibration = json.loads((SPEC / 'specification.json').read_text())['native_frame_registration']
    rotation = Rotation.from_euler('x', calibration['rotation_x_degrees'], degrees=True).as_matrix()
    six = np.zeros((3, 3, 3), bool)
    six[1, 1, 1] = True
    for axis in range(3):
        for sign in (-1, 1):
            cell = [1, 1, 1]
            cell[axis] += sign
            six[tuple(cell)] = True
    rows = []
    for item in CASES:
        name = item['id']
        folder = OUT / name
        case = OUT / 'mesh_cases' / name
        case.mkdir(parents=True, exist_ok=True)
        raw = trimesh.load(folder / 'generation/mesh.obj', force='mesh', process=False)
        main = max(raw.split(only_watertight=False), key=lambda piece: abs(piece.volume))
        main.vertices = ((main.vertices - np.asarray(calibration['source_center_m'])) @ rotation.T *
                         calibration['uniform_scale'] + np.asarray(calibration['physical_center_m']))
        main.export(case / 'aligned_main.obj')
        generated = voxel_centers_inside(main, 64, origin, pitch).astype(bool)
        candidate = (anchor_occ & protected) | (generated & ~protected)
        realized = (candidate & env) | bc
        added = int((realized & ~candidate).sum())
        removed = int((candidate & ~realized).sum())
        repair = (added + removed) / max(1, int(realized.sum()))
        components = int(label(realized, structure=six)[1])
        seat = float((candidate & spec['load']).sum() / spec['load'].sum())
        back = float((candidate & spec['back_load']).sum() / spec['back_load'].sum())
        outside = float((candidate & ~env).sum() / max(1, int(candidate.sum())))
        desc = descriptor(front_mask(realized))
        cell = cell_of(desc)
        image_gate = image_by[name]['projected_interface_gate'] and image_by[name]['projected_back_load_gate']
        reasons = []
        if seat < .5: reasons.append('seat BC')
        if back < .9: reasons.append('back BC')
        if outside > .01: reasons.append('outside envelope')
        if repair > .05: reasons.append('repair > 5%')
        if components != 1: reasons.append('disconnected')
        if not main.is_watertight: reasons.append('generated main mesh not watertight')
        if cell is None: reasons.append('descriptor out of range')
        np.savez_compressed(case / 'realized_occupancy.npz', occupied=realized)
        vertices, faces, _, _ = marching_cubes(np.pad(realized, 1), .5)
        vertices = origin + (vertices - .5) * pitch
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.export(case / 'combined_voxel.obj')
        preview(mesh, case / 'combined_preview.png')
        row = {'id': name, 'method': item.get('method', 'targeted' if name.startswith('qd_') else 'control'),
               'target_cell': item.get('target_cell'), 'source_image': str(OUT / f'{name}.png'),
               'generated_mesh': str(folder / 'generation/mesh.obj'),
               'aligned_mesh': str(case / 'aligned_main.obj'),
               'combined_mesh': str(case / 'combined_voxel.obj'),
               'image_metrics': image_by[name]['metrics'], 'image_gate': bool(image_gate),
               'seat_bc': seat, 'back_bc': back, 'outside_fraction': outside,
               'repair_fraction': repair, 'components': components,
               'source_watertight': bool(main.is_watertight),
               'descriptor': desc, 'cell': cell,
               'mass_liters': float(realized.sum() * np.prod(pitch) * 1000),
               'geometry_reasons': reasons, 'geometry_gate': not reasons}
        rows.append(row)
        print(name, 'cell', cell, 'image', image_gate, 'geometry', not reasons,
              'repair', round(repair, 3), flush=True)
    indices = np.argwhere(env)
    nodes = origin + (indices + .5) * pitch
    fea_module.OUT = OUT
    tasks = [(row, mode) for row in rows if row['geometry_gate'] for mode in ('seat', 'back')]
    with ThreadPoolExecutor(max_workers=3) as pool:
        solved = list(pool.map(lambda task: (task[0]['id'], task[1],
                                            fea_module.fea(task[0], task[1], nodes, indices)), tasks))
    by = {(name, mode): result for name, mode, result in solved}
    reference = json.loads((BASE / 'single_view_qd_2026-10-03/summary.json').read_text())['baseline_compliance_proxy']
    for row in rows:
        if row['geometry_gate']:
            seat, back = by[row['id'], 'seat'], by[row['id'], 'back']
            row['fea_035'] = {'seat': seat, 'back': back}
            row['fea_valid'] = bool(seat['valid'] and back['valid'])
            if row['fea_valid']:
                row['worst_compliance_ratio'] = max(seat['compliance_proxy'] / reference['seat'],
                                                    back['compliance_proxy'] / reference['back'])
        else:
            row['fea_valid'] = False
        row['strict_eligible'] = bool(row['image_gate'] and row['geometry_gate'] and row['fea_valid'])
        (OUT / 'mesh_cases' / row['id'] / 'evaluation.json').write_text(json.dumps(row, indent=2) + '\n')
    pilot = json.loads((BASE / 'text_reasoned_front_axes_2026-10-03/protected_parts_qd/result.json').read_text())
    initial_file = OUT / 'initial_archive.json'
    initial = ({tuple(cell) for cell in json.loads(initial_file.read_text())['occupied_cells']}
               if initial_file.exists() else
               {tuple(map(int, key.split(','))) for key in pilot['archive']})
    progression = {}
    archives = {}
    methods = list(dict.fromkeys(row['method'] for row in rows))
    for method in methods:
        occupied = set(initial)
        progress = []
        for row in [r for r in rows if r['method'] == method]:
            if row['strict_eligible']:
                occupied.add(tuple(row['cell']))
            progress.append(len(occupied))
        progression[method] = progress
        archives[method] = sorted([list(cell) for cell in occupied])
    result = {'protocol': str(OUT / 'protocol.json'),
              'status': protocol['status'],
              'initial_occupied_cells': len(initial), 'progression': progression,
              'final_archives': archives, 'rows': rows,
              'fea_scope': 'separate 800N seat and 200N back load cases on 35mm repaired voxel proxy',
              'limitations': protocol['limitations']}
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    cards = []
    for row in rows:
        name = row['id']
        cards.append(f'''<article><h2>{html.escape(name)} · {row['method']} · 3D cell {row['cell']}</h2>
<div class="pair"><img src="{name}.png" alt="Input image"><img src="mesh_cases/{name}/combined_preview.png" alt="3D mesh"></div>
<p>image gate {row['image_gate']} · geometry gate {row['geometry_gate']} · FEA {row['fea_valid']} · strict eligible <b>{row['strict_eligible']}</b></p>
<p>mass {row['mass_liters']:.2f} L · worst compliance/reference {row.get('worst_compliance_ratio', float('nan')):.3f} · BC seat/back {row['seat_bc']:.0%}/{row['back_bc']:.0%} · repair {row['repair_fraction']:.1%}</p>
<p>front arm aperture {row['descriptor']['front_arm_aperture_fraction']:.3f} · upper span {row['descriptor']['front_upper_span_ratio']:.3f} · <a href="mesh_cases/{name}/combined_voxel.obj">OBJ</a> · <a href="mesh_cases/{name}/evaluation.json">data</a></p></article>''')
    progress_text = ' · '.join(f'{method} {values}' for method, values in progression.items())
    (OUT / 'index.html').write_text(f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chair image QD → 3D → FEA</title><style>body{{font:16px/1.55 system-ui;max-width:1480px;margin:30px auto;padding:0 20px;background:#f0f3f6;color:#1c2c38}}article{{background:white;border:1px solid #d9e1e7;border-radius:14px;padding:18px;margin:18px 0}}.pair{{display:flex;gap:14px}}.pair img{{width:49%;height:350px;object-fit:contain;background:white}}a{{color:#155c98}}</style>
<h1>의자 이미지 QD → 3D → FEA</h1>
<p>시작 archive {len(initial)}/9 셀에서 {progress_text}. 동일한 Direct3D-S2 설정·seed 42, BC 보호 구조, 35 mm FEA를 적용했다.</p>
<p>엄격한 적격성은 이미지 투영 BC, 3D 연결·envelope·BC, 좌면과 등받이 개별 FEA가 모두 통과하는 경우다. FEA는 원본 OBJ 직접 해석이 아닌 repaired voxel proxy이다. <a href="protocol.json">실험 프로토콜</a> · <a href="result.json">전체 결과</a></p>{''.join(cards)}</html>''')
    print('archive', progression, flush=True)


if __name__ == '__main__':
    main()
