"""Compare chair candidates on one expanded envelope and FEA domain."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import sys

import numpy as np
from scipy.ndimage import generate_binary_structure, label
from scipy.spatial.transform import Rotation
from skimage.measure import marching_cubes
import trimesh

from chair_qd_long_protocol_2026_10_03 import BASE, OUT as MAIN
from evaluate_chair_protected_parts_qd_2026_10_03 import ANCHOR, cell_of, descriptor, front_mask, preview
import evaluate_chair_text_latent_qd_3d_2026_10_03 as fem
from make_chair_domain import ROOT

sys.path.insert(0, str(ROOT / 'codebase/code'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402

SPEC = MAIN / 'envelope_plus16_spec_2026-10-03'
OUT = MAIN / 'envelope_plus16_fea_2026-10-03'
GUIDED = MAIN / 'dense_thickness_guided_2026-10-03'
CASES = [('vanilla_baseline', MAIN / 'mesh_cases/qd_01/aligned_main.obj', False)] + [
    (name, GUIDED / name / 'generation/mesh.obj', True)
    for name in ('guided_zero', 'hard_0p01', 'hard_0p03', 'hard_0p06', 'rmin_30000')]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = np.load(SPEC / 'voxel.npz')
    env, bc = spec['bracket'].astype(bool), spec['bc'].astype(bool)
    origin, pitch = spec['origin'], spec['pitch_xyz']
    cal = json.loads((BASE / 'single_view_spec_2026-10-03/specification.json').read_text())['native_frame_registration']
    rotation = Rotation.from_euler('x', cal['rotation_x_degrees'], degrees=True).as_matrix()
    anchor = trimesh.load(ANCHOR, force='mesh', process=False)
    anchor_occ = voxel_centers_inside(anchor, 64, origin, pitch).astype(bool)
    z = origin[2] + (np.arange(64) + .5) * pitch[2]
    protected = (z <= .61)[None, None, :] | spec['back_load'].astype(bool)
    structure = generate_binary_structure(3, 1)
    rows = []
    for name, path, native in CASES:
        if not path.exists():
            continue
        folder = OUT / 'mesh_cases' / name
        folder.mkdir(parents=True, exist_ok=True)
        raw = trimesh.load(path, force='mesh', process=False)
        mesh = max(raw.split(only_watertight=False), key=lambda part: abs(part.volume))
        if native:
            mesh.vertices = ((mesh.vertices - np.asarray(cal['source_center_m'])) @ rotation.T *
                             cal['uniform_scale'] + np.asarray(cal['physical_center_m']))
        mesh.export(folder / 'aligned_main.obj')
        generated = voxel_centers_inside(mesh, 64, origin, pitch).astype(bool)
        candidate = (anchor_occ & protected) | (generated & ~protected)
        realized = (candidate & env) | bc
        outside = float((candidate & ~env).sum() / max(1, candidate.sum()))
        repair = float(((realized & ~candidate).sum() + (candidate & ~realized).sum()) /
                       max(1, realized.sum()))
        components = int(label(realized, structure)[1])
        desc = descriptor(front_mask(realized))
        cell = cell_of(desc)
        seat = float((candidate & spec['load']).sum() / spec['load'].sum())
        back = float((candidate & spec['back_load']).sum() / spec['back_load'].sum())
        gate = bool(mesh.is_watertight and outside <= .01 and repair <= .05 and
                    components == 1 and seat >= .5 and back >= .9 and cell is not None)
        np.savez_compressed(folder / 'realized_occupancy.npz', occupied=realized)
        vertices, faces, _, _ = marching_cubes(np.pad(realized, 1), .5)
        vertices = origin + (vertices - .5) * pitch
        combined = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        combined.export(folder / 'combined_voxel.obj')
        preview(combined, folder / 'preview.png')
        row = {'id': name, 'source_mesh': str(path), 'combined_mesh': str(folder / 'combined_voxel.obj'),
               'outside_fraction': outside, 'repair_fraction': repair, 'components': components,
               'seat_bc': seat, 'back_bc': back, 'descriptor': desc, 'cell': cell,
               'mass_liters': float(realized.sum() * np.prod(pitch) * 1000),
               'geometry_gate': gate, 'source_watertight': bool(mesh.is_watertight)}
        rows.append(row)
        print(name, 'geometry', gate, 'cell', cell, 'repair', round(repair, 3), flush=True)
    indices = np.argwhere(env)
    nodes = origin + (indices + .5) * pitch
    fem.SPEC = SPEC
    fem.OUT = OUT
    eligible = [row for row in rows if row['geometry_gate']]
    by = {}
    if eligible:
        first = eligible[0]
        by[first['id'], 'seat'] = fem.fea(first, 'seat', nodes, indices)
        tasks = [(row, mode) for row in eligible for mode in ('seat', 'back')
                 if not (row is first and mode == 'seat')]
        with ThreadPoolExecutor(max_workers=3) as pool:
            solved = list(pool.map(lambda task: (task[0]['id'], task[1],
                                                fem.fea(task[0], task[1], nodes, indices)), tasks))
        by.update({(name, mode): result for name, mode, result in solved})
    baseline = next((row for row in rows if row['id'] == 'vanilla_baseline'), None)
    for row in rows:
        if row['geometry_gate']:
            seat, back = by[row['id'], 'seat'], by[row['id'], 'back']
            row['fea_035'] = {'seat': seat, 'back': back}
            row['fea_valid'] = bool(seat['valid'] and back['valid'])
        else:
            row['fea_valid'] = False
    if baseline and baseline['fea_valid']:
        for row in rows:
            if row['fea_valid']:
                row['worst_compliance_ratio_to_same_envelope_baseline'] = max(
                    row['fea_035']['seat']['compliance_proxy'] /
                    baseline['fea_035']['seat']['compliance_proxy'],
                    row['fea_035']['back']['compliance_proxy'] /
                    baseline['fea_035']['back']['compliance_proxy'])
    zero = next((row for row in rows if row['id'] == 'guided_zero'), None)
    if zero and zero['fea_valid']:
        for row in rows:
            if row['fea_valid']:
                row['worst_compliance_ratio_to_guided_zero'] = max(
                    row['fea_035']['seat']['compliance_proxy'] /
                    zero['fea_035']['seat']['compliance_proxy'],
                    row['fea_035']['back']['compliance_proxy'] /
                    zero['fea_035']['back']['compliance_proxy'])
    for row in rows:
        (OUT / 'mesh_cases' / row['id'] / 'evaluation.json').write_text(json.dumps(row, indent=2) + '\n')
    (OUT / 'result.json').write_text(json.dumps({
        'status': 'same expanded envelope/BC/FEM-domain ablation',
        'spec': str(SPEC), 'rows': rows,
        'quality_note': 'worst of two separate load-case compliance ratios to vanilla baseline on same expanded FEA domain',
        'limitations': ['35mm repaired voxel FEA proxy, not raw OBJ stress',
                        'one image and one seed', 'dense 64^3 thickness loss is not a hard minimum thickness']}, indent=2) + '\n')
    cards = []
    for row in rows:
        name = row['id']
        cards.append(f'<article><h2>{name}</h2><img src="mesh_cases/{name}/preview.png">'
                     f'<p>mass {row["mass_liters"]:.2f} L · outside {row["outside_fraction"]:.1%} · '
                     f'repair {row["repair_fraction"]:.1%} · cell {row["cell"]} · FEA {row["fea_valid"]} · '
                     f'worst C / baseline {row.get("worst_compliance_ratio_to_same_envelope_baseline", float("nan")):.3f}</p>'
                     f'<a href="mesh_cases/{name}/combined_voxel.obj">OBJ</a> · '
                     f'<a href="mesh_cases/{name}/evaluation.json">data</a></article>')
    (OUT / 'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>Expanded-envelope chair FEA</title>'
        '<style>body{font:16px/1.5 system-ui;max-width:1300px;margin:30px auto;background:#eef3f5;color:#1b2c39}'
        'article{background:white;padding:17px;margin:16px 0;border-radius:10px}img{max-width:800px;width:100%}'
        'a{color:#155f94}</style><h1>+15.8 mm envelope · dense 두께 loss · 동일 FEA 도메인</h1>'
        '<p>BC/keepout은 원본과 동일하다. 모든 결과를 확장 envelope과 같은 35 mm FEA 도메인에서 다시 평가했다. '
        '<a href="result.json">수치</a></p>' + ''.join(cards))
    print('FEA valid', sum(r['fea_valid'] for r in rows), flush=True)


if __name__ == '__main__':
    main()
