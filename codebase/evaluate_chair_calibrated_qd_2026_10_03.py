"""Evaluate selected chair image-QD candidates on one expanded BC/FEA spec."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys

import numpy as np
from scipy.ndimage import generate_binary_structure, label
from scipy.spatial.transform import Rotation
from skimage.measure import marching_cubes
import trimesh

from chair_qd_long_protocol_2026_10_03 import BASE
from evaluate_chair_protected_parts_qd_2026_10_03 import ANCHOR, cell_of, descriptor, front_mask, preview
import evaluate_chair_text_latent_qd_3d_2026_10_03 as fem
from make_chair_domain import ROOT

sys.path.insert(0, str(ROOT / 'codebase/code'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402


OUT = BASE / 'text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03'
PRIOR = BASE / 'text_reasoned_front_axes_2026-10-03/long_qd_fea_2026-10-03'
SPEC = PRIOR / 'envelope_plus16_spec_2026-10-03'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, choices=(1, 2), required=True)
    parser.add_argument('--sparse-fea', action='store_true')
    parser.add_argument('--ids', default='', help='comma-separated subset for pilot evaluation')
    parser.add_argument('--overflow-aperture-max', type=float, default=0.,
                        help='Optional fourth aperture cell [.45,max] for exploratory out-of-range phenotypes')
    args = parser.parse_args()
    root = OUT / 'sparse_fea_loop_2026-10-03' if args.sparse_fea else OUT
    folder = root / f'round_{args.round:02d}'
    selection = json.loads((folder / 'selection.json').read_text())['selected']
    if args.ids:
        requested = set(args.ids.split(','))
        selection = [row for row in selection if row['id'] in requested]
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
    for selected in selection:
        name = selected['id']
        case = folder / 'mesh_cases' / name
        case.mkdir(parents=True, exist_ok=True)
        raw = trimesh.load(folder / name / 'generation/mesh.obj', force='mesh', process=False)
        main_mesh = max(raw.split(only_watertight=False), key=lambda part: abs(part.volume))
        main_mesh.vertices = ((main_mesh.vertices - np.asarray(cal['source_center_m'])) @ rotation.T *
                              cal['uniform_scale'] + np.asarray(cal['physical_center_m']))
        main_mesh.export(case / 'aligned_main.obj')
        generated = voxel_centers_inside(main_mesh, 64, origin, pitch).astype(bool)
        candidate = (anchor_occ & protected) | (generated & ~protected)
        realized = (candidate & env) | bc
        outside = float((candidate & ~env).sum() / max(1, candidate.sum()))
        repair = float(((realized & ~candidate).sum() + (candidate & ~realized).sum()) /
                       max(1, realized.sum()))
        components = int(label(realized, structure)[1])
        desc = descriptor(front_mask(realized))
        frozen_cell = cell_of(desc)
        cell = frozen_cell
        if cell is None and args.overflow_aperture_max > .45:
            aperture = desc['front_arm_aperture_fraction']
            width = desc['front_upper_span_ratio']
            if .45 < aperture <= args.overflow_aperture_max and .5 <= width <= 1.2:
                cell = [3, min(2, int((width - .5) / .7 * 3))]
        seat = float((candidate & spec['load']).sum() / spec['load'].sum())
        back = float((candidate & spec['back_load']).sum() / spec['back_load'].sum())
        reasons = []
        if seat < .5: reasons.append('seat BC')
        if back < .9: reasons.append('back BC')
        if outside > .01: reasons.append('outside envelope')
        if repair > .05: reasons.append('repair > 5%')
        if components != 1: reasons.append('disconnected')
        if not main_mesh.is_watertight: reasons.append('source not watertight')
        if cell is None: reasons.append('descriptor out of range')
        np.savez_compressed(case / 'realized_occupancy.npz', occupied=realized)
        vertices, faces, _, _ = marching_cubes(np.pad(realized, 1), .5)
        vertices = origin + (vertices - .5) * pitch
        combined = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        combined.export(case / 'combined_voxel.obj')
        preview(combined, case / 'preview.png')
        row = {'id': name, 'method': selected['method'], 'round': args.round,
               'image': selected['image'], 'image_metrics': selected['image_metrics'],
               'image_cell': selected['image_cell'], 'image_gate': selected['image_gate'],
               'generated_mesh': str(folder / name / 'generation/mesh.obj'),
               'aligned_mesh': str(case / 'aligned_main.obj'),
               'combined_mesh': str(case / 'combined_voxel.obj'),
               'preview': str(case / 'preview.png'), 'descriptor': desc, 'cell': cell,
               'frozen_cell': frozen_cell,
               'outside_fraction': outside, 'repair_fraction': repair, 'components': components,
               'seat_bc': seat, 'back_bc': back, 'source_watertight': bool(main_mesh.is_watertight),
               'mass_liters': float(realized.sum() * np.prod(pitch) * 1000),
               'geometry_reasons': reasons, 'geometry_gate': not reasons}
        rows.append(row)
        print(name, 'image cell', row['image_cell'], '3D cell', cell,
              'geometry', not reasons, 'repair', round(repair, 3), flush=True)
    indices = np.argwhere(env)
    nodes = origin + (indices + .5) * pitch
    fem.SPEC = SPEC
    fem.OUT = folder
    tasks = [(row, mode) for row in rows if row['geometry_gate'] for mode in ('seat', 'back')]
    with ThreadPoolExecutor(max_workers=3) as pool:
        solved = list(pool.map(lambda task: (task[0]['id'], task[1],
                                            fem.fea(task[0], task[1], nodes, indices)), tasks))
    by = {(name, mode): result for name, mode, result in solved}
    prior_fea = json.loads((PRIOR / 'envelope_plus16_fea_2026-10-03/result.json').read_text())
    baseline = next(r for r in prior_fea['rows'] if r['id'] == 'vanilla_baseline')
    for row in rows:
        if row['geometry_gate']:
            seat, back = by[row['id'], 'seat'], by[row['id'], 'back']
            row['fea_035'] = {'seat': seat, 'back': back}
            row['fea_valid'] = bool(seat['valid'] and back['valid'])
            if row['fea_valid']:
                row['worst_compliance_ratio'] = max(
                    seat['compliance_proxy'] / baseline['fea_035']['seat']['compliance_proxy'],
                    back['compliance_proxy'] / baseline['fea_035']['back']['compliance_proxy'])
        else:
            row['fea_valid'] = False
        row['strict_eligible'] = bool(row['image_gate'] and row['geometry_gate'] and row['fea_valid'])
        (folder / 'mesh_cases' / row['id'] / 'evaluation.json').write_text(json.dumps(row, indent=2) + '\n')
    result = {'round': args.round, 'status': 'selected candidates evaluated on expanded envelope and identical BC/FEA domain',
              'overflow_aperture_max': args.overflow_aperture_max,
              'spec': str(SPEC), 'baseline_fea': str(PRIOR / 'envelope_plus16_fea_2026-10-03/result.json'),
              'rows': rows, 'limitations': ['35 mm repaired voxel FEA proxy, not raw OBJ stress',
                                            'two separate load cases, not simultaneous load']}
    result_name = 'evaluation_pilot.json' if args.ids else 'evaluation.json'
    (folder / result_name).write_text(json.dumps(result, indent=2) + '\n')
    print('strict eligible', sum(row['strict_eligible'] for row in rows), '/', len(rows), flush=True)


if __name__ == '__main__':
    main()
