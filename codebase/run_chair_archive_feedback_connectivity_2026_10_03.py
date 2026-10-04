#!/usr/bin/env python3
"""Exploratory pruning of tiny, BC-free 64^3 components in the QD round."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
from scipy.ndimage import label
from skimage.measure import marching_cubes
import trimesh

from make_chair_domain import ROOT
from build_chair_single_view_qd_archive_2026_10_03 import features, cell_for
import evaluate_chair_text_latent_qd_3d_2026_10_03 as prior_fea

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
ROUND = BASE / 'archive_feedback_round_02_2026-10-03'
OUT = ROUND / 'connectivity_diagnostic'
CASE = OUT / 'mesh_cases/pruned'
SPEC = BASE / 'single_view_spec_2026-10-03'
NAME = 'round__diagonal__standard'


def main() -> None:
    CASE.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROUND / 'result.json').read_text())
    row = next(r for r in old['rows'] if r['id'] == NAME)
    spec = np.load(SPEC / 'voxel.npz')
    occ = np.load(ROUND / NAME / 'realized_occupancy.npz')['occupied'].astype(bool)
    six = np.zeros((3, 3, 3), bool)
    six[1, 1, 1] = True
    for axis in range(3):
        for sign in (-1, 1):
            xyz = [1, 1, 1]
            xyz[axis] += sign
            six[tuple(xyz)] = True
    labels, n = label(occ, structure=six)
    bc = spec['bc'].astype(bool)
    components = []
    for i in range(1, n + 1):
        mask = labels == i
        components.append({'label': i, 'voxels': int(mask.sum()),
                           'bc_voxels': int((mask & bc).sum()),
                           'seat_voxels': int((mask & spec['load']).sum()),
                           'back_voxels': int((mask & spec['back_load']).sum()),
                           'fix_voxels': int((mask & spec['fix']).sum())})
    components.sort(key=lambda r: -r['voxels'])
    main = components[0]
    discarded = [c for c in components[1:] if c['bc_voxels'] == 0]
    other = [c for c in components[1:] if c['bc_voxels'] > 0]
    removed = sum(c['voxels'] for c in discarded)
    fraction = removed / int(occ.sum())
    # Explicit exploratory rule; leave the original archive intact.
    qualified = not other and fraction <= .002 and main['bc_voxels'] == int(bc.sum())
    cleaned = labels == main['label']
    assert qualified and int(cleaned.sum()) + removed == int(occ.sum())
    np.savez_compressed(CASE / 'realized_occupancy.npz', occupied=cleaned)
    verts, faces, _, _ = marching_cubes(np.pad(cleaned, 1), level=.5,
                                         spacing=tuple(spec['pitch_xyz']))
    verts += spec['origin'] - spec['pitch_xyz'] * .5
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(CASE / 'pruned_voxel_proxy.obj')
    descriptor = features(cleaned, spec)
    cell = cell_for(descriptor)
    indices = np.argwhere(spec['bracket'].astype(bool))
    nodes = spec['origin'] + (indices + .5) * spec['pitch_xyz']
    prior_fea.OUT = OUT
    fea_row = {'id': 'pruned'}
    with ThreadPoolExecutor(max_workers=2) as pool:
        seat, back = list(pool.map(lambda mode: prior_fea.fea(fea_row, mode, nodes, indices),
                                   ('seat', 'back')))
    baseline = json.loads((BASE / 'single_view_qd_2026-10-03/summary.json').read_text())
    seat_ref = baseline['baseline_compliance_proxy']['seat']
    back_ref = baseline['baseline_compliance_proxy']['back']
    worst = (max(seat['compliance_proxy'] / seat_ref, back['compliance_proxy'] / back_ref)
             if seat['valid'] and back['valid'] else None)
    result = {'status': 'exploratory post hoc connectivity ablation; original strict archive unchanged',
              'source_case': str(ROUND / NAME), 'component_rule': 'remove BC-free components if total <=0.2% of realized voxels',
              'components_before': components, 'removed_voxels': removed,
              'removed_fraction': fraction, 'qualified': qualified,
              'connected_after': int(label(cleaned, structure=six)[1]) == 1,
              'cell_before': row['cell'], 'cell_after': cell,
              'descriptor_after': {'side_open_fraction': descriptor[0],
                                   'backrest_taper_ratio': descriptor[1]},
              'mass_liters_after': float(cleaned.sum() * np.prod(spec['pitch_xyz']) * 1000),
              'repair_fraction_upper_bound': row['repair_fraction'] + fraction,
              'fea_035': {'seat': seat, 'back': back},
              'worst_compliance_ratio_to_baseline': worst,
              'main_archive_occupied_before_and_after': [old['occupied_before'], old['occupied_after']],
              'limitations': ['Post hoc gate change cannot be counted as a prespecified new elite.',
                              'FEM evaluates 35 mm repaired voxel density, not the raw OBJ surface.']}
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'REPORT.md').write_text(
        '# Exploratory connectivity diagnostic\n\n'
        f"The QD candidate has {len(components)} 6-connected components. The main component contains all BC voxels; "
        f"two fragments total {removed} voxels ({fraction:.3%}) and contain no BC.\n\n"
        f"After removing only these fragments, the 3D cell is {cell} and the two-load 35 mm FEM proxy "
        f"{'succeeded' if worst is not None else 'failed'}"
        f"{f' (worst compliance ratio {worst:.3f})' if worst is not None else ''}.\n\n"
        'This rule was introduced after inspecting this candidate. Keep the strict primary archive at 3/9; '
        'preregister and re-evaluate uniformly before treating this as a new elite.\n\n'
        f"Result: {OUT / 'result.json'}\nMesh: {CASE / 'pruned_voxel_proxy.obj'}\n")
    print('removed', removed, 'fraction', fraction, 'cell', cell,
          'worst compliance', worst, flush=True)


if __name__ == '__main__':
    main()
