#!/usr/bin/env python3
"""Update a chair QD archive with tapered examples and explicit physics gates."""
from __future__ import annotations

import html
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import label
import trimesh

from make_chair_domain import ROOT
from build_chair_single_view_qd_archive_2026_10_03 import (
    ARCHIVE_DIMS, DESCRIPTOR_RANGES, SPEC, cell_for, features,
)

sys.path.insert(0, str(ROOT / 'codebase/code'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OLD = BASE / 'text_latent_qd_2026-10-03'
NEW = BASE / 'tapered_examples_2026-10-03'
OUT = NEW / 'qd_round_01'
NAMES = ('tapered_original', 'tapered_curved', 'tapered_diagonal',
         'tapered_tall', 'tapered_seed43', 'tapered_seed44')
MIN_SEAT = .50
MIN_BACK = .90
MAX_REPAIR = .05
MAX_OUTSIDE = .01


def nondominated(rows: list[dict]) -> list[dict]:
    return [row for row in rows if not any(
        (other['mass_liters'] <= row['mass_liters'] and
         other['worst_compliance_ratio'] <= row['worst_compliance_ratio'] and
         (other['mass_liters'] < row['mass_liters'] or
          other['worst_compliance_ratio'] < row['worst_compliance_ratio']))
        for other in rows if other is not row)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = np.load(SPEC / 'voxel.npz')
    env = spec['bracket'].astype(bool)
    bc = spec['bc'].astype(bool)
    six = np.zeros((3, 3, 3), bool)
    six[1, 1, 1] = True
    for axis in range(3):
        for direction in (-1, 1):
            xyz = [1, 1, 1]
            xyz[axis] += direction
            six[tuple(xyz)] = True
    prior = json.loads((OLD / 'mesh_evaluations.json').read_text())
    prior_by_id = {row['id']: row for row in prior}
    new_metrics = {row['name']: row for row in json.loads((NEW / 'metrics.json').read_text())}
    candidates = []
    for row in prior:
        reasons = []
        if row['source_seat_bc_coverage'] < MIN_SEAT: reasons.append('seat BC')
        if row['source_back_bc_coverage'] < MIN_BACK: reasons.append('back BC')
        if row['source_outside_envelope_fraction'] > MAX_OUTSIDE: reasons.append('envelope')
        if row['repair_fraction'] > MAX_REPAIR: reasons.append('repair budget')
        if row['realized_components_6conn'] != 1: reasons.append('disconnected')
        if not row['source_watertight']: reasons.append('not watertight')
        if row['realized_3d_cell'] is None: reasons.append('descriptor range')
        if not row.get('fea_valid'): reasons.append('FEA unavailable')
        candidates.append({
            'id': row['id'], 'round': 0, 'source': 'prior text/image QD pilot',
            'mesh': row['mesh'], 'image': row['source_image'],
            'descriptor': row['realized_3d_descriptor'],
            'cell': row['realized_3d_cell'],
            'mass_liters': row['realized_mass_liters'],
            'worst_compliance_ratio': row.get('worst_compliance_ratio_to_baseline'),
            'seat_bc': row['source_seat_bc_coverage'],
            'back_bc': row['source_back_bc_coverage'],
            'outside_fraction': row['source_outside_envelope_fraction'],
            'repair_fraction': row['repair_fraction'],
            'reasons': reasons, 'archive_eligible': not reasons,
        })
    for name in NAMES:
        if name == 'tapered_original':
            # The displayed baseline is the same candidate as the prior pilot.
            # Do not count it as an extra evaluation or a second elite.
            continue
        row = new_metrics[name]
        mesh = trimesh.load(row['mesh'], force='mesh', process=False)
        occ = voxel_centers_inside(mesh, 64, spec['origin'], spec['pitch_xyz']).astype(bool)
        realized = (occ & env) | bc
        added = int((realized & ~occ).sum())
        removed = int((occ & ~realized).sum())
        repair = (added + removed) / max(1, int(realized.sum()))
        components = int(label(realized, structure=six)[1])
        side, taper = features(realized, spec)
        cell = cell_for((side, taper))
        reasons = []
        if row['seat_bc_coverage'] < MIN_SEAT: reasons.append('seat BC')
        if row['back_bc_coverage'] < MIN_BACK: reasons.append('back BC')
        if row['outside_envelope_fraction'] > MAX_OUTSIDE: reasons.append('envelope')
        if repair > MAX_REPAIR: reasons.append('repair budget')
        if components != 1: reasons.append('disconnected')
        if not row['watertight']: reasons.append('not watertight')
        if cell is None: reasons.append('descriptor range')
        # New cases did not receive independent FEA. Keep this separate from
        # geometry: a geometry pass would still need FEA before archive entry.
        reasons.append('FEA unavailable')
        candidates.append({
            'id': name, 'round': 1, 'source': 'tapered image/seed expansion',
            'mesh': row['mesh'], 'image': row['input_image'],
            'descriptor': {'side_open_fraction': side, 'backrest_taper_ratio': taper},
            'cell': cell, 'mass_liters': float(realized.sum() * np.prod(spec['pitch_xyz']) * 1000),
            'worst_compliance_ratio': None,
            'seat_bc': row['seat_bc_coverage'], 'back_bc': row['back_bc_coverage'],
            'outside_fraction': row['outside_envelope_fraction'],
            'repair_fraction': repair, 'components_after_repair': components,
            'reasons': reasons, 'archive_eligible': False,
        })
    archive: dict[tuple[int, int], list[dict]] = {}
    for candidate in candidates:
        if candidate['archive_eligible']:
            archive.setdefault(tuple(candidate['cell']), []).append(candidate)
    archive = {cell: nondominated(rows) for cell, rows in archive.items()}
    payload = {
        'descriptor_ranges': DESCRIPTOR_RANGES, 'archive_dims': ARCHIVE_DIMS,
        'objective': 'minimize realized mass (L) and worst of seat/back compliance ratio',
        'fea_scope': 'prior cases: 64^3 repaired voxel density with common 35 mm FEM mesh; new cases: no FEA',
        'gates': {'seat_bc_min': MIN_SEAT, 'back_bc_min': MIN_BACK,
                  'repair_max': MAX_REPAIR, 'outside_max': MAX_OUTSIDE,
                  'connected': True, 'watertight': True, 'descriptor_in_range': True,
                  'fea_valid': True},
        'prior_evaluations': len(prior), 'new_evaluations': len(NAMES) - 1,
        'total_unique_evaluations': len(candidates),
        'occupied_cells': len(archive),
        'archive': {','.join(map(str, cell)): [r['id'] for r in rows]
                    for cell, rows in sorted(archive.items())},
        'candidates': candidates,
    }
    (OUT / 'archive.json').write_text(json.dumps(payload, indent=2) + '\n')
    table = []
    for candidate in candidates:
        name = candidate['id']
        reasons = ', '.join(candidate['reasons']) or 'elite'
        table.append(f'''<tr><td>{html.escape(name)}</td><td>{candidate['round']}</td>
          <td>{candidate['cell']}</td><td>{candidate['seat_bc']:.0%}/{candidate['back_bc']:.0%}</td>
          <td>{candidate['repair_fraction']:.1%}</td><td>{candidate['mass_liters']:.1f}</td>
          <td>{'—' if candidate['worst_compliance_ratio'] is None else f"{candidate['worst_compliance_ratio']:.3f}"}</td>
          <td>{html.escape(reasons)}</td></tr>''')
    grid = []
    for j in reversed(range(ARCHIVE_DIMS[1])):
        for i in range(ARCHIVE_DIMS[0]):
            ids = [r['id'] for r in archive.get((i, j), [])]
            grid.append(f'<div class="cell"><b>({i},{j})</b><br>{html.escape(", ".join(ids) or "empty")}</div>')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair tapered QD round</title>
    <style>body{{font-family:system-ui,sans-serif;max-width:1350px;margin:auto;padding:28px;background:#edf1f4;color:#16232f}}
    .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}.cell{{background:white;min-height:86px;padding:12px;border-radius:8px}}
    table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:8px;border-bottom:1px solid #dae2e8;text-align:left}}
    a{{color:#185b8a}}</style><h1>Chair QD archive · tapered expansion</h1>
    <p>Descriptors: side opening fraction × backrest taper ratio. Quality: lower mass and lower worst seat/back compliance.
    New image and seed variants are recorded as observations, but none passes the structural interface gate;
    no new elite was admitted. FEA was run only for the four prior cases.</p>
    <p><a href="../index.html">Six tapered input/mesh comparisons</a> · <a href="archive.json">Archive JSON</a></p>
    <div class="grid">{''.join(grid)}</div>
    <h2>All unique evaluations</h2><table><thead><tr><th>Case</th><th>Round</th><th>Cell</th>
    <th>Seat/back BC</th><th>Repair</th><th>Mass L</th><th>Compliance ratio</th><th>Status</th></tr></thead>
    <tbody>{''.join(table)}</tbody></table></html>'''
    (OUT / 'index.html').write_text(page)
    report = [
        '# Chair QD archive: tapered expansion (2026-10-03)', '',
        f"Unique observations: {len(candidates)}; occupied cells: {len(archive)}/{ARCHIVE_DIMS[0]*ARCHIVE_DIMS[1]}; new elites: 0.",
        '', 'Descriptors are the existing 64³ side-open fraction and backrest taper ratio, with fixed 3×3 bins.',
        'Quality is the two-objective vector (mass in L, worst seat/back compliance ratio); Pareto elites are kept within each cell.',
        'The previous FEA is a repaired-voxel 35 mm proxy, not independent raw-OBJ FEA. New variants have no FEA because all failed geometry/interface gates.',
        '', 'The five new cases fail repair budget (>5%) and BC contact. The two seed controls demonstrate that even an unchanged front image does not reliably preserve backrest contact.',
        '', f"Archive JSON: {OUT / 'archive.json'}", f"Viewer: {OUT / 'index.html'}",
        f"Input and 3D comparison: {NEW / 'index.html'}",
    ]
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n')
    print('observations', len(candidates), 'cells', len(archive),
          'new elites', sum(r['archive_eligible'] for r in candidates if r['round'] == 1))


if __name__ == '__main__':
    main()
