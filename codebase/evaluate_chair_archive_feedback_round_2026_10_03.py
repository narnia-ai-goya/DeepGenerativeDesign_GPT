#!/usr/bin/env python3
"""Evaluate the frozen archive-feedback pair and update the realized 3D archive."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label
from scipy.spatial.transform import Rotation
import trimesh

from make_chair_domain import ROOT
from build_chair_single_view_qd_archive_2026_10_03 import features, cell_for
from build_chair_tapered_qd_round_2026_10_03 import nondominated
import evaluate_chair_text_latent_qd_3d_2026_10_03 as prior_fea

sys.path.insert(0, str(ROOT / 'codebase/code'))
sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'archive_feedback_round_02_2026-10-03'
SPEC = BASE / 'single_view_spec_2026-10-03'
OLD = BASE / 'tapered_examples_2026-10-03/qd_round_01'
OLD_ARCHIVE_PATH = OLD / 'archive.json'
PRUNE_TINY_BC_FREE = False
ROUND_INDEX = 2


def preview(mesh: trimesh.Trimesh, case, name: str) -> None:
    canvas = Image.new('RGB', (1080, 370), '#f5f7fa')
    draw = ImageDraw.Draw(canvas)
    for i, (elev, azim, target, extent) in enumerate((
        (15, 0, np.array([0., .01, .46]), .56),
        (15, 90, np.array([0., .01, .46]), .56),
        (10, 15, np.array([0., .17, .77]), .20),
    )):
        eye, up = camera_from_elev_azim(target, 2.0, elev, azim)
        rgb = render_lit(mesh, eye, target, up, size=350, fit_extent=extent,
                         color=(.62, .66, .70))
        canvas.paste(Image.fromarray(rgb).convert('RGB'), (i * 360, 18))
        draw.text((i * 360 + 8, 348), ('front', 'side', 'back detail')[i],
                  fill='#17212b')
    canvas.save(case / 'mesh_preview.png')


def main() -> None:
    selected = json.loads((OUT / 'selection.json').read_text())['selected']
    image_gate = {r['id']: r for r in json.loads((OUT / 'image_gate.json').read_text())}
    old = json.loads(OLD_ARCHIVE_PATH.read_text())
    spec = np.load(SPEC / 'voxel.npz')
    env, bc = spec['bracket'].astype(bool), spec['bc'].astype(bool)
    cal = json.loads((SPEC / 'specification.json').read_text())['native_frame_registration']
    rotation = Rotation.from_euler('x', cal['rotation_x_degrees'], degrees=True).as_matrix()
    six = np.zeros((3, 3, 3), bool)
    six[1, 1, 1] = True
    for axis in range(3):
        for sign in (-1, 1):
            ijk = [1, 1, 1]
            ijk[axis] += sign
            six[tuple(ijk)] = True
    rows = []
    for item in selected:
        name = item['id']
        case = OUT / name
        raw = trimesh.load(case / 'generation/mesh.obj', force='mesh', process=False)
        pieces = raw.split(only_watertight=False)
        mesh = max(pieces, key=lambda p: abs(p.volume))
        mesh.vertices = ((mesh.vertices - np.asarray(cal['source_center_m'])) @ rotation.T *
                         cal['uniform_scale'] + np.asarray(cal['physical_center_m']))
        mesh.export(case / 'aligned_main.obj')
        mesh.export(case / 'aligned_main.glb')
        preview(mesh, case, name)
        occ = voxel_centers_inside(mesh, 64, spec['origin'], spec['pitch_xyz']).astype(bool)
        realized = (occ & env) | bc
        before_labels, before_components = label(realized, structure=six)
        tiny_removed = 0
        tiny_fraction = 0.0
        if PRUNE_TINY_BC_FREE and before_components > 1:
            sizes = np.bincount(before_labels.ravel())
            main_label = int(np.argmax(sizes[1:]) + 1)
            other = (before_labels != 0) & (before_labels != main_label)
            tiny_fraction = float(other.sum() / max(1, int(realized.sum())))
            if not np.any(other & bc) and tiny_fraction <= .002 and np.all((before_labels == main_label)[bc]):
                tiny_removed = int(other.sum())
                realized = before_labels == main_label
        added = int((realized & ~occ).sum())
        removed = int((occ & ~realized).sum())
        repair = (added + removed) / max(1, int(realized.sum()))
        components = int(label(realized, structure=six)[1])
        side, taper = features(realized, spec)
        cell = cell_for((side, taper))
        seat = float((occ & spec['load']).sum() / max(1, spec['load'].sum()))
        back = float((occ & spec['back_load']).sum() / max(1, spec['back_load'].sum()))
        outside = float((occ & ~env).sum() / max(1, int(occ.sum())))
        reasons = []
        if not image_gate[name]['image_gate']: reasons.append('2D image gate')
        if seat < .50: reasons.append('seat BC')
        if back < .90: reasons.append('back BC')
        if outside > .01: reasons.append('envelope')
        if repair > .05: reasons.append('repair budget')
        if components != 1: reasons.append('disconnected after repair')
        if not mesh.is_watertight: reasons.append('not watertight')
        if cell is None: reasons.append('descriptor range')
        row = {
            'id': name, 'method': item['method'], 'round': ROUND_INDEX,
            'image': str(case / 'input.png'), 'mesh': str(case / 'aligned_main.obj'),
            'image_cell': image_gate[name]['image_cell'],
            'descriptor': {'side_open_fraction': side, 'backrest_taper_ratio': taper},
            'cell': cell, 'seat_bc': seat, 'back_bc': back,
            'outside_fraction': outside, 'repair_fraction': repair,
            'components_after_repair': components, 'source_watertight': bool(mesh.is_watertight),
            'raw_components': len(pieces),
            'tiny_fragment_voxels_removed': tiny_removed,
            'tiny_fragment_fraction_before': tiny_fraction,
            'mass_liters': float(realized.sum() * np.prod(spec['pitch_xyz']) * 1000),
            'geometry_reasons': reasons,
            'geometry_gate': not reasons,
        }
        np.savez_compressed(case / 'realized_occupancy.npz', occupied=realized)
        rows.append(row)
        print(name, 'cell', cell, 'BC', seat, back, 'repair', repair,
              'geometry gate', not reasons, flush=True)
    indices = np.argwhere(env)
    nodes = spec['origin'] + (indices + .5) * spec['pitch_xyz']
    prior_fea.OUT = OUT
    tasks = [(row, mode) for row in rows if row['geometry_gate'] for mode in ('seat', 'back')]
    for row in rows:
        if row['geometry_gate']:
            link = OUT / 'mesh_cases' / row['id']
            link.parent.mkdir(exist_ok=True)
            if not link.exists():
                link.symlink_to(OUT / row['id'], target_is_directory=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fea_results = list(pool.map(lambda pair: (pair[0]['id'], pair[1],
                                                  prior_fea.fea(pair[0], pair[1], nodes, indices)),
                                    tasks))
    by_fea = {(name, mode): value for name, mode, value in fea_results}
    baseline = json.loads((BASE / 'single_view_qd_2026-10-03/summary.json').read_text())
    seat_ref = baseline['baseline_compliance_proxy']['seat']
    back_ref = baseline['baseline_compliance_proxy']['back']
    prior_archive = {key: list(ids) for key, ids in old['archive'].items()}
    old_by_id = {row['id']: row for row in old['candidates']}
    new_elites = []
    for row in rows:
        name = row['id']
        if row['geometry_gate']:
            a, b = by_fea[name, 'seat'], by_fea[name, 'back']
            row['fea_035'] = {'seat': a, 'back': b}
            row['fea_valid'] = bool(a['valid'] and b['valid'])
            if row['fea_valid']:
                row['worst_compliance_ratio'] = max(a['compliance_proxy'] / seat_ref,
                                                    b['compliance_proxy'] / back_ref)
            else:
                row['geometry_reasons'].append('FEA unavailable')
        else:
            row['fea_valid'] = False
            row['geometry_reasons'].append('FEA not run: geometry gate failed')
        row['archive_eligible'] = bool(row['geometry_gate'] and row['fea_valid'])
        if row['archive_eligible']:
            key = ','.join(map(str, row['cell']))
            pool = [old_by_id[x] for x in prior_archive.get(key, [])] + [row]
            prior_archive[key] = [r['id'] for r in nondominated(pool)]
            if name in prior_archive[key]: new_elites.append(name)
        (OUT / name / 'evaluation.json').write_text(json.dumps(row, indent=2) + '\n')
    first = np.load(OUT / rows[0]['id'] / 'realized_occupancy.npz')['occupied']
    second = np.load(OUT / rows[1]['id'] / 'realized_occupancy.npz')['occupied']
    pair_iou = float((first & second).sum() / max(1, int((first | second).sum())))
    result = {
        'input_archive': str(OLD_ARCHIVE_PATH),
        'round': ROUND_INDEX,
        'tiny_fragment_rule': ('prospective: prune BC-free components only when total <=0.2% of realized occupancy'
                               if PRUNE_TINY_BC_FREE else 'disabled'),
        'generation_budget_each': 1,
        'occupied_before': old['occupied_cells'],
        'occupied_after': len(prior_archive),
        'new_elites': new_elites,
        'archive': prior_archive,
        'rows': rows,
        'pair_realized_voxel_iou': pair_iou,
        'fea_scope': '35 mm repaired 64^3 voxel proxy; not independent raw-OBJ FEA',
    }
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    state = {'archive': prior_archive,
             'occupied_cells': len(prior_archive),
             'candidates': old['candidates'] + rows,
             'archive_dims': old['archive_dims'],
             'descriptor_ranges': old['descriptor_ranges']}
    (OUT / 'archive_state.json').write_text(json.dumps(state, indent=2) + '\n')
    cards = []
    for row in rows:
        name = row['id']
        case = OUT / name
        reasons = ', '.join(row['geometry_reasons']) or 'passes all gates'
        quality = row.get('worst_compliance_ratio')
        cards.append(f'''<article><h2>{html.escape(row['method'])}: {html.escape(name)}</h2>
        <div class="visual"><img class="input" src="{name}/input.png"><img class="mesh" src="{name}/mesh_preview.png"></div>
        <p>Image cell {row['image_cell']} → 3D cell {row['cell']} · seat/back BC {row['seat_bc']:.0%}/{row['back_bc']:.0%} ·
        repair {row['repair_fraction']:.1%} · mass {row['mass_liters']:.1f} L ·
        worst compliance ratio {'—' if quality is None else f'{quality:.3f}'}</p>
        <p>Gate: {html.escape(reasons)} · new elite: {row['id'] in new_elites}</p>
        <p><a href="{name}/aligned_main.obj">OBJ</a> · <a href="{name}/aligned_main.glb">GLB</a> ·
        <a href="{name}/evaluation.json">evaluation</a></p></article>''')
    diagnostic_path = OUT / 'connectivity_diagnostic/result.json'
    diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else None
    diagnostic_note = (f'''<article><h2>Exploratory connectivity diagnostic</h2><p>The QD candidate's
        main component contains all BC voxels; {diagnostic['removed_voxels']} BC-free voxels
        ({diagnostic['removed_fraction']:.2%}) formed two tiny fragments. Removing only those fragments
        retains cell {diagnostic['cell_after']}; two-load 35 mm proxy worst compliance ratio:
        {diagnostic['worst_compliance_ratio_to_baseline']:.3f}. This post hoc rule is not counted in
        the primary archive.</p><a href="connectivity_diagnostic/result.json">Diagnostic JSON</a> ·
        <a href="connectivity_diagnostic/mesh_cases/pruned/pruned_voxel_proxy.obj">Pruned voxel OBJ</a></article>'''
        if diagnostic else '')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair archive-feedback QD round {ROUND_INDEX}</title>
    <style>body{{font:16px/1.5 system-ui;background:#eef2f4;color:#182932;max-width:1450px;margin:auto;padding:24px}}
    article{{background:white;border-radius:12px;padding:18px;margin:18px 0}}.visual{{display:flex;gap:12px;align-items:center}}
    img.input{{width:28%}}img.mesh{{width:70%}}a{{color:#1a5e91}}@media(max-width:700px){{.visual{{display:block}}
    img.input,img.mesh{{width:100%}}}}</style><h1>Chair 3D archive feedback · round {ROUND_INDEX}</h1>
    <p>Frozen bank and archive, one archive-aware acquisition versus one uniform-random prompt. Same reference,
    seed 42, Direct3D-S2 settings, BC and envelope. Both pass the 2D gate. Archive coverage:
    {old['occupied_cells']}/9 → {len(prior_archive)}/9. New elites: {len(new_elites)}.
    Pair realized-voxel IoU: {result['pair_realized_voxel_iou']:.3f}.
    FEM is run only after geometry/interface gates, on the common 35 mm repaired-voxel proxy.</p>
    <p><a href="selection.json">Selection and acquisition scores</a> ·
    <a href="image_generation.json">Exact image prompts</a> ·
    <a href="image_gate.json">2D gates</a> · <a href="result.json">3D result</a> ·
    <a href="archive_state.json">Archive after round</a> ·
    <a href="../tapered_examples_2026-10-03/qd_round_01/index.html">Previous archive</a></p>
    {''.join(cards)}{diagnostic_note}</html>'''
    (OUT / 'index.html').write_text(page)
    report = [f'# Chair archive-feedback round {ROUND_INDEX}', '',
              'Frozen 24-prompt bank; archive-aware selection vs one uniform-random control.',
              'Same reference image, seed 42 and Direct3D-S2 settings; no training.',
              f"Coverage: {old['occupied_cells']}/9 -> {len(prior_archive)}/9; new elites: {len(new_elites)}.",
              f"Pair realized-voxel IoU: {result['pair_realized_voxel_iou']:.3f}.",
              '', '| Method | Prompt | 3D cell | Seat/back BC | Repair | FEA | Elite |',
              '|---|---|---|---:|---:|---|---|']
    for row in rows:
        report.append(f"| {row['method']} | {row['id']} | {row['cell']} | "
                      f"{row['seat_bc']:.0%}/{row['back_bc']:.0%} | "
                      f"{row['repair_fraction']:.1%} | {row['fea_valid']} | "
                      f"{row['id'] in new_elites} |")
    report.extend(['', 'The acquisition probabilities are heuristic and uncalibrated; this two-candidate round does not establish method superiority.',
                   f"Tiny-component rule: {result['tiny_fragment_rule']}.",
                   'FEA values, if any, are from the common 35 mm repaired-voxel proxy, not independent raw-OBJ FEA.',
                   '', str(OUT / 'index.html'), str(OUT / 'result.json'),
                   str(OUT / 'image_generation.json')])
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n')
    print('coverage', old['occupied_cells'], '->', len(prior_archive),
          'new elites', new_elites, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--round-dir', type=Path, default=OUT)
    parser.add_argument('--archive-in', type=Path, default=OLD_ARCHIVE_PATH)
    parser.add_argument('--round-index', type=int, default=2)
    parser.add_argument('--prune-tiny-bc-free', action='store_true')
    args = parser.parse_args()
    OUT = args.round_dir.resolve()
    OLD_ARCHIVE_PATH = args.archive_in.resolve()
    ROUND_INDEX = args.round_index
    PRUNE_TINY_BC_FREE = args.prune_tiny_bc_free
    main()
