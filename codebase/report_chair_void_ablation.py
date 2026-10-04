#!/usr/bin/env python3
"""Render and score chair empty-space ablations against the same reference."""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import label

from run_chair_void_ablation import BASE, OUT, ROOT
from report_chair_dense_grid import VIEWS, frontmost_label, iou

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit


def score(case: Path, name: str, grid: np.lib.npyio.NpzFile,
          env: trimesh.Trimesh) -> dict:
    mesh_path = case / 'dense/mesh_dense.obj'
    ids_path = case / 'dense_cache.npz'
    if not mesh_path.exists() or not ids_path.exists():
        return {'name': name, 'status': 'missing', 'case': str(case)}
    mesh = trimesh.load(mesh_path, force='mesh')
    ids = np.load(ids_path)['latent_index']
    occ = np.zeros((64, 64, 64), bool)
    occ[ids[:, 1], ids[:, 2], ids[:, 3]] = True
    leg_path = np.load(BASE / 'support_pw3.5/load_corridors.npz')['mask']
    full_path = np.load(OUT / 'rails_out100_backpath/load_corridors.npz')['mask']
    back_path = full_path & ~leg_path
    labels, ncomp = label(occ)
    fix_labels, _ = label(grid['fix'])
    seat = frontmost_label(labels, grid['load'])
    foot_labels = [frontmost_label(labels, fix_labels == k) for k in range(1, 5)]
    bc_connected = bool(seat > 0 and all(k == seat for k in foot_labels))
    p = grid['pitch_xyz']; o = grid['origin']
    ax = [o[i] + (np.arange(64) + .5)*p[i] for i in range(3)]
    x, y, z = np.meshgrid(*ax, indexing='ij')
    center_void = ((np.abs(x) < .115) & (np.abs(y) < .115)
                   & (z >= .08) & (z <= .42))
    underseat = ((np.abs(x) < .265) & (np.abs(y) < .255)
                 & (z >= .08) & (z <= .42))
    center = env.bounds.mean(axis=0)
    radius = np.linalg.norm(env.extents) * 1.5
    scale = env.extents.max() / 2
    render_scores, previews = [], []
    for view, elev, azim in VIEWS:
        eye, up = camera_from_elev_azim(center, radius, elev, azim)
        rendered = Image.fromarray(render_lit(mesh, eye, center, up,
             size=512, fit_extent=scale, margin=1.15,
             color=(.25, .28, .31))).convert('RGB')
        ref = Image.open(BASE / 'input' / f'{view}.png').convert('RGB')
        score_value = iou(np.asarray(rendered).min(2)<210,
                          np.asarray(ref).min(2)<210)
        render_scores.append({'view': view, 'iou': round(score_value, 4)})
        previews.append(rendered)
    sheet = Image.new('RGB', (1536, 548), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (row, im) in enumerate(zip(render_scores, previews)):
        sheet.paste(im, (i*512, 36))
        draw.text((i*512+12, 11),
                  f'{name} · {row["view"]} · IoU={row["iou"]:.3f}', fill='#263139')
    contact = case / 'dense_contact.png'
    sheet.save(contact)
    return {'name': name, 'status': 'complete', 'case': str(case),
            'mesh': str(mesh_path), 'contact': str(contact),
            'mask_components': int(ncomp), 'bc_all_connected': bc_connected,
            'back_height_m': round(float(mesh.bounds[1, 2]), 4),
            'dense_volume_litres': round(abs(float(mesh.volume))*1000, 2),
            'center_void_occupancy': round(float(occ[center_void].mean()), 4),
            'underseat_occupancy': round(float(occ[underseat].mean()), 4),
            'leg_path_coverage': round(float(occ[leg_path].mean()), 4),
            'back_path_coverage': round(float(occ[back_path].mean()), 4),
            'front_side_iou': round(float(np.mean([v['iou'] for v in render_scores[:2]])), 4),
            'view_scores': render_scores}


def main() -> None:
    manifest = json.loads((OUT / 'manifest.json').read_text())
    backpath_manifest = json.loads((OUT / 'backpath_manifest.json').read_text()) \
        if (OUT / 'backpath_manifest.json').exists() else {}
    grid = np.load(ROOT / 'data_real/chair/voxel.npz')
    env = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    cases = [('baseline_pw3.5', BASE / 'support_pw3.5'),
             ('no_corridor', BASE),
             *[(name, Path(row['case'])) for name, row in manifest.items()],
             *[(name, Path(row['case'])) for name, row in backpath_manifest.items()],
             ('backpath_eta0', OUT / 'rails_out100_backpath_eta0'),
             ('backpath_vanilla', OUT / 'rails_out100_backpath_vanilla')]
    records = [score(path, name, grid, env) for name, path in cases]
    (OUT / 'metrics.json').write_text(json.dumps(records, indent=2) + '\n')
    final_case = OUT / 'rails_out100_backpath'
    final_shape = json.loads((final_case / 'final_shape_metrics.json').read_text()) \
        if (final_case / 'final_shape_metrics.json').exists() else None
    baseline_shape = json.loads((OUT / 'baseline_final_shape_metrics.json').read_text()) \
        if (OUT / 'baseline_final_shape_metrics.json').exists() else None
    final_metrics = json.loads((final_case / 'metrics.json').read_text()) \
        if (final_case / 'metrics.json').exists() else None
    rows, cards = [], []
    for r in records:
        name = html.escape(r['name'])
        if r['status'] != 'complete':
            rows.append(f'<tr><td>{name}</td><td colspan="9">missing</td></tr>')
            continue
        valid = r['mask_components'] == 1 and r['bc_all_connected'] and r['back_height_m'] > .85
        rows.append(f'<tr class="{"good" if valid else "limited"}"><td>{name}</td>'
                    f'<td>{r["mask_components"]}</td><td>{r["bc_all_connected"]}</td>'
                    f'<td>{r["back_height_m"]:.3f}</td>'
                    f'<td>{r["center_void_occupancy"]:.3f}</td>'
                    f'<td>{r["underseat_occupancy"]:.3f}</td>'
                    f'<td>{r["leg_path_coverage"]:.3f}</td>'
                    f'<td>{r["back_path_coverage"]:.3f}</td>'
                    f'<td>{r["front_side_iou"]:.3f}</td>'
                    f'<td>{r["dense_volume_litres"]:.1f}</td></tr>')
        image = Path(r['contact']).relative_to(ROOT)
        obj = Path(r['mesh']).relative_to(ROOT)
        cards.append(f'<article><h2>{name}</h2><a href="/{image}"><img src="/{image}"></a>'
                     f'<p><a href="/{obj}">dense OBJ</a></p><small>{html.escape(r["mesh"])}</small></article>')
    final_section = ''
    if final_shape and baseline_shape and final_metrics:
        baseline_image = (BASE / 'support_pw3.5/final_contact.png').relative_to(ROOT)
        final_image = (final_case / 'final_contact.png').relative_to(ROOT)
        final_obj = Path(final_metrics['final_mesh']).relative_to(ROOT)
        final_fea = Path(final_metrics['fea_summary']).relative_to(ROOT)
        final_section = f'''<section><h2>Selected final mesh after sparse, boolean, remesh and independent FEA</h2>
<p>Baseline → selected: front/right silhouette IoU {baseline_shape['front_side_iou']:.3f} → {final_shape['front_side_iou']:.3f};
central gap occupied {baseline_shape['center_void_occupancy']:.1%} → {final_shape['center_void_occupancy']:.1%};
final volume 67.86 → {final_metrics['volume_litres']:.2f} L. The new final mesh is watertight,
one component, and retains fix/load coverage {final_metrics['bc_containment']['fix']:.4f}/{final_metrics['bc_containment']['load']:.4f}.
Independent 800 N FEA compliance is {final_metrics['fea']['compliance']:.6f} J.
The compliance increase from the heavy baseline is expected because much of the under-seat wall was removed; this is not an equal-mass comparison.</p>
<div class="cards"><article><h3>Earlier final</h3><img src="/{baseline_image}"></article>
<article><h3>Selected final</h3><img src="/{final_image}"><p><a href="/{final_obj}">final OBJ</a> ·
<a href="/{final_fea}">independent FEA JSON</a></p></article></div></section>'''
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair void ablation</title>
<style>body{font:15px system-ui,sans-serif;background:#f4f6f8;color:#1b252f;max-width:1600px;margin:2rem auto;padding:0 1rem}
table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #cbd2d8;padding:.5rem;text-align:left}
tr.good{background:#e0f3e7}tr.limited{background:#fff3e4}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}
article,section{background:white;border-radius:10px;padding:1rem;margin:1rem 0}img{width:100%}small{word-break:break-all}</style>
<h1>Chair empty-space ablations · 2026-09-27</h1>
<p>Same seed, neutral reference and BCs. The ablation grid is dense-only; its selected candidate also ran through sparse, post-processing and independent FEA. In-loop FEA was OFF. The center-void and four-rail cases change the allowed 64³ mask below the seat while preserving all fixed and load voxels; the physical FEA domain remains the original broad envelope. Center-void occupancy is the active-voxel fraction in the intended open central space; lower is better. Image IoU compares front/right silhouettes to the rendered reference. The added backpath is an explicit design scaffold, so this test establishes the value of designer-provided geometric intent rather than unaided shape discovery.</p>
''' + final_section + '''<section><h2>Does the dense loss work?</h2>
<p>Yes, but the hard mask also matters. With the same mask and backrest path, setting η=0 keeps the mask and BC filter but removes every gradient-guidance update: back-path occupancy is 10.1% versus 100% with η=300, the mask has four rather than one component, and front/right IoU is 0.450 versus 0.705. With the same four-rail mask and no backrest path, increasing out_w from 20 to 100 reduces pre-filter tokens from 30,837 to 13,934; tokens discarded by the hard mask fall from 14,533 to 452. VANILLA is a weaker control because that code path bypasses the 64³ bracket filter as well. Sparse out-region BCE, sparse FEA and dense FEA were all OFF in these runs; the explicit geometric guidance acts during dense.</p></section>
<table><tr><th>case</th><th>components</th><th>BC path</th><th>back height m</th><th>center occupied</th><th>underseat occupied</th><th>leg path covered</th><th>back path covered</th><th>front/right IoU</th><th>dense volume L</th></tr>''' + ''.join(rows) + '''</table><div class="cards">''' + ''.join(cards) + '</div></html>'
    (OUT / 'index.html').write_text(page)
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
