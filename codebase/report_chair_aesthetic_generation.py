#!/usr/bin/env python3
"""Compare the aesthetic chair reference to the generated dense candidates."""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import label

from make_chair_domain import ROOT
from report_chair_dense_grid import VIEWS, frontmost_label, iou

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit


BASE = ROOT / 'experiments/chair/aesthetic_reference_2026-09-27'
CASES = ('generation', 'archpath_cfg7', 'archpath_cfg9', 'archpath_4view',
         'projection_w5', 'projection_w20',
         'projection_w10_pw5', 'projection_w10_pw5_sym',
         'projection_w10_pw5_sym_reach')
GRID = BASE / 'eta_pw_grid'


def main() -> None:
    grid = np.load(ROOT / 'data_real/chair/voxel.npz')
    env = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents))*1.5
    scale = float(env.extents.max())/2
    foot_labels, _ = label(grid['fix'])
    origin, pitch = grid['origin'], grid['pitch_xyz']
    axes = [origin[i]+(np.arange(64)+.5)*pitch[i] for i in range(3)]
    x, y, z = np.meshgrid(*axes, indexing='ij')
    core = (np.abs(x)<.115)&(np.abs(y)<.115)&(z>=.08)&(z<=.42)
    rows = []
    cards = []
    cases = [(name, BASE / name) for name in CASES]
    if (GRID / 'manifest.json').exists():
        for record in json.loads((GRID / 'manifest.json').read_text()):
            cases.append((f'eta{record["eta"]:g}_pw{record["pw"]:g}',
                          Path(record['case'])))
    for name, case in cases:
        mesh_path = case / 'dense/mesh_dense.obj'
        cache_path = case / 'dense_cache.npz'
        if not mesh_path.exists() or not cache_path.exists():
            continue
        mesh = trimesh.load(mesh_path, force='mesh')
        ids = np.load(cache_path)['latent_index']
        occ = np.zeros((64, 64, 64), bool)
        occ[ids[:, 1], ids[:, 2], ids[:, 3]] = True
        labels, components = label(occ)
        load_id = frontmost_label(labels, grid['load'])
        foot_ids = [frontmost_label(labels, foot_labels == k) for k in range(1, 5)]
        connected = bool(load_id and all(k == load_id for k in foot_ids))
        sheet = Image.new('RGB', (1536, 548), 'white')
        draw = ImageDraw.Draw(sheet)
        view_scores = []
        for i, (view, elev, azim) in enumerate(VIEWS):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rendered = Image.fromarray(render_lit(mesh, eye, center, up,
                size=512, fit_extent=scale, margin=1.15,
                color=(.19, .22, .25))).convert('RGB')
            reference = Image.open(BASE / 'input' / f'{view}.png').convert('RGB')
            rendered_mask = np.asarray(rendered).min(axis=2)<210
            reference_mask = np.asarray(reference).min(axis=2)<210
            overlap = iou(rendered_mask, reference_mask)
            false_positive = float((rendered_mask & ~reference_mask).sum()
                                   / max(rendered_mask.sum(), 1))
            view_scores.append({'view': view, 'iou': round(overlap, 4),
                                'false_positive_fraction': round(false_positive, 4)})
            sheet.paste(rendered, (i*512, 36))
            draw.text((i*512+12, 10), f'{name} · {view} · IoU={overlap:.3f}',
                      fill='#263139')
        contact = case / 'dense_contact.png'
        sheet.save(contact)
        row = {'case': name, 'mesh': str(mesh_path), 'contact': str(contact),
               'mask_components': int(components), 'bc_connected': connected,
               'back_height_m': round(float(mesh.bounds[1, 2]), 4),
               'central_gap_occupied': round(float(occ[core].mean()), 4),
               'front_side_iou': round(float(np.mean([v['iou'] for v in view_scores[:2]])), 4),
               'front_side_false_positive': round(float(np.mean(
                   [v['false_positive_fraction'] for v in view_scores[:2]])), 4),
               'view_scores': view_scores}
        rows.append(row)
        relative_contact = contact.relative_to(ROOT)
        relative_obj = mesh_path.relative_to(ROOT)
        cards.append(f'<article><h2>{html.escape(name)}</h2><a href="/{relative_contact}">'
                     f'<img src="/{relative_contact}"></a><p><a href="/{relative_obj}">dense OBJ</a></p></article>')
    (BASE / 'generation_metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    fea_path = BASE / 'fea_reference/fea_tet_summary.json'
    fea = json.loads(fea_path.read_text()) if fea_path.exists() else None
    fea_note = (f'<p>Independent 800 N seat-load FEA on the source: '
                f'compliance {fea["compliance"]:.6f} J, maximum displacement '
                f'{fea["u_max"]*1000:.4f} mm. '
                f'<a href="fea_reference/fea_tet_summary.json">FEA JSON</a>. '
                f'This checks the pilot load case only; a separate backrest load is not included.</p>') \
                if fea else ''
    sparse_path = BASE / 'sparse_checks.json'
    sparse_note = ''
    if sparse_path.exists():
        sparse = json.loads(sparse_path.read_text())['cases']
        sparse_rows = ''.join(
            f'<tr><td>{html.escape(name)}</td><td>{sparse[name]["components"]}</td>'
            f'<td>{sparse[name]["component_volumes_litres"][0]:.2f}</td>'
            f'<td>{sparse[name]["bc_containment"]["fix"]:.3f}</td></tr>'
            for name in ('archpath_cfg9', 'sparse_buffer5', 'sparse_buffer20',
                         'post', 'post_dilate3', 'post_dilate6'))
        sparse_note = ('<article><h2>Sparse and post-process audit</h2>'
                       '<p>The dense arch-path candidate is connected, but sparse refining splits one '
                       'foot into a separate ~2 L component. BC-buffer guidance at weights 5 and 20 '
                       'does not reconnect it. The post-process keeps the largest component and loses '
                       'about one-quarter of the fixed-foot volume. Dilating the BC solids by 3 or 6 mm '
                       'at post-processing does not restore the connection. The generated mesh below is '
                       'diagnostic and fails geometry validity; no FEA result is claimed for it.</p>'
                       '<table><tr><th>stage</th><th>components</th><th>largest volume L</th>'
                       '<th>fixed BC coverage</th></tr>' + sparse_rows + '</table>'
                       '<p><a href="sparse_checks.json">Full audit JSON</a> · '
                       '<a href="archpath_cfg9/post/final.obj">invalid final OBJ</a></p>'
                       '<img src="archpath_cfg9/post/final_preview.png" alt="Diagnostic generated chair mesh">'
                       '</article>')
    table = ''.join(f'<tr><td>{html.escape(r["case"])}</td><td>{r["mask_components"]}</td>'
                    f'<td>{r["bc_connected"]}</td><td>{r["back_height_m"]:.3f}</td>'
                    f'<td>{r["central_gap_occupied"]:.3f}</td>'
                    f'<td>{r["front_side_iou"]:.3f}</td>'
                    f'<td>{r["front_side_false_positive"]:.3f}</td></tr>' for r in rows)
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Curved-back chair concept</title>
<style>body{font:15px system-ui,sans-serif;background:#f4f6f8;color:#1b252f;max-width:1600px;margin:2rem auto;padding:0 1rem}
img{width:100%}article{background:white;border-radius:10px;padding:1rem;margin:1rem 0}table{border-collapse:collapse;background:white;width:100%}
td,th{border:1px solid #cbd2d8;padding:.5rem;text-align:left}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}</style>
<h1>Curved-back chair concept</h1><p>The same four fixed feet, seat load patch and envelope are preserved. The new source is a rounded-seat, tapered-leg, open-arch chair. Dense trials use in-loop FEA OFF. The generated meshes are diagnostic; the new reference itself is a validated BC-aligned input. Some dense trials still have unwanted bulges on the legs and side, so this page does not present them as final designs. The projection-guided trials add registered front/right 2D silhouettes to the dense loss; <a href="projection_registration.json">registration checks</a> confirm 0.957/0.971 overlap with the source mesh projections.</p>
<article><h2>New source geometry</h2><img src="hero.png"><p><a href="reference.obj">source OBJ</a> · <a href="validation.json">BC/envelope checks</a> · <a href="contact.png">six aligned views</a></p>''' + fea_note + '''</article>
''' + sparse_note + '''<table><tr><th>dense condition</th><th>components</th><th>BC path</th><th>back height m</th><th>center occupied</th><th>front/right IoU</th><th>extra silhouette fraction</th></tr>''' + table + '''</table><div class="grid">''' + ''.join(cards) + '</div></html>'
    (BASE / 'index.html').write_text(page)
    print(BASE / 'index.html')


if __name__ == '__main__':
    main()
