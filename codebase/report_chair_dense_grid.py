#!/usr/bin/env python3
"""Score the neutral-chair dense grid on BC connectivity and image fidelity."""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import label

from run_chair_dense_grid import BASE, GRID, ROOT

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit


VIEWS = [('v00_front_lo', 15, 0), ('v02_right_lo', 15, 90), ('v_top', 85, 0)]


def case_path(pw: float, cfg: float) -> Path:
    return BASE / f'support_pw{pw:g}' if cfg == 7 else GRID / f'pw{pw:g}_cfg{cfg:g}'


def frontmost_label(labels: np.ndarray, mask: np.ndarray) -> int:
    values = labels[mask]
    counts = np.bincount(values[values > 0])
    return int(counts.argmax()) if counts.size else 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1))


def main() -> None:
    GRID.mkdir(parents=True, exist_ok=True)
    env = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    grid = np.load(ROOT / 'data_real/chair/voxel.npz')
    center = env.bounds.mean(axis=0)
    radius = np.linalg.norm(env.extents) * 1.5
    scale = env.extents.max() / 2
    fix_labels, _ = label(grid['fix'])
    records = []
    for pw in (2.0, 3.5, 5.0):
        for cfg in (5.0, 7.0, 9.0):
            case = case_path(pw, cfg)
            mesh_path = case / 'dense/mesh_dense.obj'
            cache_path = case / 'dense_cache.npz'
            if not mesh_path.exists() or not cache_path.exists():
                records.append({'pw': pw, 'cfg': cfg, 'status': 'pending'})
                continue
            mesh = trimesh.load(mesh_path, force='mesh')
            ids = np.load(cache_path)['latent_index']
            occ = np.zeros((64, 64, 64), bool)
            occ[ids[:, 1], ids[:, 2], ids[:, 3]] = True
            labels, ncomp = label(occ)
            load_label = frontmost_label(labels, grid['load'])
            foot_labels = [frontmost_label(labels, fix_labels == i) for i in range(1, 5)]
            connected = bool(load_label > 0 and all(x == load_label for x in foot_labels))
            views = []
            previews = []
            for name, elev, azim in VIEWS:
                eye, up = camera_from_elev_azim(center, radius, elev, azim)
                rendered = Image.fromarray(render_lit(
                    mesh, eye, center, up, size=512, fit_extent=scale,
                    margin=1.15, color=(.25, .28, .31))).convert('RGB')
                render_path = case / f'dense_{name}.png'
                rendered.save(render_path)
                reference = Image.open(BASE / 'input' / f'{name}.png').convert('RGB')
                mask_a = np.asarray(reference).min(axis=2) < 210
                mask_b = np.asarray(rendered).min(axis=2) < 210
                views.append({'name': name, 'iou': round(iou(mask_a, mask_b), 4),
                              'render': str(render_path)})
                previews.append(rendered)
            sheet = Image.new('RGB', (512 * 3, 548), 'white')
            draw = ImageDraw.Draw(sheet)
            for i, (v, im) in enumerate(zip(views, previews)):
                sheet.paste(im, (i * 512, 36))
                draw.text((i * 512 + 10, 10), f'{v["name"]} · IoU={v["iou"]:.3f}', fill='#20252b')
            sheet_path = case / 'dense_contact.png'
            sheet.save(sheet_path)
            row = {'pw': pw, 'cfg': cfg, 'status': 'complete',
                   'mesh': str(mesh_path), 'contact': str(sheet_path),
                   'n_active': int(len(ids)), 'mask_components': int(ncomp),
                   'bc_all_connected': connected,
                   'left_right_symmetry_iou': round(iou(occ, np.flip(occ, axis=0)), 4),
                   'back_height_m': round(float(mesh.bounds[1, 2]), 4),
                   'volume_litres': round(abs(float(mesh.volume))*1000, 2),
                   'front_side_iou': round(float(np.mean([v['iou'] for v in views[:2]])), 4),
                   'view_metrics': views}
            records.append(row)
    (GRID / 'metrics.json').write_text(json.dumps(records, indent=2) + '\n')
    cards = []
    by_key = {(r['pw'], r['cfg']): r for r in records}
    table = '<table><tr><th>pw \\ cfg</th><th>5</th><th>7</th><th>9</th></tr>'
    for pw in (2.0, 3.5, 5.0):
        table += f'<tr><th>{pw:g}</th>'
        for cfg in (5.0, 7.0, 9.0):
            r = by_key[(pw, cfg)]
            if r['status'] != 'complete':
                table += '<td>running</td>'
                continue
            good = r['bc_all_connected'] and r['mask_components'] == 1 and r['back_height_m'] >= .82
            klass = 'good' if good else 'limited'
            table += (f'<td class="{klass}">BC {"✓" if r["bc_all_connected"] else "✗"}'
                      f' · mask {r["mask_components"]} comps<br>height {r["back_height_m"]:.3f} m'
                      f'<br>front/side IoU {r["front_side_iou"]:.3f}'
                      f'<br>symmetry IoU {r["left_right_symmetry_iou"]:.3f}</td>')
        table += '</tr>'
    table += '</table>'
    for r in records:
        title = f'pw={r["pw"]:g}, cfg={r["cfg"]:g}'
        if r['status'] != 'complete':
            cards.append(f'<article><h2>{title}</h2><p>running</p></article>')
            continue
        rel = Path(r['contact']).relative_to(ROOT)
        cards.append(f'<article><h2>{title}</h2><p>BC connected: <b>{r["bc_all_connected"]}</b>'
                     f' · height: {r["back_height_m"]:.3f} m · front/side IoU: {r["front_side_iou"]:.3f}'
                     f' · symmetry: {r["left_right_symmetry_iou"]:.3f}'
                     f' · volume: {r["volume_litres"]:.1f} L</p>'
                     f'<a href="/{rel}"><img src="/{rel}"></a><small>{html.escape(r["mesh"])}</small></article>')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair dense grid</title>
<style>body{font:15px system-ui,sans-serif;background:#f4f6f8;color:#1b252f;max-width:1600px;margin:2rem auto;padding:0 1rem}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}article{background:white;border-radius:10px;padding:1rem}
img{width:100%}small{word-break:break-all}h2{margin:.2rem 0}table{border-collapse:collapse;background:white;width:100%}
td,th{border:1px solid #cbd2d8;padding:.8rem;text-align:left}td.good{background:#ddf3e6}td.limited{background:#fff2e2}</style>
<h1>Neutral chair: support-path weight × image CFG</h1>
<p>same three views, seed and BC; dense only. BC connected means all four feet and seat patch share one active-voxel component. IoU is the average front/side silhouette overlap with the rendered neutral reference. The pilot has no in-loop FEA.</p>
''' + table + '''<div class="grid">''' + ''.join(cards) + '</div></html>'
    (GRID / 'index.html').write_text(page)
    print(GRID / 'metrics.json')


if __name__ == '__main__':
    main()
