#!/usr/bin/env python3
"""Publish the chair pilot's grid, 3D stages, BC checks and independent FEA."""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from run_chair_dense_grid import BASE, GRID, ROOT
from report_chair_dense_grid import VIEWS

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit


def render_case(mesh_path: Path, output: Path, title: str) -> None:
    envelope = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    mesh = trimesh.load(mesh_path, force='mesh')
    center = envelope.bounds.mean(axis=0)
    radius = np.linalg.norm(envelope.extents) * 1.5
    scale = envelope.extents.max() / 2
    sheet = Image.new('RGB', (512 * len(VIEWS), 548), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (name, elev, azim) in enumerate(VIEWS):
        eye, up = camera_from_elev_azim(center, radius, elev, azim)
        image = Image.fromarray(render_lit(mesh, eye, center, up,
                              size=512, fit_extent=scale, margin=1.15,
                              color=(.25, .28, .31))).convert('RGB')
        sheet.paste(image, (i * 512, 36))
        draw.text((i * 512 + 10, 10), f'{title} · {name}', fill='#20252b')
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    grid_metrics = json.loads((GRID / 'metrics.json').read_text())
    reference = json.loads((BASE / 'reference_fea/fea_tet_summary.json').read_text())
    cases = []
    for weight in (3.5, 5.0):
        case = BASE / f'support_pw{weight:g}'
        metrics = case / 'metrics.json'
        if not metrics.exists():
            continue
        data = json.loads(metrics.read_text())
        final = Path(data['final_mesh'])
        contact = case / 'final_contact.png'
        render_case(final, contact, f'pw={weight:g} final')
        cases.append({'weight': weight, 'case': str(case), 'contact': str(contact),
                      'metrics': data})
    four_view_case = BASE / 'view4_pw3.5_cfg7'
    four_view = json.loads((four_view_case / 'dense_metrics.json').read_text())
    manifest = {'reference': {'mesh': str(BASE / 'neutral_reference.obj'),
                              'fea': reference},
                'grid_metrics': str(GRID / 'metrics.json'),
                'cases': cases,
                'four_view_dense': {'case': str(four_view_case), 'metrics': four_view,
                                    'contact': str(four_view_case / 'dense_contact.png')}}
    (BASE / 'report.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows = ''.join(
        f'<tr><td>pw={r["weight"]:g}</td><td>{r["metrics"]["volume_litres"]:.1f}</td>'
        f'<td>{r["metrics"].get("fea",{}).get("compliance",float("nan")):.6f}</td>'
        f'<td>{r["metrics"]["bc_containment"]["fix"]:.4f}</td>'
        f'<td>{r["metrics"]["bc_containment"]["load"]:.4f}</td>'
        f'<td>{r["metrics"]["geometry_valid"]}</td></tr>' for r in cases)
    cards = ''.join(
        f'<article><h2>pw={r["weight"]:g} final</h2>'
        f'<img src="/{Path(r["contact"]).relative_to(ROOT)}">'
        f'<p><a href="/{Path(r["metrics"]["final_mesh"]).relative_to(ROOT)}">final OBJ</a>'
        f' · <a href="/{Path(r["metrics"]["fea_summary"]).relative_to(ROOT)}">FEA JSON</a></p></article>'
        for r in cases)
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair pilot</title>
<style>body{{font:15px system-ui,sans-serif;background:#f4f6f8;color:#1b252f;max-width:1600px;margin:2rem auto;padding:0 1rem}}
article,section{{background:white;border-radius:10px;padding:1rem;margin:1rem 0}}img{{width:100%}}
table{{border-collapse:collapse}}td,th{{border:1px solid #cbd2d8;padding:.5rem}}</style>
<h1>Chair concept pilot · 2026-09-25</h1>
<section><p>Neutral chair, identical three rendered views and seed 42. Dense-stage grid search varied support-corridor weight and image CFG. The corridor is a functional design prior, not a fixed FEA boundary. Both dense and sparse runs here have in-loop FEA OFF; final meshes were independently verified under 800 N seat load and four fixed floor patches.</p>
<p><a href="dense_grid_pw_cfg/index.html">3×3 dense grid</a> · <a href="/data_real/chair/domain_preview.png">BC/envelope preview</a> · <a href="report.json">manifest</a></p>
<p>The best admissible dense grid case is pw=3.5, CFG=7: one active component, full-height back, front/side silhouette IoU 0.505. The added back-view trial raises front/side IoU to {four_view['front_side_iou']:.3f}, but splits into {four_view['mask_components']} active components and expands to {four_view['volume_litres']:.1f} L, so it was not continued to sparse. All cases remain visually overfilled below the seat; the generated geometry is a baseline, not a satisfactory chair design.</p>
<table><tr><th>geometry</th><th>volume L</th><th>compliance J</th><th>fix coverage</th><th>load coverage</th><th>valid</th></tr>
<tr><td>neutral reference</td><td>40.8</td><td>{reference['compliance']:.6f}</td><td>—</td><td>—</td><td>reference</td></tr>{rows}</table>
<p>Compliance values use equal force/material/BC, but volumes differ substantially; this is not a mass-matched performance claim.</p></section>
<article><h2>Input reference</h2><img src="input_contact.png"></article>{cards}
<article><h2>Four-view dense check: pw=3.5, CFG=7</h2><img src="view4_pw3.5_cfg7/dense_contact.png"><p><a href="view4_pw3.5_cfg7/dense/mesh_dense.obj">dense OBJ</a> · <a href="view4_pw3.5_cfg7/dense_metrics.json">metrics</a></p></article></html>'''
    (BASE / 'index.html').write_text(page)
    print(BASE / 'index.html')


if __name__ == '__main__':
    main()
