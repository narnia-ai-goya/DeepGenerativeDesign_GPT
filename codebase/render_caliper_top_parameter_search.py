#!/usr/bin/env python3
"""Render and rank the top-view-only caliper parameter search."""
import html
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

ROOT = Path('/home/goya/SDL/3d_qd')
OUT = ROOT / 'experiments/caliper_top_parameter_search_2026-09-17'
CASES = OUT / 'caliper'
ASSETS = OUT / 'report_assets'
ASSETS.mkdir(parents=True, exist_ok=True)
CAM = [(0.22, -0.31, 0.20), (0, 0, 0), (0, 0, 1)]


def poly(m):
    return pv.PolyData(m.vertices, np.hstack([np.full((len(m.faces), 1), 3), m.faces]).ravel())


def measure(case):
    raw = trimesh.load(case / 'gen/mesh.obj', force='mesh', process=False)
    final = trimesh.load(case / 'gen/final.obj', force='mesh', process=False)
    return {'name': case.name, 'raw_components': len(raw.split(only_watertight=False)),
            'raw_volume_mm3': round(abs(raw.volume) * 1e9),
            'final_volume_mm3': round(abs(final.volume) * 1e9),
            'watertight': bool(final.is_watertight)}


def render(case, row):
    m = trimesh.load(case / 'gen/final.obj', force='mesh', process=False)
    path = ASSETS / f'{case.name}.png'
    pl = pv.Plotter(off_screen=True, window_size=(560, 400))
    pl.set_background('#f6f7f9')
    pl.add_mesh(poly(m), color='#65727b', smooth_shading=True, specular=.28, specular_power=20)
    pl.camera_position = CAM
    pl.add_text(f"{case.name} | sparse components={row['raw_components']}", font_size=11, color='#17212b')
    pl.show(screenshot=str(path), auto_close=True)
    return path


def main():
    rows = []
    for case in CASES.iterdir():
        if (case / 'gen/mesh.obj').exists() and (case / 'gen/final.obj').exists():
            row = measure(case)
            row['image'] = render(case, row).name
            rows.append(row)
    rows.sort(key=lambda r: (r['raw_components'], r['name']))
    (ASSETS / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    table = ''.join(f"<tr><td>{i+1}</td><th>{html.escape(r['name'])}</th><td>{r['raw_components']}</td><td>{r['raw_volume_mm3']:,}</td><td>{r['final_volume_mm3']:,}</td><td>{'yes' if r['watertight'] else 'no'}</td></tr>" for i, r in enumerate(rows))
    cards = ''.join(f"<figure><img src='report_assets/{r['image']}'><figcaption><b>{html.escape(r['name'])}</b><br>sparse raw components: {r['raw_components']}</figcaption></figure>" for r in rows)
    page = f'''<!doctype html><meta charset="utf-8"><title>Top-view Caliper Parameter Search</title>
    <style>body{{font:15px system-ui;background:#f6f7f9;color:#17212b;margin:30px}}main{{max-width:1800px;margin:auto}}img{{width:100%;background:white}}table{{border-collapse:collapse;background:white;margin:20px 0}}th,td{{padding:8px 12px;border:1px solid #d7dce2;text-align:right}}th{{text-align:left}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:18px}}figure{{margin:0;background:white;padding:10px;border-radius:8px}}figcaption{{padding:8px 4px}}.note{{max-width:1000px;line-height:1.55}}</style>
    <main><h1>Top-view-only caliper parameter search</h1><p class="note">Every case uses only <code>v_top.png</code>, seed 42, the same envelope, and FEA-off geometry guidance. Ranking uses the number of connected components in sparse <code>mesh.obj</code>, before post-processing drops all but the largest component. Lower is better.</p><img src="../../caliper_tearing_ablation_2026-09-17/report_assets/top_only_input.png" style="max-width:512px"><table><thead><tr><th>rank</th><th>candidate</th><th>sparse components</th><th>raw mm³</th><th>final mm³</th><th>final watertight</th></tr></thead><tbody>{table}</tbody></table><div class="grid">{cards}</div></main>'''
    (OUT / 'index.html').write_text(page)
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
