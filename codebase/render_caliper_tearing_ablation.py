#!/usr/bin/env python3
"""Render a stage-by-stage visual and topology table for the tear ablation."""
import html
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh
from PIL import Image, ImageDraw

ROOT = Path('/home/goya/SDL/3d_qd')
OUT = ROOT / 'experiments/caliper_tearing_ablation_2026-09-17'
CASES = OUT / 'caliper'
ASSET = OUT / 'report_assets'
ASSET.mkdir(parents=True, exist_ok=True)
STAGES = ('mesh_dense_raw.obj', 'mesh_dense.obj', 'mesh.obj', 'hybrid.obj', 'final.obj')
LABEL = {'mesh_dense_raw.obj': 'dense raw', 'mesh_dense.obj': 'dense',
         'mesh.obj': 'sparse raw', 'hybrid.obj': 'boolean/post', 'final.obj': 'delivered'}
CAM = [(0.22, -0.31, 0.20), (0.0, 0.0, 0.0), (0, 0, 1)]
REF_DIR = ROOT / 'data/caliper/conditioning/builtin_kagome_trial'
REFS = ('v00_front_lo.png', 'v02_right_lo.png', 'v04_back_lo.png',
        'v06_left_lo.png', 'v_top.png', 'v_bottom.png')


def poly(mesh):
    return pv.PolyData(mesh.vertices, np.hstack([np.full((len(mesh.faces), 1), 3), mesh.faces]).ravel())


def stats(path):
    m = trimesh.load(path, force='mesh', process=False)
    return dict(faces=len(m.faces), components=len(m.split(only_watertight=False)),
                watertight=bool(m.is_watertight), volume_mm3=round(abs(m.volume) * 1e9, 1))


def render(path, out, title):
    m = trimesh.load(path, force='mesh', process=False)
    pl = pv.Plotter(off_screen=True, window_size=(480, 360))
    pl.set_background('#f7f8fa')
    pl.add_mesh(poly(m), color='#65727b', smooth_shading=True, specular=.25, specular_power=18)
    pl.camera_position = CAM
    pl.add_text(title, font_size=11, color='#15202b')
    pl.show(screenshot=str(out), auto_close=True)


def render_refs():
    """The exact six calibrated images supplied to Direct3D-S2 for this run."""
    tile_w, tile_h, label_h = 512, 512, 34
    canvas = Image.new('RGB', (3 * tile_w, 2 * (tile_h + label_h)), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, name in enumerate(REFS):
        image = Image.open(REF_DIR / name).convert('RGB')
        image.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = (i % 3) * tile_w + (tile_w - image.width) // 2
        y = (i // 3) * (tile_h + label_h) + label_h
        canvas.paste(image, (x, y))
        draw.text(((i % 3) * tile_w + 12, (i // 3) * (tile_h + label_h) + 9),
                  name.removesuffix('.png'), fill='#17212b')
    path = ASSET / 'reference_views.png'
    canvas.save(path)
    return path


def render_top_only_ref():
    """Standalone image used by the top-only control row."""
    src = REF_DIR / 'v_top.png'
    image = Image.open(src).convert('RGB')
    image.save(ASSET / 'top_only_input.png')
    return ASSET / 'top_only_input.png'


def main():
    reference = render_refs()
    top_only_ref = render_top_only_ref()
    rows = []
    imgs = []
    # include historical failure signatures as context, then every finished new case.
    historic = ROOT / 'experiments/caliper_parameter_sweep_2026-09-17/caliper_parameter_sweep'
    candidates = [('historic_control', historic / 'control/gen'),
                  ('historic_guarded_mid', historic / 'guarded_mid/gen')]
    candidates += [(p.name, p / 'gen') for p in sorted(CASES.glob('*')) if (p / 'gen').exists()]
    for name, directory in candidates:
        row = {'case': name, 'stages': {}}
        for stage in STAGES:
            source = directory / stage
            if not source.exists():
                continue
            st = stats(source)
            row['stages'][LABEL[stage]] = st
            image = ASSET / f'{name}_{stage[:-4]}.png'
            render(source, image, f'{name}: {LABEL[stage]} | components={st["components"]}')
            imgs.append((name, LABEL[stage], image))
        rows.append(row)
    # five stage columns, one case per visual row
    w, h = 384, 288
    canvas = Image.new('RGB', (w * len(STAGES), h * len(rows)), '#f7f8fa')
    lookup = {(a, b): c for a, b, c in imgs}
    for y, row in enumerate(rows):
        for x, stage in enumerate(STAGES):
            im = lookup.get((row['case'], LABEL[stage]))
            if im:
                canvas.paste(Image.open(im).convert('RGB').resize((w, h), Image.Resampling.LANCZOS), (x*w, y*h))
    gallery = ASSET / 'stage_gallery.png'
    canvas.save(gallery)
    (ASSET / 'topology.json').write_text(json.dumps(rows, indent=2) + '\n')
    trs = []
    for row in rows:
        cells = ''.join('<td>' + html.escape(str(row['stages'].get(LABEL[s], {}).get('components', '—'))) + '</td>' for s in STAGES)
        trs.append(f'<tr><th>{html.escape(row["case"])}</th>{cells}</tr>')
    names = ''.join(f'<th>{LABEL[s]} components</th>' for s in STAGES)
    page = f'''<!doctype html><meta charset="utf-8"><title>Caliper tear ablation</title>
    <style>body{{font:15px system-ui;margin:30px;color:#17212b;background:#f7f8fa}}main{{max-width:2000px;margin:auto}}img{{width:100%;background:white}}table{{border-collapse:collapse;background:white;margin:20px 0}}th,td{{padding:8px 12px;border:1px solid #d8dde3;text-align:right}}th:first-child{{text-align:left}}p{{max-width:1000px;line-height:1.55}}</style>
    <main><h1>Disk-brake caliper: tear-mechanism ablation</h1><p>Rows use the six conditioning images below, except <code>top_only_control</code>, which uses only the standalone top-view image shown next. The count is measured separately after dense generation, sparse refinement, boolean/post union, and final remesh. A fall from many sparse components to one post component means the post step discarded fragments; it does not mean sparse refinement preserved connectivity.</p><h2>Reference conditioning views (six-view runs)</h2><img src="report_assets/{reference.name}"><h2>Top-only input</h2><img src="report_assets/{top_only_ref.name}" style="max-width:512px"><h2>Top-only delivered mesh</h2><p><b>v_top only · dense 6 components → sparse 25 components → delivered 1 component</b></p><img src="report_assets/top_only_control_final.png" style="max-width:900px"><h2>Pipeline stages</h2><img src="report_assets/stage_gallery.png"><table><thead><tr><th>case</th>{names}</tr></thead><tbody>{''.join(trs)}</tbody></table></main>'''
    (OUT / 'index.html').write_text(page)
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
