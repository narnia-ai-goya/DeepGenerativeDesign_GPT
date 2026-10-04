#!/usr/bin/env python3
"""Paper-style two-load displacement contours on the tapered original chair mesh.

The FEM solve uses the registered 64^3 repaired density on the common 35 mm
tetrahedral design domain.  Only its displacement field is sampled onto a
display-decimated copy of the requested original surface; this is not direct
tetrahedralization or stress evaluation of the OBJ.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
import pyvista as pv
import trimesh

from make_chair_domain import ROOT

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'tapered_original_physics_contour_2026-10-03'
MESH = BASE / 'tapered_examples_2026-10-03/final_meshes/tapered_original.obj'
SPEC = BASE / 'single_view_spec_2026-10-03/specification.json'
CASES = [('seat', 'Seat load · 800 N, −Z'), ('back', 'Backrest load · 200 N, +Y')]


def render(surface: pv.PolyData, field: str, vmax: float,
           case: str, view: str, feet: list[list[float]]) -> Image.Image:
    pl = pv.Plotter(off_screen=True, window_size=(820, 760))
    pl.set_background('white')
    pl.enable_anti_aliasing('msaa')
    pl.add_mesh(surface, scalars=field, cmap='turbo', clim=(0, vmax),
                show_scalar_bar=False, smooth_shading=True,
                ambient=.46, diffuse=.54, specular=.13, specular_power=24)
    for x, y in feet:
        marker = pv.Cylinder(center=(x, y, .007), direction=(0, 0, 1),
                             radius=.039, height=.012, resolution=28)
        pl.add_mesh(marker, color='#bb4135', lighting=False)
    if case in ('seat', 'combined') and view != 'detail':
        arrow = pv.Arrow(start=(0, -.035, .76), direction=(0, 0, -1),
                         scale=.18, shaft_radius=.035, tip_radius=.13, tip_length=.27)
        pl.add_mesh(arrow, color='#7856a0', lighting=False)
    if case in ('back', 'combined'):
        arrow = pv.Arrow(start=(0, .085, .78), direction=(0, 1, 0),
                         scale=.17, shaft_radius=.035, tip_radius=.13, tip_length=.27)
        pl.add_mesh(arrow, color='#7856a0', lighting=False)
    center = np.array([0., .17, .76]) if view == 'detail' else np.array([0., .01, .46])
    if view == 'front':
        eye = center + np.array([.48, -1.7, .43])
    elif view == 'side':
        eye = center + np.array([1.8, -.36, .37])
    else:
        eye = center + np.array([.55, -1.6, .30])
    pl.camera_position = [eye, center, (0, 0, 1)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = .24 if view == 'detail' else .54
    pl.add_light(pv.Light(position=(1, -1, 2), focal_point=center, intensity=.85))
    pl.add_light(pv.Light(position=(-1, .6, 1.2), focal_point=center, intensity=.35))
    frame = Image.fromarray(pl.screenshot(return_img=True, transparent_background=True)).convert('RGBA')
    pl.close()
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    physical = trimesh.load(MESH, force='mesh', process=False)
    display = physical.simplify_quadric_decimation(face_count=150000)
    display.export(OUT / 'display_decimation.obj')
    surface = pv.wrap(display)
    load_fields = {}
    extrema = {}
    for case, _ in CASES:
        volume = pv.read(OUT / case / 'displacement_p0_000001.vtu')
        sampled = surface.sample(volume)
        validity = float(np.mean(sampled['vtkValidPointMask']))
        if validity < .999:
            raise RuntimeError(f'{case} FEM field sampled at only {validity:.1%} of surface vertices')
        vector = np.asarray(sampled['displacement'])
        displacement_mm = np.linalg.norm(vector, axis=1) * 1000
        field = f'{case}_displacement_mm'
        surface[field] = displacement_mm
        load_fields[case] = field
        extrema[case] = {'surface_max_mm': float(displacement_mm.max()),
                         'surface_p99_mm': float(np.percentile(displacement_mm, 99)),
                         'surface_sample_valid_fraction': validity,
                         'compliance_J': float(json.loads((OUT / case / 'dc_info.json').read_text())['compliance']),
                         'FEM_field': str(OUT / case / 'displacement_p0_000001.vtu')}
        print(case, extrema[case], flush=True)
    surface.save(OUT / 'surface_displacement_fields.vtp')
    spec = json.loads(SPEC.read_text())
    feet = spec['foot_centers_m']
    fig = plt.figure(figsize=(15.2, 11.8), facecolor='white')
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, .065], height_ratios=[1, 1],
                            left=.035, right=.94, top=.925, bottom=.095,
                            wspace=.045, hspace=.20)
    fig.suptitle('Chair structural response · surface displacement contours',
                 fontsize=20, weight='bold', y=.985, color='#1b2b37')
    for row, (case, label) in enumerate(CASES):
        vmax = extrema[case]['surface_max_mm']
        for col, view in enumerate(('front', 'side')):
            frame = render(surface, load_fields[case], vmax, case, view, feet)
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(frame)
            ax.set_axis_off()
            ax.set_title(f'{label}  |  {view} view', loc='left', fontsize=13,
                         color='#233643', pad=8)
            ax.text(.02, .035, f'max |u| = {vmax:.4f} mm', transform=ax.transAxes,
                    fontsize=11, color='#13242e',
                    bbox={'facecolor':'white','edgecolor':'#d5dee2','alpha':.92,'pad':5})
        cax = fig.add_subplot(grid[row, 2])
        cb = mpl.colorbar.ColorbarBase(cax, cmap=mpl.colormaps['turbo'],
                                        norm=mpl.colors.Normalize(vmin=0, vmax=vmax),
                                        orientation='vertical')
        cb.set_label('Displacement magnitude |u| (mm)', fontsize=11)
        cb.ax.tick_params(labelsize=9)
    fig.text(.05, .035,
             'Purple arrow: applied load     Red markers: four fixed feet     '
             'Undeformed geometry shown; displacement is color only.\n'
             '35 mm tetrahedral FEA of registered 64³ repaired density, sampled onto a display-decimated '
             'copy of the original OBJ. This is a proxy contour, not a direct-OBJ stress plot.',
             fontsize=10.1, color='#53636d', ha='left', va='top')
    fig.savefig(OUT / 'physics_contours.png', dpi=190, bbox_inches='tight', pad_inches=.24)
    fig.savefig(OUT / 'physics_contours.pdf', bbox_inches='tight', pad_inches=.24)
    plt.close(fig)
    source = json.loads((OUT / 'source.json').read_text())
    source.update({'display_decimation_faces': int(len(display.faces)),
                   'original_faces': int(len(physical.faces)),
                   'contour_quantity': 'magnitude of FE displacement (mm), sampled onto original chair surface',
                   'solver_material': {'E0_Pa': 110e9, 'Emin_Pa': 110e6, 'nu': .3, 'SIMP_penal': 2},
                   'cases': extrema, 'image': str(OUT / 'physics_contours.png'),
                   'pdf': str(OUT / 'physics_contours.pdf'),
                   'surface_field': str(OUT / 'surface_displacement_fields.vtp')})
    (OUT / 'source.json').write_text(json.dumps(source, indent=2) + '\n')
    (OUT / 'index.html').write_text(f'''<!doctype html><html lang="ko"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>Chair physics contours</title>
    <style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1500px;margin:25px auto;padding:0 20px;
    background:#f1f4f5;color:#172936}}article{{background:white;padding:20px;border-radius:12px}}
    img{{width:100%;height:auto}}a{{color:#1b6496}}code{{overflow-wrap:anywhere}}</style>
    <article><h1>원본 의자 mesh · 두 하중 변위 contour</h1>
    <p>좌판 800 N (−Z): 최대 표면 변위 {extrema['seat']['surface_max_mm']:.4f} mm ·
    등받이 200 N (+Y): {extrema['back']['surface_max_mm']:.4f} mm.
    색은 변위 크기이며 형상은 변형시키지 않았다. 원본 OBJ를 64³ 밀도장으로 변환해
    공통 35 mm 사면체 영역에서 선형탄성 FEA를 수행하고, 변위장을 원본 형상의 표시용
    경량 mesh 표면에 사상했다. 따라서 직접 OBJ 응력 해석 결과로 해석하면 안 된다.</p>
    <img src="physics_contours.png" alt="Chair seat and backrest load displacement contours">
    <p><a href="physics_contours.pdf">논문용 PDF</a> ·
    <a href="surface_displacement_fields.vtp">두 변위장 포함 VTP</a> ·
    <a href="display_decimation.obj">표시용 OBJ</a> ·
    <a href="source.json">해석 조건·출처</a> ·
    <a href="../tapered_examples_2026-10-03/final_meshes/tapered_original.obj">원본 OBJ</a></p>
    </article></html>''')
    (OUT / 'REPORT.md').write_text(
        '# Original chair physics contours\n\n'
        f"Original surface: {MESH}\n\n"
        f"Seat 800 N -Z: max surface |u| {extrema['seat']['surface_max_mm']:.6f} mm; "
        f"compliance {extrema['seat']['compliance_J']:.6e} J.\n"
        f"Backrest 200 N +Y: max surface |u| {extrema['back']['surface_max_mm']:.6f} mm; "
        f"compliance {extrema['back']['compliance_J']:.6e} J.\n\n"
        'The common 35 mm tetrahedral FEA uses a registered 64^3 repaired-density proxy. '
        'The displacement field is sampled onto a 150k-face display decimation of the same original OBJ; '
        'this is not direct tetrahedral FEA or von Mises stress on the OBJ.\n\n'
        f"Figure: {OUT / 'physics_contours.png'}\nPDF: {OUT / 'physics_contours.pdf'}\n"
        f"Viewer: {OUT / 'index.html'}\n")
    print(OUT / 'physics_contours.png')


if __name__ == '__main__':
    main()
