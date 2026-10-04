#!/usr/bin/env python3
"""Combined seat + backrest loading contour on the original chair surface."""
from __future__ import annotations

import json
import os

os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pyvista as pv
import trimesh

from draw_chair_physics_contour_2026_10_03 import BASE, OUT, MESH, SPEC, render


def main() -> None:
    original = trimesh.load(MESH, force='mesh', process=False)
    display = original.simplify_quadric_decimation(face_count=150000)
    surface = pv.wrap(display)
    seat = pv.read(OUT / 'seat/displacement_p0_000001.vtu')
    back = pv.read(OUT / 'back/displacement_p0_000001.vtu')
    if seat.n_points != back.n_points or not np.allclose(seat.points, back.points, atol=1e-10):
        raise RuntimeError('Seat and backrest FEA fields do not share the same tetrahedral mesh')
    u_seat = np.asarray(seat['displacement'])
    u_back = np.asarray(back['displacement'])
    combined = u_seat + u_back  # exact superposition for this linear-elastic model
    volume = seat.copy(deep=True)
    volume['u_seat_xyz_m'] = u_seat
    volume['u_back_xyz_m'] = u_back
    volume['u_combined_xyz_m'] = combined
    volume['u_combined_mm'] = np.linalg.norm(combined, axis=1) * 1000
    volume.save(OUT / 'combined_tetra_displacement.vtu')
    sampled = surface.sample(volume)
    valid = float(np.mean(sampled['vtkValidPointMask']))
    if valid < .999:
        raise RuntimeError(f'Combined displacement valid at only {valid:.1%} surface vertices')
    surface['combined_u_xyz_m'] = np.asarray(sampled['u_combined_xyz_m'])
    surface['combined_u_mm'] = np.linalg.norm(surface['combined_u_xyz_m'], axis=1) * 1000
    surface.save(OUT / 'combined_surface_contour.vtp')
    vmax = float(surface['combined_u_mm'].max())
    spec = json.loads(SPEC.read_text())
    fig = plt.figure(figsize=(17.8, 6.8), facecolor='white')
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, .055],
                            left=.018, right=.96, top=.83, bottom=.16, wspace=.055)
    fig.suptitle('Combined chair loading · displacement contour',
                 fontsize=21, weight='bold', y=.97, color='#1b2b37')
    fig.text(.5, .895, 'Seat 800 N −Z  +  Backrest 200 N +Y  |  four fixed feet',
             fontsize=14, ha='center', color='#344c5b')
    for i, (view, title) in enumerate((('front', 'Front'), ('side', 'Right side'),
                                       ('detail', 'Backrest detail'))):
        frame = render(surface, 'combined_u_mm', vmax, 'combined', view,
                       spec['foot_centers_m'])
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(frame)
        ax.set_axis_off()
        ax.set_title(title, fontsize=13, loc='left', color='#233643', pad=8)
    cb_ax = fig.add_subplot(grid[0, 3])
    cb = mpl.colorbar.ColorbarBase(cb_ax, cmap=mpl.colormaps['turbo'],
                                    norm=mpl.colors.Normalize(0, vmax),
                                    orientation='vertical')
    cb.set_label('Combined displacement |u| (mm)', fontsize=11)
    cb.ax.tick_params(labelsize=9)
    fig.text(.03, .085,
             f'Maximum surface displacement: {vmax:.4f} mm. '
             'Purple arrows show both loads; red pads mark four fixed feet. Undeformed mesh shown.',
             fontsize=11.4, color='#20333e')
    fig.text(.03, .045,
             'The two displacement vectors are superposed on the same linear-elastic 35 mm tetrahedral '
             'FEA model of registered 64³ repaired density, then sampled onto the original chair surface '
             '(150k-face display decimation). This is a displacement proxy, not direct-OBJ stress.',
             fontsize=9.1, color='#64747d')
    fig.savefig(OUT / 'combined_load_contour.png', dpi=210, bbox_inches='tight', pad_inches=.22)
    fig.savefig(OUT / 'combined_load_contour.pdf', bbox_inches='tight', pad_inches=.22)
    plt.close(fig)
    meta = json.loads((OUT / 'source.json').read_text())
    meta['combined_loading'] = {
        'method': 'linear superposition of nodal displacement vectors from the two load-case solves',
        'mesh_identical': True,
        'seat_force_N_xyz': [0, 0, -800],
        'back_force_N_xyz': [0, 200, 0],
        'surface_sample_valid_fraction': valid,
        'maximum_surface_displacement_mm': vmax,
        'figure_png': str(OUT / 'combined_load_contour.png'),
        'figure_pdf': str(OUT / 'combined_load_contour.pdf'),
        'tetra_field': str(OUT / 'combined_tetra_displacement.vtu'),
        'surface_field': str(OUT / 'combined_surface_contour.vtp'),
    }
    (OUT / 'source.json').write_text(json.dumps(meta, indent=2) + '\n')
    (OUT / 'index.html').write_text(f'''<!doctype html><html lang="ko"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>Combined chair FEA contour</title>
    <style>body{{font:16px/1.5 system-ui;max-width:1550px;margin:26px auto;padding:0 20px;
    background:#f1f4f5;color:#182b38}}article{{background:white;padding:22px;border-radius:12px}}
    img{{width:100%;height:auto}}a{{color:#1b6496}}</style><article>
    <h1>좌판·등받이 동시 하중 변위 contour</h1>
    <p>좌판 800 N −Z와 등받이 200 N +Y를 동시에 적용했다. 동일한 선형탄성 모델의
    두 변위 벡터장을 합성했으며, 원본 의자 형상의 최대 표면 변위는 {vmax:.4f} mm다.</p>
    <img src="combined_load_contour.png" alt="Combined loading displacement contour">
    <p><a href="combined_load_contour.pdf">논문용 PDF</a> ·
    <a href="combined_surface_contour.vtp">표면 contour VTP</a> ·
    <a href="combined_tetra_displacement.vtu">사면체 변위장 VTU</a> ·
    <a href="../tapered_examples_2026-10-03/final_meshes/tapered_original.obj">원본 OBJ</a> ·
    <a href="source.json">해석 조건</a> ·
    <a href="physics_contours.png">개별 하중 비교 그림</a></p>
    <p>색은 응력이 아닌 변위 크기다. 공통 35 mm FEA/64³ 밀도장을 원본 형상에 사상한
    proxy contour이므로 원본 OBJ를 직접 사면체화한 응력 해석으로 해석하지 않는다.</p>
    </article></html>''')
    (OUT / 'REPORT.md').write_text(
        '# Combined-load chair displacement contour\n\n'
        f"Original mesh: {MESH}\n\n"
        'Simultaneous loads: seat 800 N -Z and backrest 200 N +Y; four fixed feet. '
        'The same linear-elastic stiffness matrix and fixed-DOF set are used in both solves, '
        'so their nodal displacement vectors are superposed before taking magnitude.\n\n'
        f"Maximum surface displacement: {vmax:.6f} mm. Surface sample valid fraction: {valid:.3f}.\n\n"
        '35 mm tetrahedral FEA of registered 64³ repaired density, projected to a 150k-face '
        'display copy of the exact original chair OBJ. It is a displacement proxy, not direct-OBJ stress.\n\n'
        f"Figure: {OUT / 'combined_load_contour.png'}\n"
        f"PDF: {OUT / 'combined_load_contour.pdf'}\nViewer: {OUT / 'index.html'}\n")
    print('maximum combined displacement mm', vmax)
    print(OUT / 'combined_load_contour.png')


if __name__ == '__main__':
    main()
