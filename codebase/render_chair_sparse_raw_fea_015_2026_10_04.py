"""Visualize the raw sparse OBJ's 15 mm density-mapped FEM displacement and stress."""
from __future__ import annotations

import json
import os

os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.spatial import cKDTree
import pyvista as pv
import trimesh

from draw_chair_physics_contour_2026_10_03 import render
from run_chair_existing_fea_on_015_2026_10_04 import OUT


YOUNG_REFERENCE_PA = 110e9
POISSON = .3


def main():
    case = OUT / 'evaluation/new_sparse_on/direct_full_mesh_15mm'
    grid = pv.read(case / 'displacement_p0_000001.vtu')
    rho = np.load(case / 'rho_cells.npy')
    cid = np.asarray(grid.cell_data['vtkOriginalCellIds'], dtype=np.int64)
    tet = grid.cells.reshape(-1, 5)[:, 1:]
    xyz = grid.points[tet]
    u = np.asarray(grid['displacement'])[tet]
    A = xyz[:, 1:] - xyz[:, :1]
    B = u[:, 1:] - u[:, :1]
    grad = np.linalg.solve(A, B).transpose(0, 2, 1)
    eps = .5*(grad + grad.transpose(0, 2, 1))
    E = .001 + rho[cid]**2 * (1 - .001)
    mu = E/(2*(1+POISSON))
    lam = E*POISSON/((1+POISSON)*(1-2*POISSON))
    sigma = 2*mu[:, None, None]*eps
    sigma[:, range(3), range(3)] += lam[:, None]*np.trace(eps, axis1=1, axis2=2)[:, None]
    sx, sy, sz = sigma[:, 0, 0], sigma[:, 1, 1], sigma[:, 2, 2]
    vm = np.sqrt(.5*((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2) +
                 3*(sigma[:, 0, 1]**2 + sigma[:, 1, 2]**2 + sigma[:, 0, 2]**2))
    if not np.all(np.isfinite(vm)):
        raise RuntimeError('nonfinite von Mises stress')
    solid = rho[cid] > .5
    grid.cell_data['rho_raw_OBJ'] = rho[cid]
    grid.cell_data['von_mises_MPa'] = vm / 1e6
    grid.point_data['displacement_mm_assuming_E110GPa'] = (
        np.linalg.norm(np.asarray(grid['displacement']), axis=1)*1000/YOUNG_REFERENCE_PA)
    grid.save(case / 'fea_displacement_stress.vtu')

    raw = trimesh.load(case / 'aligned_full.obj', force='mesh', process=False)
    display = raw.simplify_quadric_decimation(face_count=150000)
    surface = pv.wrap(display)
    sampled = surface.sample(grid)
    valid = np.asarray(sampled['vtkValidPointMask'], bool)
    u_nodes = np.asarray(sampled['displacement'])
    if not valid.all():
        _, closest = cKDTree(grid.points).query(surface.points[~valid], workers=-1)
        u_nodes[~valid] = np.asarray(grid['displacement'])[closest]
    surface['displacement_mm_assuming_E110GPa'] = np.linalg.norm(u_nodes, axis=1)*1000/YOUNG_REFERENCE_PA
    centers = xyz.mean(axis=1)
    dist, nearest_solid = cKDTree(centers[solid]).query(surface.points, workers=-1)
    surface['von_mises_MPa'] = (vm[solid][nearest_solid]/1e6).astype(np.float32)
    surface.save(case / 'raw_surface_fea_contours.vtp')
    stress_p99 = float(np.percentile(vm[solid]/1e6, 99))
    stress_p995 = float(np.percentile(vm[solid]/1e6, 99.5))
    disp_p99 = float(np.percentile(surface['displacement_mm_assuming_E110GPa'], 99))
    disp_max = float(np.max(surface['displacement_mm_assuming_E110GPa']))
    meta = {'method': 'raw sparse OBJ sampled as solid/void on fixed 15 mm envelope FEM; linear tetra displacement and cell von Mises; nearest solid tetra color projected to display-decimated raw surface',
            'load': {'seat_N_minus_Z': 800, 'back_N_plus_Y': 200, 'fixed_feet': 4},
            'material': {'E_reference_Pa_for_displacement': YOUNG_REFERENCE_PA,
                         'nu': POISSON, 'SIMP_p': 2, 'Emin_over_E0': .001},
            'compliance_E0_equals_1': json.loads((case / 'dc_info.json').read_text())['compliance'],
            'raw_solid_volume_L': json.loads((case / 'geometry_summary.json').read_text())['solid_volume_liters'],
            'solid_tetrahedra': int(solid.sum()),
            'surface_fem_sample_valid_fraction': float(valid.mean()),
            'surface_to_nearest_solid_tet_distance_mm_p99': float(np.percentile(dist, 99)*1000),
            'solid_tet_von_mises_MPa_p99': stress_p99,
            'solid_tet_von_mises_MPa_p995': stress_p995,
            'surface_displacement_mm_p99': disp_p99,
            'surface_displacement_mm_max': disp_max,
            'limitations': ['Stress is a voxelized density-mapped envelope FEM proxy, not a conforming tetrahedralization of the OBJ.',
                            'Stress displayed at each surface point is taken from the nearest solid tetrahedron.',
                            'The raw OBJ occupies only part of the fixed BC region; void ersatz modulus can carry artificial load.',
                            'Displacement in mm assumes E=110 GPa.']}
    (case / 'contour_summary.json').write_text(json.dumps(meta, indent=2) + '\n')
    feet = json.loads((OUT.parents[4] / 'single_view_spec_2026-10-03/specification.json').read_text())['foot_centers_m']
    fig = plt.figure(figsize=(14, 11), facecolor='white')
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, .055],
                          left=.04, right=.94, top=.88, bottom=.09,
                          hspace=.15, wspace=.045)
    fig.suptitle('Raw sparse chair mesh · simultaneous-load FEA', fontsize=21, y=.98)
    fig.text(.5, .927, 'Seat 800 N −Z  +  Backrest 200 N +Y  |  four fixed feet  |  15 mm tetra mesh',
             ha='center', fontsize=13, color='#354b59')
    rows = [('von_mises_MPa', stress_p995, 'Von Mises stress proxy (MPa, 99.5% cap)'),
            ('displacement_mm_assuming_E110GPa', max(disp_p99, 1e-12),
             'Displacement (mm, E = 110 GPa, 99% cap)')]
    for i, (field, vmax, label) in enumerate(rows):
        for j, view in enumerate(('front', 'side')):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(render(surface, field, vmax, 'combined', view, feet))
            ax.axis('off')
            ax.set_title(('Front' if j == 0 else 'Right side') + ' · ' + ('stress' if i == 0 else 'displacement'),
                         loc='left', fontsize=12)
        cax = fig.add_subplot(gs[i, 2])
        cb = mpl.colorbar.ColorbarBase(cax, cmap=mpl.colormaps['turbo'],
                                      norm=mpl.colors.Normalize(0, vmax))
        cb.set_label(label, fontsize=10)
    fig.text(.04, .035, 'Undeformed raw OBJ shown. Purple arrows = loads; red pads = feet. '
             'Stress is projected from the nearest solid FEM tetrahedron; void material is excluded from the contour.',
             fontsize=10, color='#54636c')
    fig.savefig(case / 'fea_contours.png', dpi=180, bbox_inches='tight', pad_inches=.18)
    plt.close(fig)
    (case / 'index.html').write_text(f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Raw sparse chair FEA</title><style>body{{font:16px/1.55 system-ui;max-width:1400px;margin:25px auto;padding:0 20px;background:#eff3f5;color:#203b4a}}article{{background:white;padding:20px;border-radius:12px}}img{{width:100%}}a{{color:#17618c}}</style><article>
<h1>원본 Sparse 의자 OBJ · 동시 하중 FEA</h1><p>좌면 800 N −Z와 등받이 200 N +Y를 한 번에 적용했습니다. 위는 von Mises 응력 proxy, 아래는 E=110 GPa 가정의 변위입니다. 최대 표면 변위 {disp_max:.5f} mm, 99% 값 {disp_p99:.5f} mm입니다. 실물 OBJ에 맞춘 사면체가 아니라 원본 형상을 15 mm envelope FEM 셀에 사상한 해석입니다.</p>
<img src="fea_contours.png"><p><a href="raw_surface_fea_contours.vtp">표면 contour VTP</a> · <a href="fea_displacement_stress.vtu">FEM 변위·응력 VTU</a> · <a href="contour_summary.json">해석 상세</a></p>
<p>원본은 고정 BC 영역을 완전히 채우지 않습니다. 빈 셀에 남는 최소 강성이 힘을 전달할 수 있으므로 이 결과를 곧바로 구조 안전성 검증으로 간주할 수 없습니다.</p></article></html>''')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
