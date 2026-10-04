"""Render and report the matched-volume tetra-element SIMP baseline."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from run_chair_msh_simp_topopt_2026_10_04 import BASE, OUT, SPEC


CHAIR = (BASE / 'text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03'
         / 'llm_text_proposal_round_03_2026-10-03'
         / 'simultaneous_load_dense_sparse_2026-10-04/sparse_evaluation/round_01'
         / 'mesh_cases/sparse_d0_s1/aligned_main.obj')
CHAIR_RESULTS = (BASE / 'text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03'
                 / 'llm_text_proposal_round_03_2026-10-03'
                 / 'simultaneous_load_dense_sparse_2026-10-04/sparse_combined_evaluation.json')


def render(mesh, envelope, camera, color):
    pl = pv.Plotter(off_screen=True, window_size=(610, 610))
    pl.set_background('#fbfcfe')
    pl.enable_anti_aliasing('msaa')
    pl.add_mesh(mesh, color=color, smooth_shading=False, ambient=.3,
                diffuse=.7, specular=.14, specular_power=18)
    pl.add_mesh(envelope, color='#62bad2', opacity=.055)
    pl.camera_position = camera
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = .59
    img = Image.fromarray(pl.screenshot(return_img=True)).convert('RGB')
    pl.close()
    return img


def main() -> None:
    summary = json.loads((OUT / 'summary.json').read_text())
    history = json.loads((OUT / 'history.json').read_text())
    grid = pv.read(OUT / 'optimized_density.vtu')
    rho = np.asarray(grid.cell_data['density'])
    volumes = grid.compute_cell_sizes(length=False, area=False, volume=True).cell_data['Volume']
    thresholds = {}
    for cutoff in (.2, .3, .5):
        mask = rho >= cutoff
        thresholds[str(cutoff)] = {'tetrahedra': int(mask.sum()),
                                   'binary_volume_liters': float(volumes[mask].sum()*1000)}
    binary_meta = json.loads((OUT / 'binary_meta.json').read_text())
    binary_fea = json.loads((OUT / 'dc_binary_matched_info.json').read_text())
    display_cutoff = float(binary_meta['threshold'])
    optimized = grid.threshold(display_cutoff, scalars='density').extract_surface()
    # Face-adjacent tetra components of the thresholded material. This is a
    # geometry check, separate from the ersatz-material FEM solve.
    all_tets = grid.cells.reshape(-1, 5)[:, 1:]
    selected = rho >= display_cutoff
    selected_tets = all_tets[selected]
    faces = np.sort(np.concatenate([selected_tets[:, [0, 1, 2]],
                                    selected_tets[:, [0, 1, 3]],
                                    selected_tets[:, [0, 2, 3]],
                                    selected_tets[:, [1, 2, 3]]]), axis=1)
    origins = np.tile(np.arange(len(selected_tets)), 4)
    _, inverse = np.unique(faces, axis=0, return_inverse=True)
    ordering = np.argsort(inverse)
    sorted_face = inverse[ordering]
    adjacent = np.flatnonzero(sorted_face[1:] == sorted_face[:-1])
    start = origins[ordering[adjacent]]
    end = origins[ordering[adjacent + 1]]
    graph = coo_matrix((np.ones(len(start)*2),
                        (np.r_[start, end], np.r_[end, start])),
                       shape=(len(selected_tets), len(selected_tets))).tocsr()
    n_components, labels = connected_components(graph)
    volume_by_component = np.bincount(labels, weights=volumes[selected])
    main_component = int(volume_by_component.argmax())
    spec_data = np.load(SPEC / 'voxel.npz')
    centers = grid.points[selected_tets].mean(axis=1)
    voxel_idx = np.clip(np.floor((centers-spec_data['origin']) /
                                 spec_data['pitch_xyz']).astype(int), 0, 63)
    bc_connected = {}
    for key in ('fix', 'load', 'back_load'):
        bc = spec_data[key][voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]].astype(bool)
        bc_connected[key] = float(np.mean(labels[bc] == main_component)) if bc.any() else 0.0
    chair = pv.read(CHAIR)
    envelope = pv.read(SPEC / 'envelope.stl')
    center = np.array(envelope.center)
    radius = 2.0
    iso = [list(center + np.array([1.0, -1.35, .9])*radius),
           center.tolist(), [0, 0, 1]]
    side = [list(center + np.array([2.0, 0, .35])),
            center.tolist(), [0, 0, 1]]
    views = [('Generated chair · same FEM evaluation', chair, iso, '#89939e'),
             (f'TO binary · ρ ≥ {display_cutoff:.3f}', optimized, iso, '#c27d50'),
             ('TO binary · side view', optimized, side, '#c27d50')]
    sheet = Image.new('RGB', (3*610, 720), '#fbfcfe')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 25)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    draw.text((25, 13), 'Chair envelope SIMP topology optimization',
              font=font, fill='#223447')
    draw.text((25, 54), 'Matched ~25.44 L · 800 N seat + 200 N back · p=2 · disconnected foot patches',
              font=small, fill='#4c6574')
    for i, (title, mesh, camera, color) in enumerate(views):
        sheet.paste(render(mesh, envelope, camera, color), (i*610, 110))
        draw.text((i*610+18, 111), title, font=small, fill='#243a48')
    sheet.save(OUT / 'comparison.png')
    x = [row['iteration'] for row in history]
    y = [row['compliance']/1e6 for row in history]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(x, y, 'o-', color='#bd784c', lw=2, ms=4)
    ax.set(xlabel='OC iteration', ylabel='Simultaneous-load compliance (million)',
           title='Matched-volume direct tetra-element optimization')
    ax.grid(alpha=.24)
    fig.tight_layout()
    fig.savefig(OUT / 'convergence.png', dpi=170)
    plt.close(fig)
    baseline = next(x for x in json.loads(CHAIR_RESULTS.read_text())
                    if x['id'] == 'sparse_d0_s1')
    comparison = {'generated_chair_mass_liters': baseline['mass_liters'],
                  'generated_chair_compliance': baseline['combined_compliance'],
                  'topopt_continuous_mass_liters': summary['final_volume_liters'],
                  'topopt_continuous_compliance': summary['final_compliance'],
                  'topopt_binary_mass_liters': binary_meta['binary_volume_liters'],
                  'topopt_binary_compliance_ersatz_fem': binary_fea['compliance'],
                  'compliance_ratio_topopt_to_chair': summary['final_compliance'] /
                        baseline['combined_compliance'],
                  'threshold_meshes': thresholds,
                  'threshold_rendered': display_cutoff,
                  'binary_components': int(n_components),
                  'binary_main_component_volume_fraction': float(volume_by_component.max()/volume_by_component.sum()),
                  'bc_main_component_fraction': bc_connected,
                  'rendered_surface': 'matched-volume thresholded tetrahedra; ersatz FEM evaluated, but BC connection gate fails',
                  'baseline_and_topopt_same_envelope_fem': True}
    (OUT / 'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chair envelope SIMP topology optimization</title>
<style>body{{font:16px/1.55 system-ui;max-width:1500px;margin:30px auto;padding:0 22px;background:#f0f4f6;color:#20384a}}article{{background:white;padding:20px;border-radius:12px;margin:15px 0}}img{{width:100%;height:auto}}.small{{max-width:850px}}code{{overflow-wrap:anywhere}}</style>
<h1>의자 envelope 직접 위상최적화</h1><article><p>고정된 35 mm 사면체 FEM 메시에서 요소 밀도를 직접 최적화했습니다. 좌면 800 N −Z와 등받이 200 N +Y는 하나의 solve에 함께 적용했습니다. 네 발 고정 영역과 두 하중 패치는 항상 고밀도입니다. SIMP p=2, sensitivity filter 45 mm, 최종 밀도 체적은 {summary['final_volume_liters']:.3f} L입니다.</p>
<p><b>동일 FEM에서 생성 의자:</b> {baseline['mass_liters']:.3f} L, C={baseline['combined_compliance']/1e6:.3f}×10⁶.<br><b>TO 연속 밀도 결과:</b> {summary['final_volume_liters']:.3f} L, C={summary['final_compliance']/1e6:.3f}×10⁶. 비율 {comparison['compliance_ratio_topopt_to_chair']:.3f}.</p>
<p><b>동일 체적 이진화:</b> ρ≥{display_cutoff:.3f} 요소, 체적 {binary_meta['binary_volume_liters']:.3f} L, ersatz-material FEM C={binary_fea['compliance']/1e6:.3f}×10⁶. 그러나 이진 형상은 {n_components}개 face-connected 성분이며, 주 성분에는 fix 요소의 {bc_connected['fix']:.1%}만 연결됩니다. 두 하중 패치는 주 성분에 연결됩니다. <b>형상 연결성 검사에는 실패</b>하므로 이진화 결과를 완성된 구조물로 해석하지 않습니다.</p></article>
<article><img src="comparison.png" alt="generated chair and topology optimized tetra elements"></article>
<article><h2>수렴</h2><img class="small" src="convergence.png"><p><a href="history.json">Iteration data</a> · <a href="comparison.json">Comparison</a> · <a href="optimized_density.vtu">Optimized tetra density VTU</a></p></article></html>'''
    (OUT / 'index.html').write_text(page)
    print(OUT / 'index.html')
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
