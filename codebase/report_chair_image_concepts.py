#!/usr/bin/env python3
"""Inspect chair image-concept dense meshes in the unchanged chair BC frame."""
from __future__ import annotations

import html
import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import label

from make_chair_domain import ROOT
from report_chair_dense_grid import frontmost_label
from run_chair_image_concept_dense import BASE, NAMES

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402


VIEWS = [('hero', 28, 40), ('front', 15, 0), ('right', 15, 90), ('top', 85, 0)]


def main() -> None:
    domain = ROOT / 'data_real/chair'
    env = trimesh.load(domain / 'original_DesignSpace.stl', force='mesh')
    grid = np.load(domain / 'voxel.npz')
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents))*1.5
    scale = float(env.extents.max())/2
    fix_labels, _ = label(grid['fix'])
    rows = []
    volumes = {}
    cards = []
    cases = [(name, BASE / name, BASE / name / 'input_512.png') for name in NAMES]
    cases.append(('diagonal_braced_multiview', BASE / 'diagonal_braced/multiview_registered',
                  BASE / 'diagonal_braced/input_512.png'))
    cases.append(('diagonal_braced_multiview_lr162', BASE / 'diagonal_braced/multiview_lr162',
                  BASE / 'diagonal_braced/input_512.png'))
    for condition in ('no_mid', 'no_mid_vw50', 'camera_proj_w3',
                      'camera_proj_w10', 'camera_proj_w25', 'camera_proj_w50'):
        cases.append((f'diagonal_braced_{condition}', BASE / 'diagonal_braced' / condition,
                      BASE / 'diagonal_braced/input_512.png'))
    for name, case, input_path in cases:
        mesh_path = case / 'dense/mesh_dense.obj'
        cache_path = case / 'dense_cache.npz'
        if not (mesh_path.exists() and cache_path.exists()):
            rows.append({'name': name, 'status': 'missing'})
            continue
        mesh = trimesh.load(mesh_path, force='mesh')
        indices = np.load(cache_path)['latent_index']
        occupancy = np.zeros((64, 64, 64), bool)
        occupancy[indices[:, 1], indices[:, 2], indices[:, 3]] = True
        labels, n_components = label(occupancy)
        load_id = frontmost_label(labels, grid['load'])
        foot_ids = [frontmost_label(labels, fix_labels == i) for i in range(1, 5)]
        connected = bool(load_id > 0 and all(i == load_id for i in foot_ids))
        volumes[name] = occupancy
        sheet = Image.new('RGB', (512*5, 548), 'white')
        draw = ImageDraw.Draw(sheet)
        input_image = Image.open(input_path).convert('RGB')
        sheet.paste(input_image, (0, 36))
        draw.text((12, 10), 'generated image input', fill='#263139')
        silhouettes = {}
        for j, (view, elev, azim) in enumerate(VIEWS, start=1):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rendered = Image.fromarray(render_lit(
                mesh, eye, center, up, size=512, fit_extent=scale,
                margin=1.15, color=(.19, .22, .25))).convert('RGB')
            sheet.paste(rendered, (512*j, 36))
            draw.text((512*j+12, 10), f'dense mesh · {view}', fill='#263139')
            target_view = {'front': 'v00_front_lo', 'right': 'v02_right_lo',
                           'top': 'v_top'}.get(view)
            if target_view:
                target_image = np.asarray(Image.open(
                    BASE / 'diagonal_braced/multiview_registered/input'
                    / f'{target_view}.png').convert('RGB'))
                generated_mask = np.asarray(rendered).min(axis=2) < 210
                target_mask = target_image.min(axis=2) < 210
                silhouettes[view] = {
                    'iou': round(float((generated_mask & target_mask).sum()
                                       / max((generated_mask | target_mask).sum(), 1)), 4),
                    'false_positive_fraction': round(float((generated_mask & ~target_mask).sum()
                                                     / max(generated_mask.sum(), 1)), 4)}
        preview = case / 'dense_comparison.png'
        sheet.save(preview)
        row = {'name': name, 'status': 'complete',
               'input': str(case / 'input/v00_front_lo.png' if '_multiview' in name
                            else BASE / 'diagonal_braced/multiview_lr162/input/v00_front_lo.png'
                            if any(part in name for part in ('_no_mid', '_camera_proj'))
                            else case / 'input_162.png'),
               'mesh': str(mesh_path), 'preview': str(preview),
               'active_voxels': int(len(indices)), 'mask_components': int(n_components),
               'four_feet_connected_to_load': connected,
               'silhouette_against_diagonal_image': silhouettes,
               'back_height_m': round(float(mesh.bounds[1, 2]), 4),
               'volume_litres': round(abs(float(mesh.volume))*1000, 3)}
        rows.append(row)
        relative = case.relative_to(BASE)
        cards.append(f'<article><h2>{html.escape(name)}</h2><a href="{relative}/dense_comparison.png">'
                     f'<img src="{relative}/dense_comparison.png"></a><p>'
                     f'<a href="{relative}/dense/mesh_dense.obj">dense OBJ</a> · '
                     f'<a href="{relative}/input_contact.png">registered inputs</a></p></article>'
                     if any(part in name for part in ('_multiview', '_no_mid', '_camera_proj')) else
                     f'<article><h2>{html.escape(name)}</h2><a href="{relative}/dense_comparison.png">'
                     f'<img src="{relative}/dense_comparison.png"></a><p>'
                     f'<a href="{relative}/dense/mesh_dense.obj">dense OBJ</a> · '
                     f'<a href="{relative}/input_162.png">162px input</a></p></article>')
    for row in rows:
        if row['status'] == 'complete':
            row['pairwise_voxel_iou'] = {
                other: round(float(np.logical_and(volumes[row['name']], vol).sum() /
                                   max(np.logical_or(volumes[row['name']], vol).sum(), 1)), 4)
                for other, vol in volumes.items() if other != row['name']}
    (BASE / 'dense_metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    final_case = BASE / 'diagonal_braced/no_mid_vw50'
    final_note = ''
    guided_case = BASE / 'diagonal_braced/camera_proj_w50'
    fidelity_path = BASE / 'diagonal_braced/fidelity_comparison.json'
    remesh_path = guided_case / 'remesh_edge_study.json'
    fea_path = guided_case / 'post_edge5/fea_independent/fea_tet_summary.json'
    if fidelity_path.exists() and remesh_path.exists() and fea_path.exists():
        fidelity = json.loads(fidelity_path.read_text())
        remesh = json.loads(remesh_path.read_text())['5mm']
        fea = json.loads(fea_path.read_text())
        old = fidelity['previous_final']['silhouettes']
        new = fidelity['camera_guided_5mm_final']['silhouettes']
        final_note += (
            '<article><h2>선택 결과: 카메라 정렬 투영 + 5 mm 최종 리메시</h2>'
            '<p>정면 IoU '
            f'{old["front"]["iou"]:.3f} → {new["front"]["iou"]:.3f}, '
            '측면 IoU '
            f'{old["right"]["iou"]:.3f} → {new["right"]["iou"]:.3f}. '
            f'체적 {fidelity["previous_final"]["volume_litres"]:.2f} → '
            f'{remesh["volume_litres"]:.2f} L. '
            f'단일 watertight 메시이며 fixed/load BC 포함률은 '
            f'{remesh["bc_containment"]["fix"]:.0%}/'
            f'{remesh["bc_containment"]["load"]:.0%}.</p>'
            '<p>5 mm 후보에서 준비한 독립 800 N 좌면 하중 FEA: '
            f'compliance {fea["compliance"]:.6f} J, '
            f'최대 변위 {fea["u_max"]*1000:.4f} mm. '
            '가는 대각 보강재와 접합부 디테일은 여전히 입력 이미지와 다릅니다. '
            '이 FEA는 단순화한 수직 좌면 하중 한 경우입니다.</p>'
            '<a href="diagonal_braced/fidelity_comparison.png">'
            '<img src="diagonal_braced/fidelity_comparison.png"></a>'
            '<p><a href="diagonal_braced/camera_proj_w50/post_edge5/final.obj">선택 OBJ</a> · '
            '<a href="diagonal_braced/fidelity_comparison.json">이미지 충실도</a> · '
            '<a href="diagonal_braced/camera_proj_w50/remesh_edge_study.json">리메시 비교</a> · '
            '<a href="diagonal_braced/camera_proj_w50/post_edge5/fea_independent/fea_tet_summary.json">FEA</a></p></article>'
        )
    if (final_case / 'metrics.json').exists():
        final = json.loads((final_case / 'metrics.json').read_text())
        status = 'passed' if final['geometry_valid'] else 'failed'
        fea_text = (f'Independent 800 N seat-load FEA: compliance '
                    f'{final["fea"]["compliance"]:.6f} J; maximum displacement '
                    f'{final["fea"]["u_max"]*1000:.4f} mm.') if 'fea' in final else 'FEA not run.'
        final_note += (f'<article><h2>이전 3D 후보 — {status}</h2>'
                      f'<p>One watertight component, volume {final["volume_litres"]:.2f} L; '
                      f'fixed BC containment {final["bc_containment"]["fix"]:.2%}, '
                      f'load BC containment {final["bc_containment"]["load"]:.2%}. '
                      f'{fea_text} This verifies only the simple seat-load case. '
                      f'The final geometry is thicker and rougher than the image concept.</p>'
                      '<a href="diagonal_braced/no_mid_vw50/post/final_contact.png">'
                      '<img src="diagonal_braced/no_mid_vw50/post/final_contact.png"></a>'
                      '<p><a href="diagonal_braced/no_mid_vw50/post/final.obj">final OBJ</a> · '
                      '<a href="diagonal_braced/no_mid_vw50/metrics.json">geometry + FEA metrics</a></p></article>')
    table = ''.join(f'<tr><td>{html.escape(row["name"])}</td><td>{row.get("mask_components", "-")}</td>'
                    f'<td>{row.get("four_feet_connected_to_load", "-")}</td>'
                    f'<td>{row.get("back_height_m", "-")}</td>'
                    f'<td>{row.get("volume_litres", "-")}</td></tr>' for row in rows)
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair image to 3D pilot</title>
<style>body{font:16px system-ui,sans-serif;background:#f4f6f8;color:#1d2730;max-width:1600px;margin:2rem auto;padding:0 1rem}article{background:white;padding:1rem;margin:1rem 0;border-radius:12px}img{width:100%}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #ccd3d9;padding:.5rem;text-align:left}</style>
<h1>의자 이미지 → 3D dense 파일럿</h1><p>세 콘셉트 이미지와 대각 브레이스형의 정렬 멀티뷰 해상도 실험에 동일한 chair envelope, 4개 fixed foot, seat load, support path, dense 생성 설정을 적용했습니다. 입력 이미지만 다릅니다. dense 연결성은 64³ mask 기준이며 최종 mesh/FEA 검증은 별도입니다.</p>
''' + final_note + '''<table><tr><th>image concept / condition</th><th>dense components</th><th>4 feet ↔ seat load</th><th>back height m</th><th>volume L</th></tr>''' + table + '''</table><p><a href="dense_metrics.json">수치 JSON</a> · <a href="contact_512.png">이미지 시안</a> · <a href="prompts.md">투시 이미지 프롬프트</a> · <a href="diagonal_braced/multiview_registered/prompts.md">정렬 멀티뷰 프롬프트</a> · <a href="RESULTS.md">결과 해설</a></p>''' + ''.join(cards) + '</html>'
    (BASE / 'dense_report.html').write_text(page)
    print(BASE / 'dense_report.html')


if __name__ == '__main__':
    main()
