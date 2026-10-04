#!/usr/bin/env python3
"""Audit and present the image-conditioned angular bracket QD pilot."""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image

from report_bracket_multiview_case_study import render
from run_connectivity_qd_sampling import ROOT


OUT = ROOT / 'experiments/bracket/angular_image_qd_loop_2026-09-24'
BASE = ROOT / ('experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/'
               'shape_language_angular_2026-09-23/full_multiview_angular_2026-09-23')
CASES = [('baseline_angular', BASE),
         ('triangular_truss', OUT / 'round_00/triangular_truss'),
         ('x_brace', OUT / 'round_00/x_brace'),
         ('off_axis_spine', OUT / 'round_01/off_axis_spine'),
         ('staggered_chevron', OUT / 'round_01/staggered_chevron')]


def render_fixed_top(path: Path, output: Path) -> None:
    mesh = pv.read(path)
    plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
    plotter.set_background('white')
    plotter.add_mesh(mesh, color='#929fa5', smooth_shading=False,
                     ambient=0.45, diffuse=0.55, specular=0.18)
    # One physical camera for every candidate. BCs therefore remain aligned.
    center = (0.014, -0.073, 0.031)
    plotter.camera_position = [(center[0], center[1], center[2] + 1), center, (0, 1, 0)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = 0.112
    plotter.screenshot(str(output))
    plotter.close()


def top_mask(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
    return rgb.min(axis=2) < 245


def normalized_hv(points: list[dict], mass_ref: float = 350.0,
                  compliance_ref: float = 0.006) -> float:
    """Exact 2D minimization HV, normalized by the fixed reference rectangle."""
    best_y = compliance_ref
    area = 0.0
    for point in sorted(points, key=lambda row: row['volume_cm3']):
        x, y = point['volume_cm3'], point['compliance_J']
        if x < mass_ref and y < best_y:
            area += (mass_ref - x) * (best_y - y)
            best_y = y
    return area / (mass_ref * compliance_ref)


def main() -> None:
    report = OUT / 'report'
    assets = report / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    rows = []
    masks = {}
    for name, case in CASES:
        final = case / 'post/final.obj'
        metrics_path = case / 'metrics.json' if name != 'baseline_angular' else OUT / 'baseline_metrics.json'
        if not final.exists() or not metrics_path.exists():
            rows.append({'name': name, 'status': 'pending', 'mesh': str(final)})
            continue
        metrics = json.loads(metrics_path.read_text())
        valid = bool(metrics.get('geometry_valid') and metrics.get('compliance_J') is not None)
        row = {'name': name, 'round': -1 if name == 'baseline_angular' else (1 if case.parent.name == 'round_01' else 0),
               'mesh': str(final), 'status': 'accepted' if valid else 'rejected',
               'volume_cm3': metrics.get('volume_cm3'),
               'compliance_J': metrics.get('compliance_J'),
               'bc_containment': metrics.get('bc_containment'),
               'bc_failure_detail': metrics.get('bc_failure_detail'),
               'watertight': metrics.get('watertight'), 'components': metrics.get('components')}
        render_fixed_top(final, assets / f'{name}_top.png')
        render(final, assets / f'{name}_iso.png', 'iso')
        render(final, assets / f'{name}_xminus.png', 'xminus')
        render(final, assets / f'{name}_xplus.png', 'xplus')
        if valid:
            masks[name] = top_mask(assets / f'{name}_top.png')
            row['top_solid_fraction'] = float(masks[name].mean())
        rows.append(row)
    accepted = [row for row in rows if row['status'] == 'accepted']
    for row in accepted:
        row['shape_distance_1_minus_iou'] = {}
        for other in accepted:
            a, b = masks[row['name']], masks[other['name']]
            row['shape_distance_1_minus_iou'][other['name']] = round(
                1 - float(np.logical_and(a, b).sum() / np.logical_or(a, b).sum()), 4)
    # Greedy niches by rendered final-mesh phenotype; keep a 2-objective Pareto set.
    niches = []
    for row in accepted:
        niche = next((n for n in niches if row['shape_distance_1_minus_iou'][n['representative']] < 0.10), None)
        if niche is None:
            niche = {'representative': row['name'], 'members': [], 'pareto_elites': []}
            niches.append(niche)
        niche['members'].append(row['name'])
    by_name = {row['name']: row for row in accepted}
    for niche in niches:
        for name in niche['members']:
            a = by_name[name]
            dominated = any(other != name and
                            by_name[other]['volume_cm3'] <= a['volume_cm3'] and
                            by_name[other]['compliance_J'] <= a['compliance_J'] and
                            (by_name[other]['volume_cm3'] < a['volume_cm3'] or
                             by_name[other]['compliance_J'] < a['compliance_J'])
                            for other in niche['members'])
            if not dominated:
                niche['pareto_elites'].append(name)
    cumulative_hv = []
    seen = []
    for row in rows:
        if row['status'] == 'accepted':
            seen.append(row)
        total = sum(normalized_hv([by_name[name] for name in niche['members'] if name in {r['name'] for r in seen}])
                    for niche in niches)
        cumulative_hv.append({'after': row['name'], 'accepted_so_far': len(seen),
                              'cumulative_qd_hv': round(total, 6)})
    archive = {'shape_descriptor': 'fixed-camera top projection of final mesh, binary solid mask',
               'shape_distance': '1 - intersection-over-union',
               'niche_threshold': 0.10,
               'quality_objectives': ['minimize volume_cm3', 'minimize independently measured compliance_J'],
               'feasibility': 'watertight; one component; fix/load BC containment >=0.99; independent FEA valid',
               'qd_hv_reference': {'volume_cm3': 350.0, 'compliance_J': 0.006},
               'qd_hv_definition': 'sum of normalized exact 2D Pareto HV across occupied shape niches; cumulative in evaluation order',
               'cumulative_qd_hv': cumulative_hv, 'rows': rows, 'niches': niches}
    (OUT / 'archive.json').write_text(json.dumps(archive, indent=2, ensure_ascii=False) + '\n')
    cards = []
    for row in rows:
        name = html.escape(row['name'])
        if row['status'] == 'rejected':
            src = Path(row['mesh']).parents[1] / 'input/그림1.png'
            if src.exists():
                shutil.copyfile(src, assets / f'{row["name"]}_input.png')
            cards.append(f'<article><h2>{name} — 제외</h2><p>최종 BC 포함률: '
                         f'{html.escape(str(row["bc_containment"]))}. '
                         f'{html.escape(str(row.get("bc_failure_detail") or ""))} 품질 점수 계산에서 제외.</p>'
                         f'<div class="images"><img src="assets/{name}_input.png" alt="input">'
                         f'<img src="assets/{name}_top.png" alt="final top">'
                         f'<img src="assets/{name}_iso.png" alt="final 3D">'
                         f'<img src="assets/{name}_xminus.png" alt="final side x-">'
                         f'<img src="assets/{name}_xplus.png" alt="final side x+"></div></article>')
            continue
        if row['status'] != 'accepted':
            cards.append(f'<article><h2>{name} — {html.escape(row["status"])}</h2><p>{html.escape(row["mesh"])}</p></article>')
            continue
        src = BASE.parent / 'outer_wall_consistent_multiview_2026-09-23/input/그림1.png' if row['name'] == 'baseline_angular' else Path(row['mesh']).parents[1] / 'input/그림1.png'
        if src.exists():
            shutil.copyfile(src, assets / f'{row["name"]}_input.png')
        metrics = (f'체적 {row["volume_cm3"]:.1f} cm³ · 독립 FEA 컴플라이언스 '
                   f'{row["compliance_J"]:.6f} J · BC 포함률 {row["bc_containment"]}')
        dist = ', '.join(f'{key}: {value:.3f}' for key, value in row['shape_distance_1_minus_iou'].items() if key != row['name'])
        input_img = f'<img src="assets/{name}_input.png" alt="input">' if src.exists() else ''
        cards.append(f'<article><h2>{name}</h2><p>{metrics}</p><p>형상 거리: {html.escape(dist)}</p>'
                     f'<div class="images">{input_img}<img src="assets/{name}_top.png" alt="final top">'
                     f'<img src="assets/{name}_iso.png" alt="final 3D">'
                     f'<img src="assets/{name}_xminus.png" alt="final side x-">'
                     f'<img src="assets/{name}_xplus.png" alt="final side x+"></div>'
                     f'<p><a href="/{Path(row["mesh"]).relative_to(ROOT)}">최종 OBJ</a></p></article>')
    niche_lines = ''.join('<li>니치 ' + str(i+1) + ': ' + html.escape(', '.join(n['members'])) +
                          ' / Pareto elite: ' + html.escape(', '.join(n['pareto_elites'])) + '</li>'
                          for i, n in enumerate(niches))
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Angular bracket FEA-on QD pilot</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;background:#f3f5f7;color:#1c2a33}}
article,.intro{{background:#fff;padding:1rem;margin:1rem 0;border-radius:12px}}.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px}}
img{{width:100%;object-fit:contain;background:#fff;border:1px solid #ddd}}a{{word-break:break-all}}</style>
<h1>FEA_ON 각진 브래킷 이미지 QD 파일럿</h1><div class="intro"><p>동일 BC·6 mm collar·생성 레시피에서 텍스트로 다른 이미지를 만들고 dense/sparse 모두 FEA_ON으로 생성했다. Side view는 각 후보의 dense 메시에서 렌더했으며 독립 생성 뷰가 아니다. 최종 boolean·remesh 후 동일한 −Z 1000 N 독립 FEA로 평가했다.</p>
<p>형상 니치는 고정 카메라 top 메시 투영의 1−IoU 0.10 기준. 품질은 체적과 컴플라이언스 동시 최소화로 비교한다. 이 파일럿은 소수 표본이므로 QD 성능 우위의 통계적 증거가 아니다.</p>
<p>관찰: 최종 메시의 외곽·개구부 배치는 변하지만, 입력 이미지의 세밀한 각진 리브와 모든 내부 개구부가 그대로 재현되지는 않는다. 따라서 여기서 측정한 다양성은 최종 메시의 거시적 형상 다양성이다.</p>
<ul>{niche_lines}</ul><p>누적 QD-HV: {html.escape(str(cumulative_hv))} (기준점 350 cm³, 0.006 J; 파일럿 비교용)</p>
<p><a href="/{(OUT / 'archive.json').relative_to(ROOT)}">아카이브 JSON</a></p></div>{''.join(cards)}</html>'''
    (report / 'index.html').write_text(page)
    print(json.dumps({'report': str(report / 'index.html'), 'accepted': len(accepted), 'niches': niches}, indent=2))


if __name__ == '__main__':
    main()
