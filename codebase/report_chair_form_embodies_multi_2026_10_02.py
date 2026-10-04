#!/usr/bin/env python3
"""Compare registered chair FEM on/off geometry for three image concepts."""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from make_chair_domain import ROOT
from report_chair_prefea_stages import silhouette_scores
from validate_chair_bc_geometry import audit

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'form_embodies_multi_2026-10-02'
REG = BASE / 'registered_spec_2026-10-02'
BC = BASE / 'consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
INPUTS = {
    'open_arm': BASE / 'open_arm/input',
    'solid_side': BASE / 'solid_side/input',
    'diagonal_braced': ROOT / 'experiments/chair/image_concepts_2026-09-27/diagonal_braced/multiview_registered/input',
}
VIEWS = [('front', 15, 0, 'v00_front_lo'),
         ('right', 15, 90, 'v02_right_lo'), ('top', 85, 0, 'v_top')]


def physical_mesh(path, target, calibration):
    mesh = trimesh.load(path, force='mesh')
    source = np.asarray(calibration['source_center_m'])
    center = np.asarray(calibration['physical_center_m'])
    rotation = Rotation.from_euler('x', calibration['rotation_x_degrees'], degrees=True).as_matrix()
    mesh.vertices = (mesh.vertices - source) @ rotation.T * float(calibration['uniform_scale']) + center
    mesh.export(target)
    return mesh


def main():
    OUT.mkdir(exist_ok=True)
    cal = json.loads((REG / 'calibration.json').read_text())['native_to_physical']
    domain = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = domain.bounds.mean(axis=0)
    radius = float(np.linalg.norm(domain.extents)) * 1.5
    fit = float(domain.extents.max()) / 2
    perf_path = OUT / 'voxel_fea_800N.json'
    perf = {(x['case'], x['mode']): x for x in json.loads(perf_path.read_text())} if perf_path.exists() else {}
    rows = []
    cards = []
    for case in INPUTS:
        for stage in ('dense', 'sparse'):
            tile, label_height = 320, 30
            modes = (['fea_off', 'fea_on'] if stage == 'dense' or case == 'diagonal_braced'
                     else ['fea_off', 'dense_on_sparse_off', 'dense_off_sparse_on', 'fea_on'])
            sheet = Image.new('RGB', (tile * 3, (tile + label_height) * len(modes)), 'white')
            draw = ImageDraw.Draw(sheet)
            for mode_index, mode in enumerate(modes):
                directory = OUT / (case + '_' + (mode + '_aligned' if stage == 'dense' and mode == 'fea_on'
                                                 else mode if stage == 'sparse' and mode.startswith('dense_')
                                                 else mode + '_sparse' if stage == 'sparse' else mode))
                source = directory / 'generation' / ('mesh_dense.obj' if stage == 'dense' else 'mesh.obj')
                if not source.exists():
                    continue
                target = directory / ('physical_dense.obj' if stage == 'dense' else 'physical_sparse.obj')
                mesh = physical_mesh(source, target, cal)
                check = audit(target, BC)
                item = {'case': case, 'stage': stage, 'mode': mode,
                        'image': str(INPUTS[case] / 'v00_front_lo.png'),
                        'mesh': str(target), 'volume_liters': round(abs(mesh.volume) * 1000, 3),
                        'watertight': bool(mesh.is_watertight),
                        'components': len(mesh.split(only_watertight=False)),
                        'bc_pass': bool(check['bc_geometry_pass']),
                        'bc_coverage': {k: round(v['coverage'], 4) for k, v in check['regions'].items()},
                        'silhouette_iou': {}}
                if mode == 'fea_on' or mode == 'dense_off_sparse_on':
                    run = json.loads((directory / 'run.json').read_text())
                    item['inloop_fea_calls_logged'] = run.get('fea_success_log_count', run.get('fea_step_log_count'))
                    item['inloop_fea_failures'] = run.get('fea_failures')
                if stage == 'sparse' and (case, mode) in perf:
                    item['post_voxel_fea_compliance_proxy'] = perf[(case, mode)].get('compliance_proxy')
                    item['occupied_domain_voxels'] = perf[(case, mode)]['occupied_domain_voxels']
                for col, (view, elev, azim, stem) in enumerate(VIEWS):
                    eye, up = camera_from_elev_azim(center, radius, elev, azim)
                    rgb = render_lit(mesh, eye, center, up, size=tile,
                                     fit_extent=fit, margin=1.15, color=(.54, .58, .62))
                    image = Image.fromarray(rgb).convert('RGB')
                    sheet.paste(image, (col * tile, mode_index * (tile + label_height) + label_height))
                    draw.text((col * tile + 8, mode_index * (tile + label_height) + 6),
                              f'{mode} / {view}', fill='#20252b')
                    reference = Image.open(INPUTS[case] / f'{stem}.png').convert('RGB').resize((tile, tile))
                    item['silhouette_iou'][view] = round(silhouette_scores(image, reference)['iou'], 4)
                rows.append(item)
            figure = OUT / f'{case}_{stage}_comparison.png'
            sheet.save(figure)
            cards.append(f'<article><h2>{html.escape(case)} · {stage}</h2><img src="{figure.name}"></article>')
    (OUT / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    trs = []
    for row in rows:
        cov = row['bc_coverage']
        feet = min(v for k, v in cov.items() if k.startswith('foot'))
        url = '/' + str(Path(row['mesh']).relative_to(ROOT))
        comp = row.get('post_voxel_fea_compliance_proxy')
        comp_text = f'{comp/1e8:.3f}×10⁸' if comp is not None else '—'
        trs.append(f'<tr><td>{row["case"]}</td><td>{row["stage"]}</td><td>{row["mode"]}</td>'
                   f'<td>{row["volume_liters"]:.1f}</td><td>{row["components"]}</td>'
                   f'<td>{cov["seat_load"]:.1%}/{cov["backrest_load"]:.1%}/{feet:.1%}</td>'
                   f'<td>{row["silhouette_iou"]["front"]:.3f}/{row["silhouette_iou"]["right"]:.3f}</td>'
                   f'<td>{comp_text}</td>'
                   f'<td><a href="{url}">OBJ</a></td></tr>')
    inputs = ''.join('<div><h3>' + html.escape(case) + '</h3><img style="width:220px" src="/' +
                     str((folder / 'v00_front_lo.png').relative_to(ROOT)) + '"></div>'
                     for case, folder in INPUTS.items())
    ratios = {}
    for case in INPUTS:
        a, b = perf.get((case, 'fea_off')), perf.get((case, 'fea_on'))
        if a and b and a.get('compliance_proxy') and b.get('compliance_proxy'):
            ratios[case] = round(100 * (b['compliance_proxy'] / a['compliance_proxy'] - 1), 2)
    (OUT / 'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair FEM multi</title>
<style>body{font:16px system-ui;max-width:1150px;margin:2rem auto;background:#f3f5f7;color:#17212b}article{background:white;border-radius:12px;padding:1.2rem;margin:1rem 0}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.4rem}</style>
<article><h1>세 의자 · Form Embodies Mechanics FEM pilot</h1><p>동일 이미지·seed·BC, 좌표 등록된 64³ domain. 좌판 수직 하중만 FEA 적용. 등받이는 형상 BC로 유지했다. sparse 최종 형상을 동일 격자에 재표본화해 800 N FEM proxy로 다시 평가했다. 이 값은 완성 메시의 독립 tetra FEM 결과나 물리적 J 단위가 아니며, 같은 문제의 ON/OFF 비율로만 읽어야 한다.</p><p><a href="metrics.json">전체 수치</a> · <a href="voxel_fea_800N.json">FEM proxy 원자료</a> · <a href="REPORT.md">판정</a></p></article>
<article><h2>입력 이미지</h2><div style="display:flex;gap:1rem;flex-wrap:wrap">''' + inputs + '''</div></article>
<article><h2>형상·BC 비교</h2><table><tr><th>의자</th><th>단계</th><th>조건</th><th>체적 L</th><th>성분</th><th>좌석/등받이/발 최소</th><th>정면/측면 IoU</th><th>FEM proxy C</th><th>메시</th></tr>''' + ''.join(trs) + '</table></article>' + ''.join(cards) + '</html>')
    (OUT / 'compliance_ratios.json').write_text(json.dumps(ratios, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
