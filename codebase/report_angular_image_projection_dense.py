#!/usr/bin/env python3
"""Evaluate FEA-on dense image-projection guidance across weights."""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from diagnose_angular_stage_fidelity import (CASES, EXP, N, OUT as STAGE_OUT,
                                            PIXELS_PER_M, align_mesh, bc_centers,
                                            render_top, summary_for_pair)
from run_semantic_qd_sparse_round import transform_matrix
from run_connectivity_qd_sampling import render_mesh


OUT = EXP / 'image_projection_guidance_2026-09-24'
WEIGHTS = (0.5, 2.0, 8.0)
CACHE_CONTROL = STAGE_OUT / 'dense_cache_reuse_ablation'


def main() -> None:
    stage_metrics = json.loads((STAGE_OUT / 'metrics.json').read_text())
    matrix = transform_matrix()
    results = {}
    cards = []
    for name in ('triangular_truss', 'staggered_chevron'):
        source = CASES[name]
        config = json.loads((source / 'config_dense_top.json').read_text())
        input_rgb = np.asarray(Image.open(STAGE_OUT / name / 'input_registered_affine.png').convert('RGB'))
        image_mask = input_rgb.min(axis=2) < 230
        yy, xx = np.indices((N, N))
        bc = np.zeros((N, N), bool)
        for i, (cx, cy) in enumerate(bc_centers(config)):
            radius = (10 if i < 4 else 8) * PIXELS_PER_M / 1000
            bc |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        case_rows = []
        reference = stage_metrics['cases'][name]['image_to_stage']['affine']['top_dense']
        case_rows.append({'weight': 0.0, 'status': 'baseline', **reference,
                          'mesh': str(source / 'dense_top/mesh.obj'),
                          'render': str(STAGE_OUT / name / 'top_dense_top.png')})
        for weight in WEIGHTS:
            case = OUT / name / f'w{weight:g}'
            mesh = case / 'generation/mesh_dense.obj'
            cache_path = case / 'dense_cache.npz'
            run_path = case / 'run.json'
            if not mesh.exists() or not cache_path.exists() or not run_path.exists() or json.loads(run_path.read_text()).get('exit_code') != 0:
                case_rows.append({'weight': weight, 'status': 'pending'})
                continue
            aligned = case / 'mesh_dense_physical.obj'
            align_mesh(mesh, aligned, matrix)
            rendering = case / 'dense_top.png'
            mask = render_top(aligned, rendering)
            metrics = summary_for_pair(image_mask, mask, ~bc, bc)
            cache = np.load(cache_path)
            case_rows.append({'weight': weight, 'status': 'complete', **metrics,
                              'n_active_dense_tokens': int(len(cache['latent_index'])),
                              'mesh': str(mesh), 'render': str(rendering)})
        results[name] = case_rows
        available = [row for row in case_rows if row['status'] != 'pending']
        sheet = Image.new('RGB', (N * (len(available) + 1), N + 45), 'white')
        draw = ImageDraw.Draw(sheet)
        images = [('input', STAGE_OUT / name / 'input_registered_affine.png')]
        images += [(f'w={row["weight"]:g}', Path(row['render'])) for row in available]
        for i, (label, path) in enumerate(images):
            sheet.paste(Image.open(path).convert('RGB'), (i * N, 45))
            draw.text((i * N + 12, 12), f'{name}: {label}', fill='black')
        sheet_path = OUT / name / 'dense_comparison.png'
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(sheet_path)
        table_rows = ''.join('<tr><td>' + f'{row["weight"]:g}' + '</td><td>' +
                             (f'{row["solid_iou_design_region"]:.3f}' if row['status'] != 'pending' else '—') +
                             '</td><td>' +
                             (f'{row["void_recall"]:.3f}' if row['status'] != 'pending' else '—') +
                             '</td><td>' +
                             (str(row.get('n_active_dense_tokens', '—'))) + '</td></tr>'
                             for row in case_rows)
        sparse_rows = []
        for label, sparse_case in (('fresh dense baseline', source),
                                   ('cached dense w=0', CACHE_CONTROL / name),
                                   ('cached dense guided w=8', OUT / name / 'w8/sparse_fea_on')):
            sparse_mesh = sparse_case / 'post/mesh_physical_aligned.obj'
            if not sparse_mesh.exists():
                continue
            sparse_top = sparse_case / 'post/sparse_top.png'
            sparse_mask = render_top(sparse_mesh, sparse_top)
            sparse_rows.append({'label': label, 'top': str(sparse_top),
                                **summary_for_pair(image_mask, sparse_mask, ~bc, bc)})
        results[name + '_sparse'] = sparse_rows
        sparse_table = ''.join(
            f'<tr><td>{html.escape(row["label"])}</td><td>{row["solid_iou_design_region"]:.3f}</td>'
            f'<td>{row["void_recall"]:.3f}</td><td>{row["enclosed_void_iou"]:.3f}</td></tr>'
            for row in sparse_rows)
        final_rows = []
        for label, final_case in (('fresh dense baseline', source),
                                  ('cached dense w=0', CACHE_CONTROL / name),
                                  ('cached dense guided w=8', OUT / name / 'w8/sparse_fea_on')):
            final_mesh = final_case / 'post/final.obj'
            final_metric = final_case / 'metrics.json'
            if not final_mesh.exists() or not final_metric.exists():
                continue
            top = final_case / 'post/final_top.png'
            iso = final_case / 'post/final_iso.png'
            final_mask = render_top(final_mesh, top)
            if not iso.exists():
                render_mesh(final_mesh, iso, f'{name} {label}')
            row = {'label': label, 'top': str(top), 'iso': str(iso),
                   **summary_for_pair(image_mask, final_mask, ~bc, bc),
                   **json.loads(final_metric.read_text())}
            final_rows.append(row)
        results[name + '_final'] = final_rows
        final_table = ''.join(
            f'<tr><td>{html.escape(row["label"])}</td>'
            f'<td>{row["solid_iou_design_region"]:.3f}</td><td>{row["void_recall"]:.3f}</td>'
            f'<td>{row["volume_cm3"]:.1f}</td><td>{row.get("compliance_J", float("nan")):.5f}</td>'
            f'<td>{row["geometry_valid"]}</td></tr>' for row in final_rows)
        final_images = ''.join(
            f'<figure><img src="/{Path(row["iso"]).relative_to(EXP.parent.parent.parent)}">'
            f'<figcaption>{html.escape(row["label"])} · <a href="/{Path(row["top"]).relative_to(EXP.parent.parent.parent)}">top view</a>'
            f' · <a href="/{Path(row["final_mesh"]).relative_to(EXP.parent.parent.parent)}">final OBJ</a></figcaption></figure>'
            for row in final_rows)
        cards.append(f'<section><h2>{html.escape(name)}</h2><table><tr><th>영상 손실 가중치</th>'
                     f'<th>solid IoU</th><th>개구부 recall</th><th>active voxel</th></tr>{table_rows}</table>'
                     f'<p><a href="/{sheet_path.relative_to(EXP.parent.parent.parent)}">비교 이미지</a></p>'
                     f'<img class="sheet" src="{name}/dense_comparison.png">'
                     f'<h3>Sparse 직후</h3><table><tr><th>설정</th><th>solid IoU</th><th>개구부 recall</th>'
                     f'<th>개구부 IoU</th></tr>{sparse_table}</table>'
                     f'<h3>최종 메쉬 (sparse FEA + BC Boolean + remesh)</h3>'
                     f'<table><tr><th>설정</th><th>solid IoU</th><th>개구부 recall</th><th>부피 cm³</th>'
                     f'<th>독립 FEA compliance J</th><th>유효 형상</th></tr>{final_table}</table>'
                     f'<div class="meshes">{final_images}</div></section>')
    (OUT / 'dense_metrics.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Angular image projection dense study</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1700px;margin:2rem auto;padding:0 1rem;background:#f2f4f6}}
section{{background:white;border-radius:12px;padding:1rem;margin:1rem 0}}table{{border-collapse:collapse}}th,td{{border:1px solid #cbd2d8;padding:7px 10px}}
.sheet{{width:100%;display:block}}.meshes{{display:flex;flex-wrap:wrap;gap:1rem}}figure{{margin:0;width:min(48%,600px)}}figure img{{width:100%}}a{{word-break:break-all}}</style>
<h1>FEA_ON dense 이미지 투영 손실 스터디</h1><section><p>동일한 이미지·seed·BC·FEA 설정에서 직접 top 투영 손실의 가중치만 변경했다. 최종 단계는 원래의 fresh multi-view dense와, 동일한 cache-reuse 연산을 적용한 w=0/w=8을 구분한다. 입력 영상의 6개 BC 마커 affine 정합에 약 11 px 오차가 있어 영상 일치도는 근사치다.</p>
<p><strong>결과:</strong> 두 이미지 모두 dense 개구부 보존이 개선되었다. matched cache-reuse 비교에서 chevron은 최종 이미지 일치도, 부피, compliance가 함께 개선되었고 truss는 이미지 일치도·부피와 compliance 사이에 절충이 생겼다. 두 사례·한 seed의 파일럿이므로 QD 성능 우위로 일반화할 수 없다.</p>
<p><a href="/{(OUT / 'dense_metrics.json').relative_to(EXP.parent.parent.parent)}">수치 JSON</a> · <a href="README.md">실험 설명</a></p></section>{''.join(cards)}</html>'''
    (OUT / 'index.html').write_text(page)
    print(OUT / 'dense_metrics.json')


if __name__ == '__main__':
    main()
