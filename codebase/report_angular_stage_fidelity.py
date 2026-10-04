#!/usr/bin/env python3
"""Publish a local stage-by-stage angular feature preservation report."""
from __future__ import annotations

import html
import json
from pathlib import Path

from run_connectivity_qd_sampling import ROOT


OUT = ROOT / 'experiments/bracket/angular_image_qd_loop_2026-09-24/stage_fidelity_2026-09-24'
STAGES = [('top_dense', 'Top-only dense'), ('multiview_dense', 'Multi-view dense'),
          ('sparse_aligned', 'Sparse'), ('final', 'Final')]


def main() -> None:
    metrics = json.loads((OUT / 'metrics.json').read_text())
    cards = []
    for name, data in metrics['cases'].items():
        affine = data['image_to_stage']['affine']
        tps = data['image_to_stage']['tps']
        pairwise = data['stage_to_stage']
        header = ''.join(f'<th>{label}</th>' for _, label in STAGES)
        solid = ''.join(f'<td>{affine[stage]["solid_iou_design_region"]:.3f}</td>' for stage, _ in STAGES)
        void = ''.join(f'<td>{affine[stage]["void_recall"]:.3f}</td>' for stage, _ in STAGES)
        tps_void = ''.join(f'<td>{tps[stage]["void_recall"]:.3f}</td>' for stage, _ in STAGES)
        images = [f'<figure><img src="{name}/input_registered_affine.png"><figcaption>BC 등록 입력</figcaption></figure>']
        images += [f'<figure><img src="{name}/{stage}_top.png"><figcaption>{label}</figcaption></figure>'
                   for stage, label in STAGES]
        transition = ' · '.join(f'{key.replace("_to_", " → ")}: '
                                f'solid {value["solid_iou_design_region"]:.3f}, '
                                f'void {value["enclosed_void_iou"]:.3f}'
                                for key, value in pairwise.items())
        widths = data['input_widths_affine']
        cards.append(f'''<section><h2>{html.escape(name)}</h2>
<p>BC 마커 affine 정합 RMS {data['registration']['affine']['marker_rmse_px']:.1f}px. 입력 중심선 폭 중앙값 {widths['median_mm']:.1f}mm;
1 dense voxel({widths['dense_pitch_mm']:.2f}mm)보다 얇은 중심선 {widths['fraction_centerline_under_one_dense_voxel']*100:.1f}%.</p>
<table><tr><th>입력 대비</th>{header}</tr><tr><td>solid IoU (BC 주변 제외)</td>{solid}</tr>
<tr><td>내부 개구부 recall (affine)</td>{void}</tr><tr><td>개구부 recall (TPS 민감도)</td>{tps_void}</tr></table>
<p>연속 단계 간 일치도: {html.escape(transition)}</p>
<div class="stages">{''.join(images)}</div></section>''')
    ablation = OUT / 'dense_cache_reuse_ablation/ablation_metrics.json'
    if ablation.exists():
        data = json.loads(ablation.read_text())
        ablation_rows = ''.join(
            f'<tr><td>{html.escape(name)}</td><td>{row["baseline_final_void_recall"]:.3f} → '
            f'{row["cache_reuse_final_void_recall"]:.3f}</td>'
            f'<td>{row["baseline_image_to_final_affine"]["solid_iou_design_region"]:.3f} → '
            f'{row["cache_reuse_image_to_final_affine"]["solid_iou_design_region"]:.3f}</td>'
            f'<td>{row["baseline_volume_cm3"]:.1f} → {row["cache_reuse_volume_cm3"]:.1f}</td>'
            f'<td>{row["baseline_compliance_J"]:.6f} → {row["cache_reuse_compliance_J"]:.6f}</td></tr>'
            for name, row in data['cases'].items())
        comparison_images = ''.join(
            f'<figure><img src="dense_cache_reuse_ablation/{name}/comparison.png">'
            f'<figcaption>{html.escape(name)}: input / top dense / 기존 final / cache 재사용 final</figcaption></figure>'
            for name in data['cases'])
        ablation_html = (f'<section><h2>Top-only dense cache 재사용 검증</h2><p>두 후보 모두 BC 포함·수밀성 검사를 통과했다. '
                         f'입력 형상 보존과 질량·강성은 서로 다른 방향으로 움직인다.</p>'
                         f'<table><tr><th>형상</th><th>최종 개구부 recall</th><th>solid IoU</th>'
                         f'<th>체적 cm³</th><th>컴플라이언스 J</th></tr>{ablation_rows}</table>'
                         f'<p><a href="/{ablation.relative_to(ROOT)}">전체 ablation JSON</a></p>'
                         f'<div>{comparison_images}</div></section>')
    else:
        ablation_html = '<section><h2>Top-only dense cache 재사용 검증</h2><p>진행 중</p></section>'
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Angular bracket stage fidelity</title>
<style>body{{font:16px system-ui,sans-serif;color:#1c2833;background:#f1f4f6;margin:2rem auto;max-width:1900px;padding:0 1rem}}
section{{background:white;border-radius:12px;padding:1.2rem;margin:1rem 0}}table{{border-collapse:collapse}}
th,td{{padding:7px 10px;border:1px solid #cbd2d8;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.stages{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}}
figure{{margin:0}}img{{width:100%;background:white;border:1px solid #ccc}}figcaption{{font-size:.85rem;color:#52606b}}
@media(max-width:1100px){{.stages{{overflow-x:auto;display:flex}}figure{{min-width:240px}}}}</style>
<h1>각진 브래킷 이미지 → 메시 단계별 형상 보존 진단</h1>
<section><p>동일 물리 좌표의 top 투영으로 입력·top-only dense·새 multi-view dense·sparse·final을 비교했다. 이미지의 빨강/초록 BC 표시 여섯 개를 실제 BC 중심에 등록했다. 평균 marker 정합 오차가 약 11px이므로, exact TPS 정합에서도 같은 경향인지 함께 표시했다.</p>
<p>핵심: 입력의 내부 개구부는 이미 top-only dense에서 일부 손실된다. 삼각 트러스·X-브레이스·비대칭 spine은 multi-view dense를 다시 만들 때 추가 손실이 크다. sparse→final은 대부분 유지되어 최종 boolean이 주원인은 아니다. Chevron은 multi-view에서 손실이 크지 않아 후보별 반응이 다르다.</p>
<p><a href="/{(OUT / 'metrics.json').relative_to(ROOT)}">전체 수치 JSON</a> · <a href="/{(OUT / 'all_stage_contact.png').relative_to(ROOT)}">전체 비교 이미지</a></p></section>
{ablation_html}{''.join(cards)}</html>'''
    (OUT / 'index.html').write_text(page)
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
