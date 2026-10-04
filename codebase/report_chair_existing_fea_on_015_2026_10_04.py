"""Browser report for the image-conditioned 15 mm chair FEA-on repeat."""
from __future__ import annotations

import html
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from run_chair_existing_fea_on_015_2026_10_04 import OUT


LABELS = {'off': '기존 FEA OFF', 'old_on': '기존 FEA ON · 35 mm',
          'new_dense_on': '새 Dense 단계 최종 · FEA ON 15 mm',
          'new_sparse_on': '새 Sparse 단계 최종 · Dense→Sparse FEA ON 15 mm'}


def main():
    rows = json.loads((OUT / 'evaluation/summary.json').read_text())
    by_name = {row['name']: row for row in rows}
    off_occ = np.load(OUT / 'evaluation/off/realized_occupancy.npz')['occupied']
    on_occ = np.load(OUT / 'evaluation/new_sparse_on/realized_occupancy.npz')['occupied']
    final_iou = float(np.count_nonzero(off_occ & on_occ) /
                      np.count_nonzero(off_occ | on_occ))
    final_delta = 100*(by_name['new_sparse_on']['compliance_15mm'] /
                       by_name['off']['compliance_15mm']-1)
    direct_dir = OUT / 'evaluation/new_sparse_on/direct_full_mesh_15mm'
    direct_geometry = json.loads((direct_dir / 'geometry_summary.json').read_text())
    direct_fea = json.loads((direct_dir / 'dc_info.json').read_text())
    direct_rows = []
    for name in ('off', 'old_on', 'new_sparse_on'):
        folder = OUT / 'evaluation' / name / 'direct_full_mesh_15mm'
        geom = json.loads((folder / 'geometry_summary.json').read_text())
        fea = json.loads((folder / 'dc_info.json').read_text())
        direct_rows.append(f'<tr><td>{html.escape(LABELS[name])}</td>'
                           f'<td>{geom["solid_volume_liters"]:.3f}</td>'
                           f'<td>{fea["compliance"]/1e6:.3f}</td></tr>')
    tile_w, tile_h = 730, 430
    montage = Image.new('RGB', (tile_w * len(rows), tile_h * 2 + 60), '#f6f8fa')
    draw = ImageDraw.Draw(montage)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    for i, row in enumerate(rows):
        name = row['name']
        draw.text((i*tile_w+12, 6), name, fill='#203b4b', font=font)
        draw.text((i*tile_w+12, 36),
                  f"{row['mass_liters']:.2f} L | C={row['compliance_15mm']/1e6:.1f} M",
                  fill='#455c69', font=font)
        case = OUT / 'evaluation' / name
        montage.paste(Image.open(case / 'raw_preview.png').convert('RGB'),
                      (i*tile_w, 60))
        montage.paste(Image.open(case / 'evaluated_preview.png').convert('RGB'),
                      (i*tile_w, 60+tile_h))
    montage.save(OUT / 'comparison.png')
    cards = []
    for row in rows:
        name = row['name']
        case = f'evaluation/{name}'
        cards.append(f'''<article><h2>{html.escape(LABELS[name])}</h2>
<p>실제 생성 메시 정면·측면</p><img src="{case}/raw_preview.png" alt="원본 {name}">
<p>64³ voxel 재구성 + BC·envelope 적용 평가 형상</p><img src="{case}/evaluated_preview.png" alt="평가 {name}">
<p>체적 <b>{row['mass_liters']:.3f} L</b> · 동시 하중 C <b>{row['compliance_15mm']/1e6:.3f}×10⁶</b></p>
<p>6-연결 성분 {row['components_6conn']}개 · BC/보호영역 보정률 {row['repair_fraction']:.1%}</p>
<p><a href="{case}/aligned_main.obj">물리 좌표 원본 OBJ</a> ·
<a href="{case}/combined_voxel.obj">평가 형상 OBJ</a> ·
<a href="{case}/fea.log">15 mm FEA 로그</a></p></article>''')
    table = ''.join(f'<tr><td>{html.escape(LABELS[r["name"]])}</td><td>{r["mass_liters"]:.3f}</td>'
                    f'<td>{r["compliance_15mm"]/1e6:.3f}</td>'
                    f'<td>{r["repair_fraction"]:.1%}</td></tr>' for r in rows)
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Chair generator FEA ON · 15 mm</title>
<style>body{{margin:0;background:#edf2f5;color:#213847;font:16px/1.5 system-ui}}main{{max-width:1450px;margin:auto;padding:24px}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}article{{background:white;border-radius:12px;padding:16px}}img{{width:100%;height:auto;background:#f8fafb}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{border:1px solid #cddae0;padding:8px;text-align:right}}td:first-child,th:first-child{{text-align:left}}a{{color:#166487}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}</style>
<main><h1>기존 이미지 조건부 의자 생성기 · FEA ON</h1>
<p>같은 정면 이미지, 시드 42, envelope, 네 발 고정과 좌면 800 N −Z·등받이 200 N +Y 동시 하중을 사용했습니다. 새 실행은 Dense와 Sparse 모두 FEA ON이며 생성 과정의 FEA 메쉬는 15 mm입니다. 독립 평가는 아래 네 형상 모두 동일한 15 mm 메쉬, SIMP p=2로 계산했습니다.</p>
<p><b>결과:</b> 새 Sparse 최종 결과는 OFF 대비 compliance가 {final_delta:+.2f}%로 개선되지 않았습니다. 두 평가 형상의 voxel IoU는 {final_iou:.3f}입니다. 새 Dense는 compliance가 낮지만 체적이 31.06 L로 증가했고, Sparse 후에는 약 25.41 L로 돌아왔습니다. 따라서 이번 실행은 15 mm FEA를 실제로 적용한 검증이지만 최종 성능 개선의 증거는 아닙니다.</p>
<p>새 Dense의 FEA 가중치는 1e-7(기존 ON 1e-8), Sparse는 8e-5이며, 생성 FEA 21회·4회가 각각 기록됐습니다. 평가용 BC·envelope 보정으로 원본 형상 차이가 줄어드는 점은 아래 두 줄의 렌더에서 확인할 수 있습니다.</p>
<p><b>그림 해석:</b> 위쪽 줄은 실제 mesh.obj입니다. 새 Sparse 최종 메시에는 543,912 vertices가 있고, Sparse 폴더의 mesh_dense.obj(10,504 vertices)는 전달된 Dense 중간 산출물입니다. 아래쪽 줄은 구조 평가를 위해 모든 결과를 64³ 격자에 재구성한 형상입니다. 명칭의 15 mm는 FEA 사면체 메쉬 설정이며 생성 메시의 삼각형 크기가 아닙니다.</p>
<p><b>원본 sparse OBJ 직접 해석:</b> 별도로 보호 영역·BC union·64³ voxel 재구성을 하지 않고 원본 전체 OBJ를 FEM 셀 중심에 샘플링하면 체적 {direct_geometry['solid_volume_liters']:.3f} L, compliance {direct_fea['compliance']/1e6:.3f}×10⁶입니다. 이 값은 위 표의 평가 형상({by_name['new_sparse_on']['mass_liters']:.3f} L, {by_name['new_sparse_on']['compliance_15mm']/1e6:.3f}×10⁶)과 형상 및 밀도 전사 방식이 다릅니다. 원본은 fix 영역을 완전히 채우지 않으며 void에도 E_min=1e-3의 ersatz 강성이 남아 있으므로 낮은 compliance만으로 원본을 더 좋은 구조라고 판단할 수 없습니다. <a href="evaluation/new_sparse_on/direct_full_mesh_15mm/dc_info.json">원본 FEA 수치</a> · <a href="evaluation/new_sparse_on/direct_full_mesh_15mm/geometry_summary.json">원본 형상 수치</a></p>
<table><tr><th>원본 OBJ 직접 해석</th><th>FEM 내부 체적 L</th><th>동시 하중 C ×10⁶</th></tr>{''.join(direct_rows)}</table>
<p>원본 OBJ 기준 새 ON의 compliance는 OFF보다 3.08% 높고 체적은 16.23% 낮습니다. 위의 BC·envelope 평가 형상 기준 새 ON의 compliance는 OFF보다 0.42% 높습니다. 고정영역 점유율과 체적이 다르므로 compliance 단독으로 강도 향상을 판정하지 않습니다.</p>
<p><b>체적과 겉보기 두께:</b> 원본 OBJ 체적은 Dense 새 ON 35.70 L에서 Sparse 새 ON 18.65 L로 줄었습니다. Sparse OFF와 비교해도 22.31 → 18.65 L입니다. 64³ 격자에서 줄어든 재료는 주로 좌면·팔걸이 높이(0.45–0.65 m)의 안쪽이며, 새 결과의 표면적은 1.44 m²로 OFF의 1.27 m²보다 커서 더 복잡하고 두꺼워 보일 수 있습니다.</p>
<p><img src="evaluation/volume_difference/volume_difference.png" alt="Dense와 Sparse 체적, 높이별 차이 및 좌면 top view"><a href="evaluation/volume_difference/summary.json">체적 비교 JSON</a></p>
<table><tr><th>설정</th><th>평가 체적 L</th><th>동시 하중 C ×10⁶</th><th>보정률</th></tr>{table}</table>
<p><img src="comparison.png" alt="FEA ON/OFF 비교: 위는 원본, 아래는 평가 형상"></p>
<p>성능 비교는 평가용 BC·envelope 적용 형상 기준입니다. 원본 메시의 시각 품질과 평가 형상 간 차이는 각 카드에서 함께 확인할 수 있습니다.</p>
<div class="grid">{''.join(cards)}</div>
<p><a href="evaluation/summary.json">평가 JSON</a> · <a href="dense/generation.log">Dense 생성 로그</a> · <a href="sparse/generation.log">Sparse 생성 로그</a></p></main></html>'''
    (OUT / 'index.html').write_text(page)
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
