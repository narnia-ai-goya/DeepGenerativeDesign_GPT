"""Summarize the controlled chair Dense/Sparse dual-load FEA study."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from run_chair_dual_load_fea_2026_10_04 import OUT


def main() -> None:
    dense = json.loads((OUT / 'dense_evaluation/round_01/evaluation.json').read_text())['rows']
    sparse = json.loads((OUT / 'sparse_evaluation/round_01/evaluation.json').read_text())['rows']
    rows = dense + sparse
    labels = {
        'dense_control': 'Dense: FEA OFF',
        'dense_dual': 'Dense: seat + back FEA',
        'sparse_d0_s0': 'Dense OFF → Sparse OFF',
        'sparse_d0_s1': 'Dense OFF → Sparse dual FEA',
        'sparse_d1_s0': 'Dense dual FEA → Sparse OFF',
        'sparse_d1_s1': 'Dense dual FEA → Sparse dual FEA',
    }
    cards = []
    summary = []
    for row in rows:
        name = row['id']
        seat = row['fea_035']['seat']['compliance_proxy'] / 1e6
        back = row['fea_035']['back']['compliance_proxy'] / 1e6
        mass = row['mass_liters']
        worst = row['worst_compliance_ratio']
        cards.append(f'''<article><h3>{html.escape(labels[name])}</h3>
<img src="{Path(row['preview']).relative_to(OUT)}" alt="{name} 3D preview">
<dl><dt>Mass</dt><dd>{mass:.3f} L</dd><dt>Seat 800 N −Z</dt><dd>{seat:.3f} M</dd>
<dt>Back 200 N +Y</dt><dd>{back:.3f} M</dd>
<dt>Worst C / reference</dt><dd>{worst:.4f}</dd></dl>
<p><a href="{Path(row['generated_mesh']).relative_to(OUT)}">Raw OBJ</a> ·
<a href="{Path(row['combined_mesh']).relative_to(OUT)}">Evaluated OBJ</a></p></article>''')
        summary.append({'id': name, 'label': labels[name], 'mass_liters': mass,
                        'seat_compliance': row['fea_035']['seat']['compliance_proxy'],
                        'back_compliance': row['fea_035']['back']['compliance_proxy'],
                        'worst_compliance_ratio': worst, 'strict_eligible': row['strict_eligible']})
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair dual-load Dense/Sparse FEA</title>
<style>body{font:16px/1.5 system-ui,sans-serif;background:#eef1f4;color:#172329;margin:0 auto;max-width:1450px;padding:30px}
h1{margin-bottom:0}.lead{max-width:1000px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}
article{background:white;border-radius:14px;padding:16px;box-shadow:0 2px 14px #182a3b18}
article img{width:100%%;height:290px;object-fit:contain;background:#eef2f5}
dl{display:grid;grid-template-columns:1fr auto;gap:4px 14px;margin:12px 0}dd{margin:0;font-variant-numeric:tabular-nums;font-weight:700}
dt{color:#52646c}a{color:#0868a5}section{margin-top:28px}code{font-size:14px}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>의자: Dense·Sparse 두 하중 FEA 비교</h1>
<p class="lead">동일 이미지와 seed 42로 Dense FEA OFF/ON × Sparse FEA OFF/ON을 비교했다.
ON은 좌면 800 N (−Z)과 등받이 200 N (+Y)를 매 호출마다 모두 해석한다. 독립 평가에서도 두 하중을 동일한 35 mm FEM 격자로 다시 계산했다.</p>
<p><a href="sparse_results.html">Sparse 결과 4개만 크게 보기</a></p>
<section id="sparse"><h2>Sparse 최종 결과 4개</h2><div class="grid">%s</div></section>
<section id="dense"><h2>Dense 결과 2개</h2><div class="grid">%s</div></section>
<section><h2>판독</h2><p>Dense dual FEA는 최악 하중 컴플라이언스 비율을 1.0768 → 1.0653으로 낮췄지만 질량은 30.815 → 30.993 L로 증가했다.
Sparse에서는 같은 Dense 출발점의 FEA OFF/ON 쌍이 35 mm FEM에서 좌면·등받이 컴플라이언스가 각각 동일했다.
Raw mesh는 서로 다르지만 64³ 평가 점유 격자에서는 각 쌍이 5 또는 8 voxel만 다르며, 이 차이가 FEM 요소 밀도까지 전달되지 않았다.
동일 GPU 재실행에서도 Dense OFF → Sparse ON의 원본 메시와 Sparse OFF 메시 간 최근접 표면 거리 p95는 0.94 mm였지만,
독립 FEA의 좌면·등받이 컴플라이언스는 정확히 같았다. 서로 다른 GPU의 Sparse ON 반복 간 p95는 0.40 mm였다.
따라서 현재 Sparse loss는 실행되지만 최종 구조 성능 개선은 입증되지 않았다.</p>
<p>평가에는 공통 BC, 확장 envelope, 보호 영역 병합을 적용했다. 아래 그림은 최종 평가 형상의 3D 음영 렌더다.</p></section>
<p>같은 GPU의 재실행 결과: <a href="sparse_repeat_evaluation/round_01/mesh_cases/sparse_d0_s1_gpu6_repeat/preview.png">Sparse ON 반복 3D 그림</a> ·
<a href="sparse_repeat_evaluation/round_01/evaluation.json">독립 FEA 수치</a></p>
<p><a href="summary.json">기계 판독용 결과 JSON</a></p></html>''' % (''.join(cards[2:]), ''.join(cards[:2]))
    (OUT / 'index.html').write_text(page)
    sparse_cards = []
    for row in sparse:
        name = row['id']
        label = labels[name]
        png = Path(row['preview'])
        raw_png = OUT / f'{name}_raw_preview.png'
        raw_embedded = base64.b64encode(raw_png.read_bytes()).decode('ascii')
        evaluated_embedded = base64.b64encode(png.read_bytes()).decode('ascii')
        seat = row['fea_035']['seat']['compliance_proxy'] / 1e6
        back = row['fea_035']['back']['compliance_proxy'] / 1e6
        sparse_cards.append(f'''<article class="case" id="{name}">
<h3>{html.escape(label)}</h3>
<p class="stage">원본 Sparse 최종 메시 · 정면 / 측면</p>
<img src="data:image/png;base64,{raw_embedded}" alt="{html.escape(label)} 원본 Sparse 정면·측면 3D 렌더">
<div class="metrics"><span>질량 <b>{row['mass_liters']:.3f} L</b></span>
<span>좌면 C <b>{seat:.3f} M</b></span><span>등받이 C <b>{back:.3f} M</b></span>
<span>최악 C/ref <b>{row['worst_compliance_ratio']:.4f}</b></span></div>
<details><summary>BC·envelope 반영 FEA 평가 형상 보기</summary>
<img class="evaluated" src="data:image/png;base64,{evaluated_embedded}"
alt="{html.escape(label)} FEA 평가용 복셀 형상"></details>
<p class="files"><a href="{Path(row['generated_mesh']).relative_to(OUT)}">원본 Sparse OBJ</a> ·
<a href="{Path(row['combined_mesh']).relative_to(OUT)}">평가 형상 OBJ</a> ·
<a href="{raw_png.relative_to(OUT)}">원본 Sparse PNG</a></p></article>''')
    sparse_report = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sparse FEA OFF/ON 비교 보고서</title>
<style>
*{{box-sizing:border-box}}body{{font:16px/1.55 system-ui,sans-serif;background:#edf1f4;color:#1a2a33;margin:0}}
main{{max-width:1500px;margin:0 auto;padding:28px}}h1{{font-size:31px;margin:0 0 7px}}h2{{font-size:22px;margin:28px 0 10px}}
.lead{{max-width:1120px;margin:0 0 14px;color:#425765}}.note{{padding:13px 17px;background:#e2edf4;border-left:5px solid #2473a4;border-radius:6px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}
.case{{background:#fff;border:1px solid #d4e0e6;border-radius:12px;padding:16px;box-shadow:0 3px 14px #132b3712}}
.case h3{{margin:0;font-size:19px}}.stage{{margin:2px 0 10px;color:#5d707a;font-size:13px}}
.case img{{display:block;width:100%;height:auto;min-height:280px;object-fit:contain;background:#f5f7f8;border:1px solid #e2e8eb}}
.case img.evaluated{{min-height:0;margin-top:8px}}details{{font-size:14px;color:#37586a;cursor:pointer}}
.metrics{{display:grid;grid-template-columns:repeat(2,1fr);gap:6px 14px;margin:12px 0;font-variant-numeric:tabular-nums}}
.metrics span{{display:flex;justify-content:space-between;gap:6px;padding:4px 0;border-bottom:1px solid #edf0f2}}
.files{{margin:8px 0 0;font-size:14px}}a{{color:#146393}}
table{{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums}}
th,td{{padding:9px 12px;border:1px solid #d7e0e5;text-align:right}}th:first-child,td:first-child{{text-align:left}}
th{{background:#dfeaf0}}p.small{{font-size:14px;color:#52636c}}
@media(max-width:850px){{.grid{{grid-template-columns:1fr}}main{{padding:14px}}h1{{font-size:25px}}}}
</style></head><body><main>
<h1>Sparse 최종 결과: FEA OFF / ON</h1>
<p class="lead">같은 의자 이미지와 seed 42에서 Dense FEA OFF/ON × Sparse FEA OFF/ON을 비교했습니다.
Sparse FEA ON은 좌면 800 N (−Z)과 등받이 200 N (+Y)를 모두 계산했습니다.
아래 네 큰 그림은 <b>원본 Sparse 최종 메시</b>의 정면·측면 3D 렌더입니다.
각 그림 아래에서 BC·envelope을 합친 FEA 평가용 형상을 따로 펼쳐볼 수 있습니다.</p>
<div class="note"><b>결과:</b> Sparse FEA ON/OFF는 원본 메시에서는 조금 다르지만,
동일한 35 mm 독립 FEA에서 좌면·등받이 컴플라이언스는 같습니다.
Dense FEA ON 여부에 따른 차이는 Sparse 이후에도 남습니다.</div>
<h2>최종 형상 4개</h2><div class="grid">{''.join(sparse_cards)}</div>
<h2>수치 비교</h2>
<table><thead><tr><th>Dense / Sparse</th><th>질량 (L)</th><th>좌면 C (×10⁶)</th>
<th>등받이 C (×10⁶)</th><th>최악 C/ref</th></tr></thead><tbody>
{''.join(f"<tr><td>{html.escape(labels[r['id']])}</td><td>{r['mass_liters']:.3f}</td><td>{r['fea_035']['seat']['compliance_proxy']/1e6:.3f}</td><td>{r['fea_035']['back']['compliance_proxy']/1e6:.3f}</td><td>{r['worst_compliance_ratio']:.4f}</td></tr>" for r in sparse)}
</tbody></table>
<h2>해석</h2>
<p>각 Sparse ON 실행에서 두 하중 FEA와 기울기 갱신이 4회 기록됐습니다.
같은 GPU에서 반복해도 OFF/ON 원본 메시의 최근접 표면 거리 p95는 약 0.94 mm였습니다.
그러나 64³ 평가 격자 차이는 3~8 voxel이고, 35 mm FEM의 구조 성능 수치는 변하지 않았습니다.
따라서 현재 Sparse FEA는 작동하지만 최종 구조 성능 개선은 확인되지 않았습니다.</p>
<p class="small"><a href="index.html">Dense 결과 포함 전체 보고서</a> ·
<a href="sparse_evaluation/round_01/evaluation.json">독립 FEA 원본 결과 JSON</a> ·
<a href="sparse_repeat_evaluation/round_01/evaluation.json">동일 GPU 반복 검증 JSON</a></p>
</main></body></html>'''
    (OUT / 'sparse_results.html').write_text(sparse_report)
    (OUT / 'sparse_raw_report.html').write_text(sparse_report)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
