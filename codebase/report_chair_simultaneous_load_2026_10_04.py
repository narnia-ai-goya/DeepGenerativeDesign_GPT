"""Static report of one-solve, simultaneous seat/back chair FEA."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from run_chair_dual_load_fea_2026_10_04 import PARENT


OUT = PARENT / 'simultaneous_load_dense_sparse_2026-10-04'
LABELS = {
    'dense_control': 'Dense FEA OFF',
    'dense_dual': 'Dense 복합 하중 FEA ON',
    'sparse_d0_s0': 'Dense OFF → Sparse OFF',
    'sparse_d0_s1': 'Dense OFF → Sparse 복합 하중 ON',
    'sparse_d1_s0': 'Dense 복합 하중 ON → Sparse OFF',
    'sparse_d1_s1': 'Dense 복합 하중 ON → Sparse 복합 하중 ON',
}


def card(row: dict) -> str:
    name = row['id']
    img = base64.b64encode(Path(row['raw_preview']).read_bytes()).decode()
    evaluated = base64.b64encode(Path(row['evaluated_preview']).read_bytes()).decode()
    return f'''<article><h3>{html.escape(LABELS[name])}</h3>
<p class="stage">원본 {html.escape(row['stage'].capitalize())} 최종 메시 · 정면 / 측면</p>
<img src="data:image/png;base64,{img}" alt="{html.escape(LABELS[name])} 원본 메시">
<div class="metrics"><span>질량</span><b>{row['mass_liters']:.3f} L</b>
<span>좌면 800 N + 등받이 200 N 동시 C</span><b>{row['combined_compliance']/1e6:.3f} M</b></div>
<details><summary>BC·envelope 반영 평가 형상</summary>
<img src="data:image/png;base64,{evaluated}" alt="평가용 형상"></details>
<p><a href="{Path(row['raw_obj']).relative_to(OUT)}">원본 OBJ</a> ·
<a href="{Path(row['fea_log']).relative_to(OUT)}">복합 하중 FEA 로그</a></p></article>'''


def main() -> None:
    dense = json.loads((OUT / 'dense_combined_evaluation.json').read_text())
    sparse = json.loads((OUT / 'sparse_combined_evaluation.json').read_text())
    table = ''.join(
        f'<tr><td>{html.escape(LABELS[r["id"]])}</td><td>{r["mass_liters"]:.3f}</td>'
        f'<td>{r["combined_compliance"]/1e6:.3f}</td>'
        f'<td>{r["seat_only_compliance"]/1e6:.3f}</td>'
        f'<td>{r["back_only_compliance"]/1e6:.3f}</td></tr>'
        for r in dense + sparse
    )
    page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>의자 Dense/Sparse 동시 복합 하중 FEA 보고서</title>
<style>
*{{box-sizing:border-box}}body{{font:16px/1.55 system-ui,sans-serif;background:#edf1f4;color:#1a2933;margin:0}}
main{{max-width:1520px;margin:0 auto;padding:24px}}h1{{font-size:30px;margin:0 0 6px}}h2{{margin:28px 0 12px}}
.lead{{max-width:1050px}}.callout{{padding:13px 18px;background:#e0edf5;border-left:5px solid #2773a4;border-radius:6px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}
article{{background:white;border:1px solid #d5e0e6;border-radius:12px;padding:15px;box-shadow:0 3px 14px #152d3810}}
article h3{{margin:0;font-size:19px}}.stage{{font-size:13px;color:#63737d;margin:2px 0 9px}}
article img{{width:100%;display:block;height:auto;min-height:260px;object-fit:contain;background:#f7f9fa;border:1px solid #e3e9ec}}
.metrics{{display:grid;grid-template-columns:1fr auto;gap:5px 15px;margin:11px 0;font-variant-numeric:tabular-nums}}
details{{font-size:14px;color:#355b70;cursor:pointer}}details img{{margin-top:8px;min-height:0}}
a{{color:#13618e}}table{{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums}}
th,td{{padding:9px 11px;border:1px solid #d5dfe4;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#dce9ef}}
.small{{font-size:14px;color:#51636d}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}main{{padding:12px}}}}
</style></head><body><main>
<h1>의자: Dense·Sparse 동시 복합 하중 FEA</h1>
<p class="lead">좌면 800 N (−Z)과 등받이 200 N (+Y)을 <b>하나의 FEA 해석에서 동시에</b> 적용했습니다.
두 힘을 합친 변위장을 풀고 C = (f₁ + f₂)ᵀK⁻¹(f₁ + f₂)를 Dense와 Sparse 지도의 물리 손실로 사용했습니다.</p>
<div class="callout">동일 이미지·시드 42로 FEA OFF/ON을 비교합니다. 좌면 단독·등받이 단독 수치는 아래 표에서
진단용으로만 제공하며, ON 생성 손실과 주요 성능 지표는 <b>동시 복합 하중 C</b>입니다.
SIMP penalty는 생성과 독립 평가 모두 <b>p = 2</b>입니다.</div>
<h2>Sparse 최종 메시 4개</h2><div class="grid">{''.join(map(card,sparse))}</div>
<h2>Dense 메시 2개</h2><div class="grid">{''.join(map(card,dense))}</div>
<h2>독립 FEA 수치</h2>
<table><thead><tr><th>실행</th><th>질량 L</th><th>동시 복합 C ×10⁶</th>
<th>좌면 단독 C ×10⁶</th><th>등받이 단독 C ×10⁶</th></tr></thead><tbody>{table}</tbody></table>
<p><b>관찰:</b> Dense FEA ON은 동시 복합 C를 약 0.12% 낮췄지만 질량이 증가했습니다.
같은 Dense 출발점에서 Sparse FEA ON/OFF의 최종 복합 C는 35 mm 독립 FEA에서 동일합니다.
따라서 현재 설정에서 Sparse FEA의 성능 개선은 입증되지 않았습니다.</p>
<p class="small">모든 수치는 동일 BC·envelope, 35 mm tetrahedral mesh와 최종 평가 형상에서 계산했습니다.
복합 하중 열은 두 단독 수치의 합이 아니라 한 번의 변위 해석 결과입니다.</p>
<p><a href="dense_combined_evaluation.json">Dense 결과 JSON</a> ·
<a href="sparse_combined_evaluation.json">Sparse 결과 JSON</a></p>
</main></body></html>'''
    (OUT / 'index.html').write_text(page)
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
