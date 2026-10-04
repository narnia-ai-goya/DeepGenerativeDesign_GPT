#!/usr/bin/env python3
"""Show the executed chair QD feedback loop and its missing archive return edge."""
from __future__ import annotations

import html
import json
from pathlib import Path

from make_chair_domain import ROOT

BASE=ROOT/'experiments/chair/sofa_style_2026-09-28'
TEXT=BASE/'text_latent_qd_2026-10-03'
ARCHIVE=BASE/'tapered_examples_2026-10-03/qd_round_01'
OUT=BASE/'qd_loop_only_2026-10-03'
FEEDBACK=BASE/'archive_feedback_round_02_2026-10-03'


def main()->None:
    OUT.mkdir(parents=True,exist_ok=True)
    bank=json.loads((TEXT/'bank.json').read_text())
    obs=json.loads((TEXT/'observations.json').read_text())
    arc=json.loads((ARCHIVE/'archive.json').read_text())
    rounds=[json.loads((TEXT/f'selection_round_{i:02d}.json').read_text()) for i in (0,1)]
    assert len(bank['candidates'])==24 and len(obs)==4
    assert [r['observations_available'] for r in rounds]==[0,2]
    assert arc['total_unique_evaluations']==9 and arc['occupied_cells']==3
    prior=[r for r in arc['candidates'] if r['round']==0]
    new=[r for r in arc['candidates'] if r['round']==1]
    feedback=json.loads((FEEDBACK/'result.json').read_text()) if (FEEDBACK/'result.json').exists() else None
    assert len(prior)==4 and len(new)==5
    summary={
        'text_bank':len(bank['candidates']),
        'text_niches':len(bank['center_ids']),
        'completed_selection_rounds':len(rounds),
        'selected_and_observed_images':len(obs),
        'image_gate_pass':sum(r['image_gate'] for r in obs),
        'unique_3d_candidates':arc['total_unique_evaluations'],
        'prior_fea_evaluations':len(prior),
        'new_tapered_candidates':len(new),
        'new_admitted_elites':sum(r['archive_eligible'] for r in new),
        'occupied_3d_cells':arc['occupied_cells'],
        'possible_3d_cells':arc['archive_dims'][0]*arc['archive_dims'][1],
        'archive_feedback_used_by_selector':False,
        'latest_archive_feedback_round': (2 if feedback else None),
        'latest_archive_feedback_used_by_selector':bool(feedback),
        'latest_archive_occupied_cells':(feedback['occupied_after'] if feedback else None),
        'latest_archive_feedback_result':(str(FEEDBACK/'result.json') if feedback else None),
        'image_feedback_used_by_selector':True,
        'archive_path':str(ARCHIVE/'archive.json'),
    }
    (OUT/'status.json').write_text(json.dumps(summary,indent=2)+'\n')
    selection_rows=''.join(
        '<tr><td>'+str(r['round'])+'</td><td>'+str(r['observations_available'])+'</td><td>'+
        html.escape(', '.join(v['method']+': '+v['id'] for v in r['selected']))+'</td></tr>'
        for r in rounds)
    candidate_rows=''.join(
        '<tr><td>'+html.escape(r['id'])+'</td><td>'+str(r['round'])+'</td><td>'+str(r['cell'])+'</td><td>'+
        ('admitted' if r['archive_eligible'] else html.escape(', '.join(r['reasons'])))+'</td></tr>'
        for r in arc['candidates'])
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Chair QD loop: early rounds and archive feedback</title>
<style>
body{{font:17px/1.5 system-ui,sans-serif;background:#f1f5f6;color:#19313b;max-width:1320px;margin:28px auto;padding:0 22px}}
h1{{font-size:2.2rem;margin-bottom:8px}}article{{background:white;border:1px solid #d3e0e3;border-radius:13px;padding:20px;margin:18px 0}}
.callout{{background:#eaf5f1;border-left:5px solid #27866a;padding:15px}}.warn{{background:#fff4e6;border-left:5px solid #b67434;padding:15px}}
.flow{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;align-items:stretch;margin:20px 0}}
.step{{background:#eef6f7;border:1px solid #c6dcdf;border-radius:10px;padding:14px;min-height:128px}}
.step b{{display:block;font-size:1.08rem}}.step small{{display:block;color:#5d7380;margin-top:7px}}
.feedback{{border:2px dashed #b67434;border-radius:10px;padding:13px;color:#73481e;background:#fffaf4}}
.stats{{display:flex;gap:12px;flex-wrap:wrap}}.stat{{padding:10px 16px;background:#eef6f7;border-radius:9px}}.stat strong{{font-size:1.5rem}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #dce5e7;padding:8px;text-align:left;vertical-align:top}}
a{{color:#1b6b80}}code{{font-size:.88em}}@media(max-width:800px){{.flow{{grid-template-columns:1fr 1fr}}}}
</style>
<h1>의자 QD 루프: 초기 단계와 아카이브 피드백</h1>
<p class="callout"><b>초기 2라운드에는 이미지 피드백 탐색 + 별도 3D QD 아카이브 평가가 실행됐다.</b>
이미지 검사 결과는 다음 텍스트 선택에 반영됐지만, 당시 3D 아카이브는 선택기에 입력되지 않았다.</p>
<p class="callout"><b>후속 round 2에서는 3D 아카이브의 점유 셀과 실패 원인을 다음 문구 선택에 사용했다.</b>
아카이브 인식 후보 1개와 동일 예산 무작위 후보 1개를 새 이미지·3D로 평가했다.
엄격한 게이트 기준 점유율은 {feedback['occupied_before'] if feedback else 3}/9 →
{feedback['occupied_after'] if feedback else '미평가'}/9이다.
<a href="../archive_feedback_round_02_2026-10-03/index.html">새 실험 결과</a></p>
<article><h2>초기 2라운드 실행 경로</h2><div class="flow">
<div class="step"><b>① 형상 문구 선택</b><small>24개 문구 · frozen CLIP · 9개 텍스트 niche<br>희소도 + novelty + 기존 이미지 통과율</small></div>
<div class="step"><b>② 이미지 생성</b><small>텍스트 + 같은 의자 reference<br>2라운드 × QD/random 각 1장</small></div>
<div class="step"><b>③ 저비용 이미지 검사</b><small>시점·실루엣·기능 영역<br>4/4 이미지 gate 통과</small></div>
<div class="step"><b>④ 3D 생성·구조 검사</b><small>Direct3D-S2 → BC/영역/연결성<br>통과한 기존 후보에 두 하중 FEM proxy</small></div>
<div class="step"><b>⑤ 실현 형상 아카이브</b><small>side opening × back taper 3×3<br>셀 안에서 질량·compliance Pareto</small></div>
</div>
<p class="callout"><b>연결됨:</b> ③ 이미지 결과 → ① 다음 문구 선택. 0라운드에는 관측 0건,
1라운드 선택에는 관측 2건을 사용했다.</p>
<p class="feedback"><b>초기 단계에서 빠진 연결:</b> ⑤ 3D archive의 빈 셀, elite, 질량·강성, 실패 원인
→ ① 다음 문구 선택. 후속 round 2에서 빈 셀과 실패 원인을 사용하는 선택기를 시험했다.</p></article>
<article><h2>초기 2라운드와 별도 tapered 확장 수치</h2><div class="stats">
<div class="stat"><strong>2</strong><br>선택 라운드</div><div class="stat"><strong>4</strong><br>이미지+3D 기존 후보</div>
<div class="stat"><strong>9</strong><br>고유 3D 후보 합계</div><div class="stat"><strong>3/9</strong><br>점유 형상 셀</div>
<div class="stat"><strong>0/5</strong><br>새 tapered 후보 입장</div></div>
<p>새 tapered 후보 5개는 archive 상태로 선택된 다음 라운드가 아니라 별도 이미지·seed 확장 실험이다.
주로 BC 접촉 또는 5% 수선량 조건을 넘겨 탈락했다. 기존 후보 4개에만 공통 35 mm FEM proxy가 있다.
이는 원본 OBJ별 독립 FEA가 아니다.</p></article>
<article><h2>라운드별 문구 선택</h2><table><tr><th>Round</th><th>선택 시 관측</th><th>후보</th></tr>{selection_rows}</table></article>
<article><h2>3D archive 입장 기록</h2><p><a href="../tapered_examples_2026-10-03/qd_round_01/index.html">기존 archive 페이지</a> ·
<a href="../tapered_examples_2026-10-03/qd_round_01/figure/index.html">전체 프레임워크</a> ·
<a href="status.json">요약 JSON</a></p><table><tr><th>후보</th><th>Round</th><th>3D cell</th><th>입장 판정</th></tr>{candidate_rows}</table></article>
<article><h2>다음에 검증할 고리</h2><p>후속 round 2는 기존 아카이브의 빈 <em>실현 3D 형상</em> 셀과 실패 원인을 사용해 후보를 골랐다.
다만 엄격한 형상 게이트를 통과한 새 후보가 없어 아카이브 갱신 후 <em>다시 선택하는</em> 반복은 아직 실행되지 않았다.
연결성 처리 규칙을 사전에 고정하고, 같은 예산의 무작위 대조군과 여러 라운드를 반복해야 한다.</p>
<p>실험 입력과 결과: <code>{TEXT/'observations.json'}</code><br><code>{ARCHIVE/'archive.json'}</code></p></article>
</html>'''
    (OUT/'index.html').write_text(page)
    print(OUT/'index.html')


if __name__=='__main__':main()
