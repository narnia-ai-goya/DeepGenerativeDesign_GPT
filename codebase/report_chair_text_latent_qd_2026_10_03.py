#!/usr/bin/env python3
"""Visual and audit report for matched latent-QD versus random chair pilot."""
from __future__ import annotations

import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from chair_text_latent_qd_pilot_2026_10_03 import OUT


def main()->None:
    bank=json.loads((OUT/'bank.json').read_text())
    rows=json.loads((OUT/'mesh_evaluations.json').read_text())
    methods=('latent_qd','random')
    colors={'latent_qd':'#176b8b','random':'#bd6a2c'}
    summary={
        'prompt_bank_size':len(bank['candidates']),'image_evaluations':len(rows),
        'generator':'built-in GPT image edit, one output per selected prompt',
        'image_to_3d':'same one-view Direct3D-S2, seed 42, VANILLA=1, dense+sparse',
        'text_embedding':bank['encoder'],'method':'fixed-library latent-niche novelty heuristic; not BO or learned BOP-Elites',
        'by_method':{},'rows':rows,
        'key_limitations':['Two rounds and two image/3D evaluations per method cannot establish superiority.',
                           'All four 2D gates passed, so observed feasibility feedback did not differentiate round-1 acquisition.',
                           '3D FEA uses 64^3 repaired voxel density and 0.035 m common tetra mesh, not independent FEA of raw OBJ.',
                           'The image envelope check is a front projection. Depth envelope is checked only after 3D conversion.',
                           'The pre-existing 3D descriptor ranges reject one flared shape despite physical feasibility; archive range and physical feasibility are reported separately.']}
    def pooled_hv(items:list[dict])->float:
        # Diagnostic mass-performance hypervolume, not QD-HV: shape cells are
        # ignored here.  Shared fixed reference mass 35 L, compliance 1.5.
        points=sorted((r['realized_mass_liters'],r['worst_compliance_ratio_to_baseline'])
                      for r in items if r['fea_valid'])
        best=float('inf');area=0.
        for index,(mass,quality) in enumerate(points):
            best=min(best,quality)
            next_mass=points[index+1][0] if index+1<len(points) else 35.
            area+=max(0.,next_mass-mass)*max(0.,1.5-best)
        return area/(35*1.5)
    for method in methods:
        selected=[r for r in rows if r['method']==method]
        summary['by_method'][method]={
            'n':len(selected),'distinct_text_cells':len({r['text_cell'] for r in selected}),
            'distinct_image_cells':len({tuple(r['image_cell']) for r in selected if r['image_gate']}),
            'voxel_repair_gate_pass':sum(r['physical_gate'] for r in selected),
            'distinct_in_range_3d_cells':len({tuple(r['realized_3d_cell']) for r in selected if r['archive_range_gate']}),
            'out_of_range_repaired_valid':sum(r['physical_gate'] and not r['archive_range_gate'] for r in selected),
            'fea_valid':sum(r['fea_valid'] for r in selected),
            'pooled_mass_performance_hv_normalized':pooled_hv(selected),
        }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    entries=bank['candidates']
    z=np.asarray([r['pca2'] for r in entries])
    axes[0].scatter(z[:,0],z[:,1],c='#d6dde2',s=45)
    by_id={r['id']:r for r in entries}
    for row in rows:
        pt=by_id[row['id']]['pca2']
        axes[0].scatter(*pt,c=colors[row['method']],s=120,edgecolor='#17212b')
        axes[0].annotate(row['id'],pt,xytext=(3,5),textcoords='offset points',fontsize=7)
    axes[0].set_title('Text latent (PCA display)');axes[0].set_xlabel('PCA 1');axes[0].set_ylabel('PCA 2')
    for row in rows:
        axes[1].scatter(row['central_open_fraction'],row['upper_span_ratio'],
                        c=colors[row['method']],s=120,edgecolor='#17212b')
        axes[1].annotate(row['id'],(row['central_open_fraction'],row['upper_span_ratio']),
                         xytext=(3,5),textcoords='offset points',fontsize=7)
    axes[1].set_title('Measured input-image phenotype');axes[1].set_xlabel('central opening')
    axes[1].set_ylabel('upper / mid silhouette span')
    for row in rows:
        d=row['realized_3d_descriptor']
        axes[2].scatter(d['side_open_fraction'],d['backrest_taper_ratio'],
                        c=colors[row['method']],s=120,edgecolor='#17212b',
                        marker='o' if row['archive_range_gate'] else 'X')
        axes[2].annotate(row['id'],(d['side_open_fraction'],d['backrest_taper_ratio']),
                         xytext=(3,5),textcoords='offset points',fontsize=7)
    axes[2].axhline(1.05,color='#b55',ls='--',lw=1,label='prior 3D archive maximum')
    axes[2].set_title('Realized 3D phenotype');axes[2].set_xlabel('side opening')
    axes[2].set_ylabel('backrest taper');axes[2].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.25)
    fig.savefig(OUT/'text_image_3d_transport.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,5.1),layout='constrained')
    for row in rows:
        if not row['fea_valid']:continue
        ax.scatter(row['realized_mass_liters'],row['worst_compliance_ratio_to_baseline'],
                   color=colors[row['method']],s=115,edgecolor='#17212b',
                   marker='o' if row['archive_range_gate'] else 'X')
        ax.annotate(row['id'],(row['realized_mass_liters'],row['worst_compliance_ratio_to_baseline']),
                    xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.axhline(1,color='#888',ls='--')
    ax.set_xlabel('realized mass (L)');ax.set_ylabel('worst normalized compliance (lower better)')
    ax.set_title('Common two-load FEM proxy, all four physical candidates');ax.grid(alpha=.25)
    fig.savefig(OUT/'mass_fea.png',dpi=180);plt.close(fig)

    cards=[]
    for row in rows:
        name=html.escape(row['id'])
        quality=f"{row['worst_compliance_ratio_to_baseline']:.3f}" if row['fea_valid'] else 'not evaluated'
        status='in prior 3D archive range' if row['archive_range_gate'] else 'outside prior 3D descriptor range'
        cards.append(f'''<article><div class="pair"><img src="images/{name}.png" alt="input"><img src="mesh_cases/{name}/preview.png" alt="3D"></div><h3>{name}</h3><p>{row['method']} · round {row['round']} · text cell {row['text_cell']} → image {row['image_cell']} → 3D {row['realized_3d_cell']}</p><p>2D envelope outside {row['projected_envelope_outside_fraction']:.1%} · raw 3D envelope outside {row['source_outside_envelope_fraction']:.1%}<br>raw seat/back BC overlap {row['source_seat_bc_coverage']:.0%}/{row['source_back_bc_coverage']:.0%} · repair {row['repair_fraction']:.1%}<br>mass {row['realized_mass_liters']:.2f} L · worst FEM ratio {quality} · {status}</p><a href="mesh_cases/{name}/aligned_main.obj">original 3D OBJ</a> · <a href="mesh_cases/{name}/evaluation.json">measures</a></article>''')
    q=summary['by_method']['latent_qd'];r=summary['by_method']['random']
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Text latent QD chair pilot</title><style>body{{font:16px/1.5 system-ui;max-width:1450px;margin:0 auto;padding:24px;background:#f0f3f6;color:#17212b}}section,article{{background:white;border:1px solid #d6dfe6;border-radius:12px;padding:17px;margin:14px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:12px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}img{{max-width:100%}}.pair img{{width:100%;height:280px;object-fit:contain}}a{{color:#146399}}.note{{border-left:4px solid #b46b2c;background:#fff4df;padding:13px}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ccd8e0;text-align:left}}</style><h1>Text latent QD → image → 3D chair pilot</h1><p>24개 형상 문구를 고정하고 frozen CLIP 텍스트 임베딩(768D)의 9개 niche에서 QD가 프롬프트를 골랐다. 두 라운드에 걸쳐 QD 2장과 균등 무작위 2장을 동일한 GPT 이미지 편집 및 Direct3D-S2 설정으로 처리했다. GPT 이미지 모델의 내부 latent에는 접근하지 않았다.</p><section><h2>결과</h2><table><tr><th>방법</th><th>이미지 셀</th><th>64³ 보정 게이트</th><th>기존 범위 안의 3D 셀</th><th>범위 밖 보정 유효</th><th>질량–강성 HV</th></tr><tr><td>Text latent QD</td><td>{q['distinct_image_cells']}</td><td>{q['voxel_repair_gate_pass']}/{q['n']}</td><td>{q['distinct_in_range_3d_cells']}</td><td>{q['out_of_range_repaired_valid']}</td><td>{q['pooled_mass_performance_hv_normalized']:.3f}</td></tr><tr><td>Random</td><td>{r['distinct_image_cells']}</td><td>{r['voxel_repair_gate_pass']}/{r['n']}</td><td>{r['distinct_in_range_3d_cells']}</td><td>{r['out_of_range_repaired_valid']}</td><td>{r['pooled_mass_performance_hv_normalized']:.3f}</td></tr></table><p>이미지 단계는 QD가 두 셀, random이 한 셀이다. 그러나 기존 범위 안의 3D 셀은 QD 한 셀, random 두 셀이다. QD의 다른 후보는 64³ 보정 형상으로는 유효하지만 등받이 폭 지표 1.074가 이전 아카이브 최대 1.05를 넘어 범위 밖이다. 네 후보 모두 64³ 기능영역 보정량 5% 이하·단일 연결 성분을 통과했다. 단, 원본 메시의 좌판 하중 패치 점유는 모두 50%라 원본 자체의 BC 적합성이 확인된 것은 아니다. HV는 질량 35 L, 기준 대비 compliance 1.5의 공통 참조점을 쓰는 pooled Pareto 면적이며, 셀별 QD-HV가 아니다.</p><img src="text_image_3d_transport.png"><img src="mass_fea.png"><p><a href="summary.json">전체 결과 JSON</a> · <a href="bank.json">프롬프트 24개와 정확한 텍스트</a> · <a href="text_latent_bank.png">프롬프트 공간</a></p></section><p class="note"><b>envelope 처리:</b> 생성 이미지에서는 기준 메시와 물리 envelope의 정면 투영 범위를 비교했고 네 이미지 모두 범위 밖 검은 픽셀 0%였다. Direct3D-S2 생성 중 envelope 유도는 꺼져 있다. 생성 후 3D envelope 밖의 점유를 측정하고 clip+BC union 보정량에 포함했다. 정면 검사만으로 깊이 방향 envelope을 증명하지 못한다.</p><section><h2>이미지 → 3D 후보</h2><div class="grid">{''.join(cards)}</div></section><section><h2>연구 판단</h2><p><b>현재로서는 방법 우위의 증거가 아니다.</b> 2개/방법이라는 극소 표본에서 이미지 셀 수는 QD가 앞서지만, 기존 범위의 3D archive coverage는 random이 앞선다. 네 이미지가 모두 2D 게이트를 통과해서 이번 반복의 feasibility feedback은 선택에 차이를 주지 못했다. FEA는 최종 고해상도 OBJ가 아닌 공통 64³ 보정 형상의 상대값이며 격자 수렴을 입증하지 않았다.</p><p>학술 기여 후보는 QD 자체나 prompt evolution이 아니라, 디자이너 의도 latent → 관측된 이미지 형상 → BC/envelope 적합 3D → 질량·FEA 사이의 불일치를 모델링하여 고비용 3D 평가를 배분하는 방법이다. 이를 주장하려면 프롬프트 풀·셀 경계를 사전에 고정하고, 다수 랜덤 시드와 예산 일치 비교, prompt/image/3D descriptor transport, 독립 FEA와 사용자 선호 실험이 필요하다.</p><p>관련 선행연구: <a href="https://proceedings.mlr.press/v235/ding24h.html">QDHF (ICML 2024)</a> · <a href="https://arxiv.org/abs/2310.13032">QDAIF</a> · <a href="https://arxiv.org/abs/2307.09326">BOP-Elites</a> · <a href="https://arxiv.org/abs/2406.09143">Prompt evolution for 3D car design</a> · <a href="https://doi.org/10.1145/3670693">QD and topology optimization</a>.</p></section></html>'''
    (OUT/'index.html').write_text(page)
    print(json.dumps(summary['by_method'],indent=2))


if __name__=='__main__':main()
