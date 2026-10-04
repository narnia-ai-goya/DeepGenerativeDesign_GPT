#!/usr/bin/env python3
"""Summarize the chair QD pilot with cumulative archive metrics and caveats."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from make_chair_domain import ROOT

OUT=ROOT/'experiments/chair/sofa_style_2026-09-28/single_view_qd_2026-10-03'
REFERENCE_MASS=35.0
REFERENCE_QUALITY=1.5


def hv_2d(items: list[dict]) -> float:
    points=sorted((p['mass_liters'],p['quality_worst_ratio']) for p in items
                  if p['mass_liters']<REFERENCE_MASS and
                  p['quality_worst_ratio']<REFERENCE_QUALITY)
    if not points:return 0.
    area=0.;best=float('inf')
    for i,(mass,quality) in enumerate(points):
        best=min(best,quality)
        next_mass=points[i+1][0] if i+1<len(points) else REFERENCE_MASS
        area+=max(0.,next_mass-mass)*max(0.,REFERENCE_QUALITY-best)
    return area/(REFERENCE_MASS*REFERENCE_QUALITY)


def pareto(items: list[dict]) -> list[dict]:
    return [a for a in items if not any(
        b is not a and b['mass_liters']<=a['mass_liters'] and
        b['quality_worst_ratio']<=a['quality_worst_ratio'] and
        (b['mass_liters']<a['mass_liters'] or
         b['quality_worst_ratio']<a['quality_worst_ratio']) for b in items)]


def main() -> None:
    geometry=json.loads((OUT/'geometry_archive.json').read_text())
    fea={r['id']:r for r in json.loads((OUT/'fea_results.json').read_text())}
    refined_rows=json.loads((OUT/'fea_refinement_035.json').read_text())
    refined={(r['id'],r['mode']):r for r in refined_rows}
    assert len(refined)==14, 'Expected both refined load cases for all seven geometry-valid candidates'
    seat_ref=refined['seed_42','seat']['compliance_proxy']
    back_ref=refined['seed_42','back']['compliance_proxy']
    coarse_seat_ref=fea['seed_42']['seat']['compliance_proxy']
    coarse_back_ref=fea['seed_42']['back']['compliance_proxy']
    rows=[]
    for g in geometry['rows']:
        case=fea.get(g['id'])
        seat=refined.get((g['id'],'seat'))
        back=refined.get((g['id'],'back'))
        valid=bool(g['geometry_gate'] and case and case['seat']['valid'] and case['back']['valid']
                   and seat and back and seat['exit_code']==0 and back['exit_code']==0
                   and seat['fixed_nodes']>0 and back['fixed_nodes']>0
                   and seat['load_nodes']>0 and back['load_nodes']>0)
        row={**g,'fea_valid':valid}
        if valid:
            row.update(mass_liters=case['mass_liters'],
                       compliance_seat=seat['compliance_proxy'],
                       compliance_back=back['compliance_proxy'],
                       quality_worst_ratio=max(seat['compliance_proxy']/seat_ref,
                                               back['compliance_proxy']/back_ref),
                       quality_worst_ratio_coarse=max(
                           case['seat']['compliance_proxy']/coarse_seat_ref,
                           case['back']['compliance_proxy']/coarse_back_ref))
        rows.append(row)
    archive:dict[tuple[int,int],list[dict]]={}
    history=[]
    for index,row in enumerate(rows,1):
        if row['fea_valid']:
            key=tuple(row['realized_cell'])
            archive.setdefault(key,[]).append(row)
        total_hv=sum(hv_2d(pareto(v)) for v in archive.values())
        history.append({'attempt':index,'id':row['id'],
                        'coverage':len(archive),'qd_hv_cumulative':total_hv})
    elites={','.join(map(str,key)):[p['id'] for p in pareto(items)]
            for key,items in sorted(archive.items())}
    final={'evaluations':len(rows),'validated_candidates':sum(r['fea_valid'] for r in rows),
           'descriptor_names':list(geometry['descriptor_ranges']),
           'descriptor_ranges':geometry['descriptor_ranges'],
           'archive_dims':geometry['archive_dims'],
           'filled_cells':len(archive),'coverage_fraction':len(archive)/9,
           'pareto_elites_by_cell':elites,'reference_mass_liters':REFERENCE_MASS,
           'reference_quality_worst_ratio':REFERENCE_QUALITY,
           'qd_hv_cumulative':history[-1]['qd_hv_cumulative'],
           'history':history,'baseline_compliance_proxy':{'seat':seat_ref,'back':back_ref},
           'tetra_mesh_sizes_m':{'primary':0.035,'sensitivity':0.05},
           'rows':rows,'method_notes':[
               'Single-view pretrained Direct3D-S2, no model training.',
               'Designer-controlled image variants form the second proposal round; no learned BO or posterior acquisition was used.',
               'Functional interfaces are clamped at 64^3 and the cost of added/removed voxels is counted; raw OBJ is not automatically BC-compliant.',
               'Primary FEM: 64^3 density field and 0.035 m common tetra mesh (30,138 tets), E0=1, seat 800 N and backrest 200 N. 0.05 m (10,787 tets) is sensitivity analysis; neither is converged or calibrated physical stiffness.',
               'guided_45 changes from better-than-baseline compliance on 0.05 m tets to worse-than-baseline on 0.035 m tets; its apparent strength improvement is not robust.',
               'Descriptor bin ranges and 5% repair budget were chosen for this pilot, not preregistered for a paper comparison.',
           ]}
    (OUT/'summary.json').write_text(json.dumps(final,indent=2)+'\n')

    fig,ax=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    matrix=np.full((3,3),np.nan)
    for (x,y),items in archive.items():matrix[2-y,x]=min(p['quality_worst_ratio'] for p in items)
    im=ax[0].imshow(matrix,cmap='viridis_r',vmin=.6,vmax=1.3,aspect='auto')
    for (x,y),items in archive.items():
        best=min(items,key=lambda p:p['quality_worst_ratio'])
        ax[0].text(x,2-y,f"{best['id'].replace('image_','').replace('_seed42','')}\n{best['quality_worst_ratio']:.2f}",ha='center',va='center',fontsize=7,color='white')
    ax[0].set_xticks(range(3),['low','mid','high']);ax[0].set_yticks(range(3),['broad rim','mid','tapered'])
    ax[0].set_xlabel('side opening');ax[0].set_title(f'Archive: {len(archive)}/9 filled cells')
    fig.colorbar(im,ax=ax[0],label='worst compliance / baseline (lower is better)')
    x=np.arange(1,len(history)+1)
    ax[1].step(x,[h['coverage'] for h in history],where='post',color='#2563a6',lw=2)
    ax[1].set_ylim(0,9.5);ax[1].set_xlabel('evaluations');ax[1].set_ylabel('cumulative filled cells')
    ax[1].grid(alpha=.25);ax[1].set_title('Cumulative QD coverage')
    ax[2].step(x,[h['qd_hv_cumulative'] for h in history],where='post',color='#c06a20',lw=2)
    ax[2].set_xlabel('evaluations');ax[2].set_ylabel('cumulative QD-HV (normalized)')
    ax[2].grid(alpha=.25);ax[2].set_title('Cumulative cellwise mass–performance HV')
    fig.savefig(OUT/'archive_dashboard.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,5.2),layout='constrained')
    for row in rows:
        if not row['fea_valid']:continue
        x,y=row['mass_liters'],row['quality_worst_ratio']
        ax.scatter(x,y,s=100,edgecolor='#17212b')
        ax.annotate(row['id'].replace('image_',''),(x,y),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.axhline(1,color='#888',ls='--',lw=1)
    ax.set_xlabel('realized mass (L)');ax.set_ylabel('worst normalized compliance (lower is better)')
    ax.set_title('Two-load mass–performance trade-off, 64³ proxy');ax.grid(alpha=.25)
    fig.savefig(OUT/'mass_performance.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,5.2),layout='constrained')
    for row in rows:
        if not row['fea_valid']:continue
        ax.scatter(row['quality_worst_ratio_coarse'],row['quality_worst_ratio'],s=100,edgecolor='#17212b')
        ax.annotate(row['id'].replace('image_',''),
                    (row['quality_worst_ratio_coarse'],row['quality_worst_ratio']),
                    xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.axhline(1,color='#888',ls='--',lw=1);ax.axvline(1,color='#888',ls='--',lw=1)
    ax.set_xlabel('0.05 m tetra mesh: worst compliance / baseline')
    ax.set_ylabel('0.035 m tetra mesh: worst compliance / baseline')
    ax.set_title('FEA resolution sensitivity, same 64³ density field');ax.grid(alpha=.25)
    fig.savefig(OUT/'fea_resolution_sensitivity.png',dpi=180);plt.close(fig)

    cards=[]
    for row in rows:
        name=row['id'];case=OUT/name
        status='검증 셀 '+str(row['realized_cell']) if row['fea_valid'] else '사양 미통과'
        performance=(f"{row['mass_liters']:.2f} L · 최악 하중 비율 {row['quality_worst_ratio']:.3f}"
                     if row['fea_valid'] else 'FEA archive 제외')
        source=f"<img src='{name}/input_image.png' alt='input'>" if (case/'input_image.png').exists() else "<img src='../open_arm/input/v00_front_lo.png' alt='input'>"
        cards.append(f"<article class='tile'><h3>{html.escape(name)}</h3>{source}<p>{html.escape(status)}<br>{html.escape(performance)}<br>보정 {100*row['repair_fraction']:.1f}% · 원본 발 최저 겹침 {100*min(row['foot_coverage']):.0f}%</p><a href='{name}/aligned_main.obj'>원본 OBJ</a> · <a href='{name}/realized_voxel.obj'>평가용 voxel OBJ</a></article>" if name!='seed_42' else
                     f"<article class='tile'><h3>{name}</h3><img src='../open_arm/input/v00_front_lo.png' alt='input'><p>{html.escape(status)}<br>{html.escape(performance)}<br>보정 {100*row['repair_fraction']:.1f}%</p><a href='../minimal_direct3ds2_2026-10-02/aligned_sparse_main.obj'>원본 OBJ</a> · <a href='{name}/realized_voxel.obj'>평가용 voxel OBJ</a></article>")
    table=''.join(f"<tr><td>{html.escape(r['id'])}</td><td>{'✓' if r['fea_valid'] else '—'}</td><td>{str(r['realized_cell'])}</td><td>{100*r['repair_fraction']:.1f}%</td><td>{f'{r['mass_liters']:.2f}' if r['fea_valid'] else '—'}</td><td>{f'{r['quality_worst_ratio']:.3f}' if r['fea_valid'] else '—'}</td></tr>" for r in rows)
    (OUT/'index.html').write_text(f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>단일 시점 의자 QD 파일럿</title>
<style>body{{font:16px/1.5 system-ui;max-width:1500px;margin:0 auto;padding:22px;background:#eef2f5;color:#17212b}}.card,.tile{{background:white;border:1px solid #d8e0e6;border-radius:12px;padding:16px;margin:12px 0}}img{{max-width:100%;height:auto}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}}.tile img{{width:100%;height:230px;object-fit:contain}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d8e0e6;padding:7px;text-align:left}}a{{color:#155e99}}.note{{background:#fff4db;border-left:4px solid #d5972a;padding:12px}}</style>
<h1>단일 시점 의자 · designer-steered QD 파일럿</h1><p>정면 한 장 D3D-S2를 기준으로 seed·사양 guidance·디자이너 이미지 변형을 순차적으로 탐색했다. 학습이나 다중 시점 토큰 결합은 사용하지 않았다.</p>
<section class="card"><h2>결과: {len(archive)}/9 형상 셀</h2><p>9개 후보를 생성·평가했고 7개가 64³ 기능영역 보존, 5% 이하 보정, 연결성 및 좌판·등받이 공통 FEM proxy 검사를 통과했다. 형상 다양성은 측면 열린 공간과 등받이 상단 폭 비율로 정의했다. 셀 내부에서는 질량과 최악 하중 compliance의 Pareto 후보를 보존했다.</p><p><a href="viewer.html">3D 메시 뷰어</a> · <a href="summary.json">전체 JSON</a> · <a href="../single_view_spec_2026-10-03/index.html">사양 페이지</a></p><img src="archive_dashboard.png" alt="QD archive, cumulative coverage and hypervolume"><img src="mass_performance.png" alt="mass performance scatter"><img src="fea_resolution_sensitivity.png" alt="FEA resolution sensitivity"></section>
<section class="card"><h2>이미지 변화가 3D로 전달되는가?</h2><img src="image_to_mesh_diversity.png" alt="original angular wing image and 3D"><p>각진 의자는 형상 차이가 전달됐지만 보정된 좌판·발 영역을 3.7% 채워야 했다. 곡선형 wing 의자는 원본 3D가 좌판 하중과 만나지 않아 보정량 12.8%로 archive에서 제외됐다.</p></section>
<section class="card"><h2>평가 대상과 원본 메시의 차이</h2><p class="note"><b>중요:</b> 구조 수치와 QD 셀은 오른쪽의 64³ 기능영역 보존 형상에서 계산했다. 왼쪽 매끈한 원본 OBJ 자체가 BC를 통과했다는 주장이 아니다. 오른쪽 voxel OBJ는 연결·하중 위치 검증용 저해상도 형상이며 최종 제조 메시가 아니다.</p><img src="source_vs_realized_mesh.png" alt="smooth source mesh versus 64 voxel realization"></section>
<section class="card"><h2>후보별 측정</h2><table><tr><th>후보</th><th>검증</th><th>형상 셀</th><th>보정량</th><th>질량 L</th><th>최악 compliance / 기준</th></tr>{table}</table><p>좌판 800 N −Z, 등받이 200 N +Y, 동일 30,138-tet 도메인과 64³ 재료 격자를 주 평가에 사용했다. 0.05 m 격자로도 민감도를 확인했다. guided_45는 거친 격자에서는 기준보다 좋아 보였지만 조밀한 격자에서는 나빠져 강성 개선으로 해석할 수 없다. E0=1의 상대 FEM proxy이며 물성 보정·격자 수렴·독립 고해상도 메시 FEA는 수행하지 않았다. QD-HV는 셀마다 질량 35 L, 최악 정규화 compliance 1.5를 고정 참조점으로 한 누적 면적이다. 이 파일럿은 BO-QD와 random의 우월성을 검증하는 대조 실험이 아니다.</p></section>
<section class="card"><h2>후보 메시</h2><div class="grid">{''.join(cards)}</div></section></html>''')
    print(json.dumps({'evaluations':len(rows),'valid':sum(r['fea_valid'] for r in rows),
                      'filled_cells':len(archive),'qd_hv':history[-1]['qd_hv_cumulative'],
                      'elites':elites},indent=2))


if __name__=='__main__':main()
