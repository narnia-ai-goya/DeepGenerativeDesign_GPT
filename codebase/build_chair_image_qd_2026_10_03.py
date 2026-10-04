#!/usr/bin/env python3
"""Pilot image-first chair QD: text intentions -> measured front-view phenotypes.

All measurements are from pixels after foreground-bbox normalization.  The
functional gate is only projected seat/foot overlap with the reference; it
cannot establish four distinct feet, 3D BC compliance, or structural quality.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from make_chair_domain import ROOT

BASE = ROOT/'experiments/chair/sofa_style_2026-09-28'
OUT = BASE/'single_view_image_qd_2026-10-03'
N=512
DESCRIPTORS={
    'central_open_fraction': [0.2,0.65],
    'upper_span_ratio': [0.5,1.1],
}
INTENT={
    'baseline': {'text':'Rounded upholstered-style metal backrest, open arms, four fixed feet.',
                 'provenance':'descriptive label for existing image; exact generation prompt unavailable',
                 'target':{'central_open_fraction':[.35,.46],'upper_span_ratio':[.70,.83]}},
    'angular': {'text':'Angular, polygonal backrest and squared armrests; preserve seat and four feet.',
                'provenance':'descriptive label for existing image; exact generation prompt unavailable',
                'target':{'central_open_fraction':[.34,.45],'upper_span_ratio':[.70,.85]}},
    'wing': {'text':'Broad, flowing wing-style backrest and curved armrests; preserve seat and four feet.',
             'provenance':'descriptive label for existing image; exact generation prompt unavailable',
             'target':{'central_open_fraction':[.38,.52],'upper_span_ratio':[.86,1.05]}},
    'tall_rect': {'text':'Make a clearly narrower, taller, straight-sided rectangular backrest (upper width visibly narrower than lower width), with thin straight geometric arm supports and open side voids. Keep the four floor-contact legs and seat.',
                  'provenance':'exact shape clause from built-in image-generation prompt',
                  'target':{'central_open_fraction':[.20,.38],'upper_span_ratio':[.50,.70]}},
    'open_arch': {'text':'A low, wide gently arched backrest rail with a broad central open cutout (open-back chair), and softly curved thin armrests that leave large visible white side openings. Keep the four floor-contact legs and seat.',
                  'provenance':'exact shape clause from built-in image-generation prompt',
                  'target':{'central_open_fraction':[.50,.70],'upper_span_ratio':[.85,1.10]}},
    'raised_wing_solid': {'text':'Make the upper silhouette broad and gently wing-shaped, with a SOLID CENTRAL BACKREST PANEL across the full middle width at approximately the original upper-back height. Raise only the bottom edge of the backrest modestly to show a larger white gap above the seat. Keep the four floor-contact legs and seat.',
                          'provenance':'exact shape clause from built-in image-generation prompt',
                          'target':{'central_open_fraction':[.40,.58],'upper_span_ratio':[.82,1.06]}},
}
EXISTING_3D={
    'baseline':'seed_42',
    'angular':'image_angular_seed42',
    'wing':'image_wing_seed42',
}


def normalized(path:Path) -> tuple[np.ndarray,np.ndarray]:
    image=cv2.imread(str(path))
    if image is None:raise FileNotFoundError(path)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    fg=gray<190
    yy,xx=np.where(fg)
    if len(xx)==0:raise ValueError(f'No dark foreground: {path}')
    crop=image[yy.min():yy.max()+1,xx.min():xx.max()+1]
    height,width=crop.shape[:2]
    scale=min(448/width,448/height)
    new_size=(round(width*scale),round(height*scale))
    resized=cv2.resize(crop,new_size,interpolation=cv2.INTER_AREA)
    canvas=np.full((N,N,3),255,np.uint8)
    x0=(N-new_size[0])//2;y0=(N-new_size[1])//2
    canvas[y0:y0+new_size[1],x0:x0+new_size[0]]=resized
    mask=cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)<190
    return canvas,mask


def roi(mask:np.ndarray,x0:float,x1:float,y0:float,y1:float)->np.ndarray:
    return mask[int(y0*N):int(y1*N),int(x0*N):int(x1*N)]


def span(mask:np.ndarray,x0:float,x1:float,y0:float,y1:float)->float:
    points=np.where(roi(mask,x0,x1,y0,y1))[1]
    if len(points)<10:return 0.
    return float((np.percentile(points,95)-np.percentile(points,5))/N)


def metrics(mask:np.ndarray,reference:np.ndarray)->dict:
    central=float(1-roi(mask,.35,.65,.24,.43).mean())
    top=span(mask,.1,.9,.09,.18)
    middle=span(mask,.1,.9,.37,.45)
    ratio=top/middle if middle else 0.
    # Shared projected interfaces in normalized image coordinates; this is a
    # front-view plausibility screen, NOT physical BC verification.
    regions=np.zeros((N,N),bool)
    regions[205:280,145:365]=True  # visible seat
    regions[390:495,100:185]=True  # left projected feet/legs
    regions[390:495,330:415]=True  # right projected feet/legs
    required=reference&regions
    retention=float((mask&required).sum()/required.sum())
    # Center of the backrest is a distinct load interface: an attractive arch
    # with this patch entirely white cannot accept the registered back load.
    back_patch=roi(mask,.44,.56,.18,.25)
    back_reference=roi(reference,.44,.56,.18,.25)
    back_retention=float((back_patch&back_reference).sum()/back_reference.sum())
    silhouette_iou=float((mask&reference).sum()/(mask|reference).sum())
    return {'central_open_fraction':central,'upper_span_ratio':float(ratio),
            'projected_interface_retention':retention,
            'projected_back_load_retention':back_retention,
            'silhouette_iou_to_reference':silhouette_iou}


def image_cell(row:dict)->list[int]|None:
    indices=[]
    for key,(lower,upper) in DESCRIPTORS.items():
        value=row[key]
        if not lower<=value<=upper:return None
        indices.append(min(2,int((value-lower)/(upper-lower)*3)))
    return indices


def target_score(row:dict,target:dict)->float:
    penalties=[]
    for key,(lo,hi) in target.items():
        value=row[key]
        miss=max(lo-value,0,value-hi)
        penalties.append(miss/(DESCRIPTORS[key][1]-DESCRIPTORS[key][0]))
    return float(max(0,1-sum(penalties)/len(penalties)))


def main()->None:
    OUT.mkdir(parents=True,exist_ok=True)
    ref_image,reference=normalized(OUT/'images/baseline.png')
    old_3d={r['id']:r for r in json.loads((BASE/'single_view_qd_2026-10-03/summary.json').read_text())['rows']}
    new_3d={r['id']:r for r in json.loads((OUT/'transfer_3d.json').read_text())} if (OUT/'transfer_3d.json').exists() else {}
    rows=[]
    masks={}
    for name,entry in INTENT.items():
        source=OUT/'images'/f'{name}.png'
        norm,mask=normalized(source)
        masks[name]=mask
        cv2.imwrite(str(OUT/'images'/f'{name}_normalized.png'),norm)
        row={'id':name,'source_image':str(source),
             'normalized_image':str(OUT/'images'/f'{name}_normalized.png'),
             'text_intent':entry['text'],'text_provenance':entry['provenance'],
             'target_intervals':entry['target'],
             **metrics(mask,reference)}
        row['intent_score']=target_score(row,entry['target'])
        row['cell']=image_cell(row)
        row['image_gate']=bool(row['cell'] is not None and
                               row['projected_interface_retention']>=.90 and
                               row['projected_back_load_retention']>=.80)
        row['existing_3d_id']=EXISTING_3D.get(name)
        mesh_id=row['existing_3d_id'] or f'image_{name}_seed42'
        mesh_result=old_3d.get(mesh_id) or new_3d.get(mesh_id)
        row['known_3d_gate']=(bool(mesh_result.get('fea_valid',mesh_result.get('geometry_gate')))
                              if mesh_result else None)
        rows.append(row)
    archive={}
    for row in rows:
        if row['image_gate']:
            archive.setdefault(tuple(row['cell']),[]).append(row)
    elites={','.join(map(str,cell)):max(items,key=lambda r:(r['intent_score']*r['projected_interface_retention'],r['projected_interface_retention']))['id']
            for cell,items in sorted(archive.items())}
    validated_3d_cells={tuple(r['cell']) for r in rows if r['image_gate'] and r['known_3d_gate']}
    summary={'evaluated_images':len(rows),'image_gate_pass':sum(r['image_gate'] for r in rows),
             'descriptor_ranges':DESCRIPTORS,'archive_dims':[3,3],
             'filled_cells':len(archive),'cell_elites':elites,
             'image_cells_with_verified_3d':len(validated_3d_cells),
             'rows':rows,
             'limitations':['The image gate sees two projected leg/foot zones, not four independent 3D contacts.',
                            'Text intent is a structured attribute target, not a text-embedding distance or an automated vision-language judgment.',
                            'Existing baseline/angular/wing exact GPT prompts were not archived, so their labels are descriptive reconstructions.',
                            'tall_rect and open_arch were both converted to 3D after the initial image gate; both failed the registered 3D geometry gate.',
                            'The back-load patch check was added after diagnosing open_arch and is exploratory, not preregistered.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    fig,axs=plt.subplots(2,3,figsize=(12,8),layout='constrained')
    for ax,row in zip(axs.flat,rows):
        ax.imshow(cv2.cvtColor(cv2.imread(row['normalized_image']),cv2.COLOR_BGR2RGB))
        ax.set_title(f"{row['id']}  cell={row['cell']}  pass={row['image_gate']}\nopen={row['central_open_fraction']:.2f} span={row['upper_span_ratio']:.2f} seat/feet={row['projected_interface_retention']:.2f} back={row['projected_back_load_retention']:.2f}",fontsize=9)
        ax.axis('off')
    for ax in list(axs.flat)[len(rows):]:ax.axis('off')
    fig.savefig(OUT/'contact_sheet.png',dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    for x in np.linspace(*DESCRIPTORS['central_open_fraction'],4)[1:-1]:ax.axvline(x,color='#aaa',lw=1)
    for y in np.linspace(*DESCRIPTORS['upper_span_ratio'],4)[1:-1]:ax.axhline(y,color='#aaa',lw=1)
    for row in rows:
        ax.scatter(row['central_open_fraction'],row['upper_span_ratio'],s=120,
                   c='#247b61' if row['image_gate'] else '#b55d44',edgecolor='black')
        ax.annotate(row['id'],(row['central_open_fraction'],row['upper_span_ratio']),
                    xytext=(5,5),textcoords='offset points')
    ax.set_xlim(*DESCRIPTORS['central_open_fraction'])
    ax.set_ylim(*DESCRIPTORS['upper_span_ratio'])
    ax.set_xlabel('central back-seat opening (white fraction)')
    ax.set_ylabel('upper / mid silhouette span')
    ax.set_title(f'Image phenotype archive: {len(archive)}/9 cells')
    ax.grid(alpha=.2)
    fig.savefig(OUT/'image_archive.png',dpi=170);plt.close(fig)

    cases=[]
    for row in rows:
        item=html.escape(row['id']);text=html.escape(row['text_intent'])
        mesh=(f"<a href='../single_view_qd_2026-10-03/{row['existing_3d_id']}/aligned_main.obj'>기존 3D OBJ</a>" if row['existing_3d_id'] not in (None,'seed_42') else
              "<a href='../minimal_direct3ds2_2026-10-02/aligned_sparse_main.obj'>기존 3D OBJ</a>" if row['existing_3d_id']=='seed_42' else '3D 미실행')
        cases.append(f"<article><img src='images/{item}_normalized.png'><h3>{item} · 셀 {row['cell']} · {'통과' if row['image_gate'] else '제외'}</h3><p>{text}</p><p>열린 공간 {row['central_open_fraction']:.3f} · 상단 폭 {row['upper_span_ratio']:.3f}<br>좌판·발 투영 유지 {row['projected_interface_retention']:.1%} · 등받이 유지 {row['projected_back_load_retention']:.1%}<br>의도 점수 {row['intent_score']:.2f}</p><p>{mesh}</p></article>")
    transfer=json.loads((OUT/'transfer_3d.json').read_text()) if (OUT/'transfer_3d.json').exists() else []
    transfer_cards=[]
    for row in transfer:
        name=row['id'].removeprefix('image_').removesuffix('_seed42')
        reason=('3D 사양 통과' if row['geometry_gate'] else
                f"3D 사양 탈락: 보정 {row['repair_fraction']:.1%}, 연결 성분 {row['realized_components_6conn']}개, 원본 좌판 하중 접촉 {row['seat_coverage']:.0%}, 등받이 접촉 {row['back_coverage']:.0%}")
        transfer_cards.append(f"<article><img src='{name}_3d_preview.png'><h3>{name} → 3D</h3><p>{reason}</p><p>기준 3D와 voxel IoU {row['realized_iou_to_baseline']:.3f}</p><a href='../single_view_qd_2026-10-03/{row['id']}/aligned_main.obj'>원본 OBJ</a></article>")
    (OUT/'index.html').write_text(f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Chair image-first QD</title><style>body{{font:16px/1.5 system-ui;max-width:1450px;margin:0 auto;padding:24px;background:#f1f4f6;color:#17212b}}article,.panel{{background:white;border:1px solid #d8e0e6;border-radius:12px;padding:16px;margin:12px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}}img{{max-width:100%}}article img{{width:100%}}a{{color:#155e99}}.note{{border-left:4px solid #b36c24;padding:12px;background:#fff2dd}}</style><h1>Text intent → image phenotype → QD → 3D 검증</h1><p>단일 정면 이미지 {len(rows)}장을 검정 금속 형상으로 분리하고, 같은 크기로 정렬한 다음 픽셀에서 측정했다. 텍스트는 목표 속성 구간을 지정한다. 이미지가 실제로 그 구간에 들어왔는지를 점검하며, 형상 셀은 문장 간 거리가 아니라 생성된 이미지의 두 지표로 정한다.</p><section class="panel"><h2>이미지 QD: {len(archive)}/9 셀 · 3D 검증된 이미지 셀: {len(validated_3d_cells)}/9</h2><p>축 1: 등받이와 좌판 사이의 중앙 열린 공간. 축 2: 상단 윤곽 폭/중간 윤곽 폭. 좌판·발 정면 투영 보존율 90% 이상과 등받이 하중 패치 보존율 80% 이상을 통과 조건으로 사용했다. 각 셀에서는 텍스트 의도 일치와 보존율이 높은 이미지를 기록한다.</p><img src="image_archive.png"><img src="contact_sheet.png"><p><a href="summary.json">측정 JSON</a> · <a href="prompt_records.json">새 이미지의 정확한 프롬프트</a> · <a href="../single_view_qd_2026-10-03/index.html">기존 3D QD 결과</a></p></section><p class="note"><b>해석 범위:</b> 정면 이미지에는 뒤쪽 다리 접촉점과 3D 연결성이 보이지 않는다. 2D 보존율은 BC 검증의 대체가 아니다. angular는 기준과 같은 이미지 셀이다. wing은 다른 이미지 셀이지만 기존 3D에서는 탈락했다. open_arch의 중앙 등받이 구멍을 보고 등받이 패치 검사를 추가했으므로 이 게이트는 사후 진단용이며 사전 등록된 실험이 아니다. 기존 3장의 정확한 생성 프롬프트는 기록되지 않아 표시된 문장은 의도 요약이다.</p><h2>생성 이미지와 3D 변환</h2><div class="grid">{''.join(cases)}</div><section class="panel"><h2>새 이미지의 3D 전달 검사</h2><p>tall_rect와 open_arch를 동일한 Direct3D-S2 한 장 입력·seed 42로 변환했다. 둘 다 3D 사양 게이트를 통과하지 못해 FEA 후보에 넣지 않았다. 2D 이미지 셀의 다양성이 물리적으로 유효한 3D 다양성을 보장하지 않는다는 실험 결과다.</p><div class="grid">{''.join(transfer_cards)}</div><a href="transfer_3d.json">3D 전달 수치</a></section></html>''')
    print(json.dumps({'images':len(rows),'gate_pass':summary['image_gate_pass'],
                      'filled_cells':summary['filled_cells'],'elites':elites},indent=2))


if __name__=='__main__':main()
