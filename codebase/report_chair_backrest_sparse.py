#!/usr/bin/env python3
"""Compare chair dense/sparse backrest BC results in five consistent views."""
from __future__ import annotations

import html
import json
import sys

import numpy as np
import trimesh
from PIL import Image,ImageDraw

from make_chair_domain import ROOT
from report_chair_prefea_stages import silhouette_scores

sys.path.insert(0,str(ROOT/'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim,render_lit  # noqa: E402

BASE=ROOT/'experiments/chair/sofa_style_2026-09-28'
REF=BASE/'consistent_reference_2026-09-28'
OUT=REF/'backrest_load_ablation'
VIEWS=(('oblique',25,35),('front',15,0),('right',15,90),('rear',15,180),('top',85,0))
STEMS={'front':'v00_front_lo','right':'v02_right_lo','rear':'v04_back_lo','top':'v_top'}


def main():
    rows=[('3D reference',BASE/'coherent_proxy_dense/prototype.obj',None)]
    for case in ('all_support_vw0','all_support_v35'):
        root=OUT/case
        rows.append((f'{case}: dense',root/'dense/mesh_dense.obj',root/'bc_audit.json'))
        for sparse in ('sparse_followup','sparse_followup_d12','sparse_followup_d13'):
            path=root/sparse/'generation/mesh.obj'
            if path.exists():
                rows.append((f'{case}: {sparse}',path,root/sparse/'bc_audit.json'))
    env=trimesh.load(ROOT/'data_real/chair/original_DesignSpace.stl',force='mesh')
    center=env.bounds.mean(axis=0)
    radius=float(np.linalg.norm(env.extents))*1.5
    scale=float(env.extents.max())/2
    size,header=340,34
    sheet=Image.new('RGB',(size*len(VIEWS),(size+header)*len(rows)),'white')
    draw=ImageDraw.Draw(sheet)
    metrics=[]
    for i,(name,path,audit_path) in enumerate(rows):
        mesh=trimesh.load(path,force='mesh')
        audit=json.loads(audit_path.read_text()) if audit_path and audit_path.exists() else None
        record={'name':name,'mesh':str(path),'audit':str(audit_path) if audit_path else None,
                'bc_pass':audit['bc_geometry_pass'] if audit else None,
                'top_bar_coverage':audit['top_bar']['coverage'] if audit else 1.0,
                'mesh_components':audit['mesh_components'] if audit else len(mesh.split()),
                'silhouettes':{}}
        for j,(view,elev,azim) in enumerate(VIEWS):
            eye,up=camera_from_elev_azim(center,radius,elev,azim)
            image=Image.fromarray(render_lit(mesh,eye,center,up,size=size,
                fit_extent=scale,margin=1.15,color=(.48,.52,.56))).convert('RGB')
            sheet.paste(image,(j*size,i*(size+header)+header))
            draw.text((j*size+8,i*(size+header)+8),f'{name}: {view}',fill='#20252b')
            if view in STEMS:
                inp=Image.open(REF/'input'/f'{STEMS[view]}.png').convert('RGBA')
                tgt=Image.new('RGB',inp.size,'white')
                tgt.paste(inp,mask=inp.getchannel('A'))
                record['silhouettes'][view]=silhouette_scores(image,tgt.resize((size,size)))
        metrics.append(record)
    sheet.save(OUT/'sparse_comparison.png')
    (OUT/'sparse_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    trs=[]
    for row in metrics:
        name=html.escape(row['name'])
        path='/'+str((ROOT/row['mesh']).resolve().relative_to(ROOT))
        parts=[f'<td><a href="{path}">{name}</a></td>',
               f'<td>{row["bc_pass"]}</td>',
               f'<td>{row["mesh_components"]}</td>',
               f'<td>{row["top_bar_coverage"]:.3f}</td>']
        parts += [f'<td>{row["silhouettes"][v]["iou"]:.3f}</td>' for v in STEMS]
        trs.append('<tr>'+''.join(parts)+'</tr>')
    (OUT/'sparse_index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<title>Chair backrest BC dense to sparse</title><style>body{font:16px system-ui;'
        'max-width:1780px;margin:2rem auto;background:#f4f6f8;color:#1d2730}'
        'article{background:white;padding:1rem;margin:1rem 0;border-radius:12px}'
        'img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;'
        'padding:.4rem}</style><article><h1>등받이 BC: dense → sparse 비교</h1>'
        '<p>동일한 dense 캐시에서 sparse를 실행했습니다. FEA는 꺼져 있습니다. '
        'sparse_followup_d12 / d13은 STL 기반 BC 채움의 확장량을 각각 12 / 13 mm로 설정한 경우입니다.</p>'
        '<table><tr><th>OBJ</th><th>BC 검증</th><th>성분</th><th>윗가로대 보존율</th>'
        '<th>앞 IoU</th><th>오른쪽 IoU</th><th>뒤 IoU</th><th>위 IoU</th></tr>'
        +''.join(trs)+'</table><p><a href="sparse_metrics.json">수치 JSON</a> · '
        '<a href="SPARSE_STUDY.md">해석</a></p><img src="sparse_comparison.png">'
        '</article></html>')
    print(OUT/'sparse_index.html')


if __name__=='__main__': main()
