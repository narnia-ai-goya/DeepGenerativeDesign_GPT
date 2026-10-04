#!/usr/bin/env python3
"""Visual summary of the chair BC connectivity dense ablation."""
from __future__ import annotations

import json
import sys

import numpy as np
import trimesh
from PIL import Image,ImageDraw

from make_chair_domain import ROOT
from report_chair_prefea_stages import silhouette_scores

sys.path.insert(0,str(ROOT/'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim,render_lit  # noqa: E402

BASE=ROOT/'experiments/chair/sofa_style_2026-09-28/consistent_reference_2026-09-28'
OUT=BASE/'bc_connectivity_ablation'
CASES=(
 ('image baseline',BASE/'image_only/dense/mesh_dense.obj',BASE/'image_only/bc_audit.json'),
 ('shape baseline',BASE/'shape_guided/dense/mesh_dense.obj',BASE/'shape_guided/bc_audit.json'),
 ('image + BC path',OUT/'image_bc_connected/dense/mesh_dense.obj',OUT/'image_bc_connected/bc_audit.json'),
 ('shape + BC path 3.5',OUT/'shape_bc_connected/dense/mesh_dense.obj',OUT/'shape_bc_connected/bc_audit.json'),
 ('shape + BC path 1.0',OUT/'shape_bc_pw1/dense/mesh_dense.obj',OUT/'shape_bc_pw1/bc_audit.json'),
 ('shape + local reach',OUT/'shape_bc_local/dense/mesh_dense.obj',OUT/'shape_bc_local/bc_audit.json'),
)
VIEWS=(('oblique',25,35),('front',15,0),('right',15,90),('rear',15,180))
STEMS={'front':'v00_front_lo','right':'v02_right_lo','rear':'v04_back_lo'}


def main():
    env=trimesh.load(ROOT/'data_real/chair/original_DesignSpace.stl',force='mesh')
    center=env.bounds.mean(axis=0)
    radius=float(np.linalg.norm(env.extents))*1.5
    extent=float(env.extents.max())/2
    size,header=340,34
    sheet=Image.new('RGB',(size*len(VIEWS),(size+header)*len(CASES)),'white')
    draw=ImageDraw.Draw(sheet)
    rows=[]
    for i,(name,path,audit_path) in enumerate(CASES):
        mesh=trimesh.load(path,force='mesh')
        audit=json.loads(audit_path.read_text())
        row={'name':name,'mesh':str(path),'bc_pass':audit['bc_geometry_pass'],
             'shape_pass':audit['shape_gate_pass'],
             'largest_component_volume_fraction':audit['largest_component_volume_fraction'],
             'top_bar_coverage':audit['top_bar']['coverage'],'silhouettes':{}}
        for j,(view,elev,azim) in enumerate(VIEWS):
            eye,up=camera_from_elev_azim(center,radius,elev,azim)
            rendered=Image.fromarray(render_lit(mesh,eye,center,up,size=size,
                fit_extent=extent,margin=1.15,color=(.48,.52,.56))).convert('RGB')
            sheet.paste(rendered,(j*size,i*(size+header)+header))
            draw.text((j*size+8,i*(size+header)+8),f'{name}: {view}',fill='#20252b')
            if view in STEMS:
                inp=Image.open(BASE/'input'/f'{STEMS[view]}.png').convert('RGBA')
                target=Image.new('RGB',inp.size,'white')
                target.paste(inp,mask=inp.getchannel('A'))
                row['silhouettes'][view]=silhouette_scores(rendered,target.resize((size,size)))
        rows.append(row)
    sheet.save(OUT/'comparison.png')
    (OUT/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    table=''.join('<tr><td><a href="/'+str(path.relative_to(ROOT))+'">'+name+'</a></td>'
        +f'<td>{"예" if row["bc_pass"] else "아니오"}</td>'
        +f'<td>{row["largest_component_volume_fraction"]:.3f}</td>'
        +f'<td>{row["top_bar_coverage"]:.3f}</td>'
        +''.join(f'<td>{row["silhouettes"][v]["iou"]:.3f}</td>' for v in ('front','right','rear'))
        +'</tr>' for row,(_,path,_) in zip(rows,CASES))
    (OUT/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
      '<title>Chair BC connectivity ablation</title><style>body{font:16px system-ui;'
      'max-width:1500px;margin:2rem auto;background:#f4f6f8;color:#1d2730}'
      'article{background:white;padding:1rem;border-radius:12px}img{max-width:100%}'
      'table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.4rem}</style>'
      '<article><h1>의자 BC 연결 제약 비교</h1><p>모든 결과는 같은 4뷰 이미지, '
      '네 발·좌석 하중 BC, envelope, seed를 사용했습니다. FEA와 sparse는 껐습니다.</p>'
      '<p><strong>결론:</strong> 현재 cw/pw 손실을 켜면 발–좌석 연결은 좋아질 수 '
      '있지만 등받이가 사라지거나 분리됩니다. BC와 형상 기준을 동시에 만족한 '
      '후보는 없습니다.</p><table><tr><th>dense OBJ</th><th>BC 검증</th>'
      '<th>최대 성분 체적 비율</th><th>윗가로대 보존율</th><th>앞 IoU</th>'
      '<th>오른쪽 IoU</th><th>뒤 IoU</th></tr>'+table+'</table>'
      '<p><a href="metrics.json">전체 수치</a> · <a href="STUDY.md">해석</a> · '
      '<a href="corridor_validation.json">경로 마스크 검증</a></p>'
      '<a href="comparison.png"><img src="comparison.png"></a></article></html>')
    print(OUT/'index.html')


if __name__=='__main__':main()
