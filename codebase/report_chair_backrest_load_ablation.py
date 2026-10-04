#!/usr/bin/env python3
"""Render the chair backrest-load BC ablation and publish a local report."""
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

BASE=ROOT/'experiments/chair/sofa_style_2026-09-28'
REF=BASE/'consistent_reference_2026-09-28'
OUT=REF/'backrest_load_ablation'
CASES=(
 ('3D reference',BASE/'coherent_proxy_dense/prototype.obj',None),
 ('seat BC only',REF/'shape_guided/dense/mesh_dense.obj',REF/'shape_guided/bc_audit.json'),
 ('seat + back BC',OUT/'dense/mesh_dense.obj',OUT/'bc_audit.json'),
 ('back BC + connectivity loss',OUT/'back_cw_local/dense/mesh_dense.obj',OUT/'back_cw_local/bc_audit.json'),
 ('back BC + rear path',OUT/'back_support_pw1/dense/mesh_dense.obj',OUT/'back_support_pw1/bc_audit.json'),
 ('all paths, volume off',OUT/'all_support_vw0/dense/mesh_dense.obj',OUT/'all_support_vw0/bc_audit.json'),
 ('all paths, volume .35',OUT/'all_support_v35/dense/mesh_dense.obj',OUT/'all_support_v35/bc_audit.json'),
)
VIEWS=(('oblique',25,35),('front',15,0),('right',15,90),('rear',15,180),('top',85,0))
STEMS={'front':'v00_front_lo','right':'v02_right_lo','rear':'v04_back_lo','top':'v_top'}


def bc_preview(center,radius,scale):
    import pyvista as pv
    pv.OFF_SCREEN=True
    proxy=trimesh.load(BASE/'coherent_proxy_dense/prototype.obj',force='mesh')
    back=trimesh.load(OUT/'backrest_load.stl',force='mesh')
    seat=trimesh.load(ROOT/'data_real/chair/load_remesh.stl',force='mesh')
    feet=trimesh.load(ROOT/'data_real/chair/fixed_remesh.stl',force='mesh')
    eye,up=camera_from_elev_azim(center,radius,25,35)
    pl=pv.Plotter(off_screen=True,window_size=(1100,900))
    pl.background_color='white'
    pl.add_mesh(pv.wrap(proxy),color='#a7b0b9',opacity=.28)
    pl.add_mesh(pv.wrap(feet),color='#bc503f',opacity=1)
    pl.add_mesh(pv.wrap(seat),color='#249062',opacity=1)
    pl.add_mesh(pv.wrap(back),color='#b632a0',opacity=1)
    pl.camera_position=[eye.tolist(),center.tolist(),up.tolist()]
    pl.camera.parallel_projection=True
    pl.camera.parallel_scale=scale*1.15
    pl.screenshot(str(OUT/'bc_patch_preview.png'))
    pl.close()


def main():
    env=trimesh.load(ROOT/'data_real/chair/original_DesignSpace.stl',force='mesh')
    center=env.bounds.mean(axis=0)
    radius=float(np.linalg.norm(env.extents))*1.5
    scale=float(env.extents.max())/2
    bc_preview(center,radius,scale)
    size,header=340,34
    sheet=Image.new('RGB',(size*len(VIEWS),(size+header)*len(CASES)),'white')
    draw=ImageDraw.Draw(sheet)
    rows=[]
    for i,(name,path,audit_path) in enumerate(CASES):
        mesh=trimesh.load(path,force='mesh')
        audit=json.loads(audit_path.read_text()) if audit_path else None
        row={'name':name,'mesh':str(path),'components':len(mesh.split(only_watertight=False)),
             'volume_litres':round(abs(float(mesh.volume))*1000,3),
             'bc_pass':audit['bc_geometry_pass'] if audit else None,
             'top_bar_coverage':audit['top_bar']['coverage'] if audit else 1.0,
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
                row['silhouettes'][view]=silhouette_scores(image,tgt.resize((size,size)))
        rows.append(row)
    sheet.save(OUT/'comparison.png')
    (OUT/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    table=''.join('<tr><td><a href="/'+str(path.relative_to(ROOT))+'">'+name+'</a></td>'
        +f'<td>{"—" if row["bc_pass"] is None else ("통과" if row["bc_pass"] else "실패")}</td>'
        +f'<td>{row["top_bar_coverage"]:.3f}</td><td>{row["components"]}</td>'
        +f'<td>{row["volume_litres"]:.1f}</td>'
        +''.join(f'<td>{row["silhouettes"][v]["iou"]:.3f}</td>' for v in ('front','right','rear','top'))
        +'</tr>' for row,(_,path,_) in zip(rows,CASES))
    (OUT/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
      '<title>Chair backrest load experiment</title><style>body{font:16px system-ui;'
      'max-width:1780px;margin:2rem auto;background:#f4f6f8;color:#1d2730}'
      'article{background:white;padding:1rem;margin:1rem 0;border-radius:12px}'
      'img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;'
      'padding:.4rem}</style><article><h1>등받이 하중 BC를 추가한 의자 dense 실험</h1>'
      '<p>빨강: 네 고정 발, 초록: 좌석 하중, 자홍: 새 등받이 하중 패치. '
      '현재는 FEA off인 기하학적 BC 실험입니다. 미래 물리 하중의 의도 방향은 +Y입니다.</p>'
      '<img src="bc_patch_preview.png"><p><strong>결과:</strong> 등받이 패치만 추가하면 '
      '윗가로대가 살아나지만 떨어져 있습니다. 뒤 지지 경로만 더하면 앞 두 발이 끊어집니다. '
      '네 발과 등받이 경로를 함께 주었을 때 여섯 BC 영역이 한 메시로 연결됩니다.</p>'
      '<table><tr><th>OBJ</th><th>BC 검증</th><th>윗가로대 보존율</th><th>성분</th>'
      '<th>체적 L</th><th>앞 IoU</th><th>오른쪽 IoU</th><th>뒤 IoU</th>'
      '<th>위 IoU</th></tr>'+table+'</table><p><a href="metrics.json">수치</a> · '
      '<a href="STUDY.md">해석</a> · <a href="patch_validation.json">패치 검증</a>'
      '</p><img src="comparison.png"></article></html>')
    print(OUT/'index.html')


if __name__=='__main__':main()
