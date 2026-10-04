#!/usr/bin/env python3
"""Publish an input/dense/sparse gallery for chair backrest BC examples."""
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
OUT=BASE/'backrest_bc_multi_examples_2026-09-29'
VIEWS=(('oblique',25,35),('front',15,0),('right',15,90),('top',85,0))
STEMS={'front':'v00_front_lo','right':'v02_right_lo','top':'v_top'}


def image_source(case):
    if case=='diagonal_braced':
        return ROOT/'experiments/chair/image_concepts_2026-09-27/diagonal_braced/multiview_registered'
    return BASE/case


def render_rows(case):
    sub=OUT/case
    entries=[('shape reference',sub/'prototype.obj',None),
             ('dense',sub/'dense/mesh_dense.obj',sub/'dense_bc_audit.json')]
    if case=='diagonal_braced':
        entries += [('dense pw2',sub/'dense_pw2/mesh_dense.obj',sub/'dense_pw2_bc_audit.json'),
                    ('sparse pw2 d13',sub/'sparse_pw2_d13/generation/mesh.obj',
                     sub/'sparse_pw2_d13/bc_audit.json')]
    else:
        entries += [('sparse d13',sub/'sparse_d13/generation/mesh.obj',
                     sub/'sparse_d13/bc_audit.json')]
    entries=[e for e in entries if e[1].exists()]
    env=trimesh.load(ROOT/'data_real/chair/original_DesignSpace.stl',force='mesh')
    center=env.bounds.mean(axis=0)
    radius=float(np.linalg.norm(env.extents))*1.5
    scale=float(env.extents.max())/2
    size,header=340,34
    sheet=Image.new('RGB',(size*len(VIEWS),(size+header)*len(entries)),'white')
    draw=ImageDraw.Draw(sheet)
    records=[]
    for i,(name,path,audit_path) in enumerate(entries):
        mesh=trimesh.load(path,force='mesh')
        audit=json.loads(audit_path.read_text()) if audit_path and audit_path.exists() else None
        record={'stage':name,'mesh':str(path),'bc_pass':audit['bc_geometry_pass'] if audit else None,
                'mesh_components':audit['mesh_components'] if audit else len(mesh.split()),
                'bar_coverage':audit['top_bar']['coverage'] if audit else None,
                'silhouettes':{}}
        for j,(view,elev,azim) in enumerate(VIEWS):
            eye,up=camera_from_elev_azim(center,radius,elev,azim)
            image=Image.fromarray(render_lit(mesh,eye,center,up,size=size,
                fit_extent=scale,margin=1.15,color=(.48,.52,.56))).convert('RGB')
            sheet.paste(image,(j*size,i*(size+header)+header))
            draw.text((j*size+8,i*(size+header)+8),f'{name}: {view}',fill='#20252b')
            if view in STEMS:
                target=Image.open(image_source(case)/'input'/f'{STEMS[view]}.png').convert('RGB')
                record['silhouettes'][view]=silhouette_scores(image,target.resize((size,size)))
        records.append(record)
    sheet.save(sub/'dense_sparse_comparison.png')
    (sub/'report_metrics.json').write_text(json.dumps(records,indent=2)+'\n')
    return records


def make_gallery():
    env=trimesh.load(ROOT/'data_real/chair/original_DesignSpace.stl',force='mesh')
    center=env.bounds.mean(axis=0)
    radius=float(np.linalg.norm(env.extents))*1.5
    scale=float(env.extents.max())/2
    eye,up=camera_from_elev_azim(center,radius,25,35)
    size,header=400,45
    sheet=Image.new('RGB',(size*3,(size+header)*3),'white')
    draw=ImageDraw.Draw(sheet)
    for i,case in enumerate(('open_arm','solid_side','diagonal_braced')):
        sub=OUT/case
        dense=sub/('dense_pw2/mesh_dense.obj' if case=='diagonal_braced' else 'dense/mesh_dense.obj')
        sparse=sub/('sparse_pw2_d13/generation/mesh.obj' if case=='diagonal_braced'
                     else 'sparse_d13/generation/mesh.obj')
        inp=Image.open(image_source(case)/'input/v00_front_lo.png').convert('RGB').resize((size,size))
        ims=[inp]
        for path in (dense,sparse):
            mesh=trimesh.load(path,force='mesh')
            ims.append(Image.fromarray(render_lit(mesh,eye,center,up,size=size,
                fit_extent=scale,margin=1.15,color=(.48,.52,.56))).convert('RGB'))
        for j,(im,label) in enumerate(zip(ims,('input front','dense 3D','sparse 3D'))):
            sheet.paste(im,(j*size,i*(size+header)+header))
            draw.text((j*size+10,i*(size+header)+13),f'{case}: {label}',fill='#20252b')
    sheet.save(OUT/'gallery.png')


def main():
    make_gallery()
    cards=[]; all_records={}
    for case,title in (('open_arm','Open arm'),('solid_side','Solid side'),
                       ('diagonal_braced','Diagonal braced')):
        records=render_rows(case)
        all_records[case]=records
        trs=[]
        for row in records:
            path='/'+str((ROOT/row['mesh']).resolve().relative_to(ROOT))
            bar='—' if row['bar_coverage'] is None else format(row['bar_coverage'],'.3f')
            cells=[f'<td><a href="{path}">{html.escape(row["stage"])}</a></td>',
                   f'<td>{row["bc_pass"]}</td>',f'<td>{row["mesh_components"]}</td>',
                   f'<td>{bar}</td>']
            cells += [f'<td>{row["silhouettes"][v]["iou"]:.3f}</td>' for v in STEMS]
            trs.append('<tr>'+''.join(cells)+'</tr>')
        input_contact='/'+str((image_source(case)/'input_contact.png').relative_to(ROOT))
        cards.append(f'<article><h2>{title}</h2>'
          f'<p>입력 이미지: <a href="{input_contact}">원본</a></p>'
          f'<img src="{input_contact}">'
          '<table><tr><th>단계/OBJ</th><th>BC 검증</th><th>성분</th><th>등받이 윗부분</th>'
          '<th>앞 IoU</th><th>오른쪽 IoU</th><th>위 IoU</th></tr>'
          +''.join(trs)+'</table>'
          f'<p><a href="{case}/report_metrics.json">수치 JSON</a></p>'
          f'<img src="{case}/dense_sparse_comparison.png"></article>')
    (OUT/'metrics.json').write_text(json.dumps(all_records,indent=2)+'\n')
    (OUT/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
      '<title>Chair backrest BC: multiple examples</title><style>body{font:16px system-ui;'
      'max-width:1600px;margin:2rem auto;background:#f4f6f8;color:#1d2730}'
      'article{background:white;padding:1rem;margin:1rem 0;border-radius:12px}'
      'img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;'
      'padding:.4rem}</style><article><h1>등받이 BC: 여러 의자 입력</h1>'
      '<p>서로 다른 open-arm/solid-side/diagonal-braced 입력을 사용한 dense→sparse 비교. '
      '고정 발·좌석 하중·등받이 BC 및 경로 정의는 동일하고, '
      'diagonal-braced의 경로 가중치만 높였습니다. '
      'FEA는 꺼져 있습니다.</p><p><a href="REPORT.md">해석</a> · '
      '<a href="metrics.json">전체 수치</a> · '
      '<a href="prototype_comparison.png">형상 기준 비교</a> · '
      '<a href="qd_framework.svg">QD 프레임워크 SVG</a></p>'
      '<img src="qd_framework.png" alt="디자이너 중심 QD 프레임워크">'
      '<h2>세 가지 입력의 최종 형태</h2><img src="gallery.png"></article>'
      +''.join(cards)+'</html>')
    print(OUT/'index.html')


if __name__=='__main__':main()
