#!/usr/bin/env python3
"""Render and compare chair sparse thickness/image-guidance ablations."""
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
from cond_render_pv import camera_from_elev_azim,render_lit

SOURCE=ROOT/'experiments/chair/sofa_style_2026-09-28/backrest_bc_multi_examples_2026-09-29/diagonal_braced'
INPUT=ROOT/'experiments/chair/image_concepts_2026-09-27/diagonal_braced/multiview_registered/input'
OUT=ROOT/'experiments/chair/sofa_style_2026-09-28/sparse_style_image_guidance_2026-09-29'
ROWS=(
 ('baseline thick 10',SOURCE/'sparse_pw2_d13/generation/mesh.obj',SOURCE/'sparse_pw2_d13/bc_audit.json',10.,0.),
 ('thick 3',OUT/'thick_3/generation/mesh.obj',OUT/'thick_3/bc_audit.json',3.,0.),
 ('thick 6',OUT/'thick_6/generation/mesh.obj',OUT/'thick_6/bc_audit.json',6.,0.),
 ('thick 6 + image 0.5',OUT/'thick_6_proj_0p5/generation/mesh.obj',OUT/'thick_6_proj_0p5/bc_audit.json',6.,.5),
 ('thick 6 + image 2',OUT/'thick_6_proj_2/generation/mesh.obj',OUT/'thick_6_proj_2/bc_audit.json',6.,2.),
)
VIEWS=(('oblique',25,35),('front',15,0),('right',15,90),('top',85,0))
TARGETS={'front':INPUT/'v00_front_lo.png','right':INPUT/'v02_right_lo.png','top':INPUT/'v_top.png'}


def main():
    env=trimesh.load(ROOT/'data_real/chair/original_DesignSpace.stl',force='mesh')
    center=env.bounds.mean(axis=0);radius=float(np.linalg.norm(env.extents))*1.5
    fit=float(env.extents.max())/2
    rows=[r for r in ROWS if r[1].exists()]
    size,header=400,36
    sheet=Image.new('RGB',(size*len(VIEWS),(size+header)*len(rows)),'white')
    draw=ImageDraw.Draw(sheet)
    records=[]
    for i,(name,path,audit_path,thick,proj) in enumerate(rows):
        mesh=trimesh.load(path,force='mesh')
        check=json.loads(audit_path.read_text()) if audit_path.exists() else {}
        record={'name':name,'mesh':str(path),'sp_thick_w':thick,'sp_image_proj_w':proj,
            'bc_pass':check.get('bc_geometry_pass'),
            'shape_pass':check.get('shape_gate_pass'),
            'components':check.get('mesh_components'),
            'largest_fraction':check.get('largest_component_volume_fraction'),
            'silhouettes':{}}
        for j,(view,elev,azim) in enumerate(VIEWS):
            eye,up=camera_from_elev_azim(center,radius,elev,azim)
            rendered=Image.fromarray(render_lit(mesh,eye,center,up,size=size,
                fit_extent=fit,margin=1.15,color=(.47,.52,.56))).convert('RGB')
            x,y=j*size,i*(size+header)
            sheet.paste(rendered,(x,y+header))
            draw.text((x+8,y+10),f'{name}: {view}',fill='#20252b')
            if view in TARGETS:
                target=Image.open(TARGETS[view]).convert('RGB').resize((size,size))
                record['silhouettes'][view]=silhouette_scores(rendered,target)
        records.append(record)
    sheet.save(OUT/'comparison.png')
    (OUT/'metrics.json').write_text(json.dumps(records,indent=2)+'\n')
    table=''.join('<tr><td>'+html.escape(r['name'])+'</td><td><a href="/'+
        r['mesh'].removeprefix(str(ROOT)+'/')+'">OBJ</a></td><td>'+str(r['bc_pass'])+
        '</td><td>'+str(r['components'])+'</td><td>'+
        '/'.join(f'{r["silhouettes"][v]["iou"]:.3f}' for v in ('front','right','top'))+
        '</td></tr>' for r in records)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair sparse style ablation</title>
<style>body{font:16px system-ui;margin:30px;background:#f4f6f8;color:#18232d}img{max-width:100%}table{border-collapse:collapse;background:white}td,th{border:1px solid #c8d0da;padding:8px}a{color:#075f9e}</style>
<h1>Chair sparse style: thickness and image guidance</h1><p>Same dense cache, seed, 13 mm BC dilation, envelope and FEA-off settings. Only sparse thickness weight and new sparse image-projection weight vary. Camera-matched multi-view silhouette IoU is front/right/top.</p>
<a href="comparison.png"><img src="comparison.png"></a><table><tr><th>Variant</th><th>OBJ</th><th>BC pass</th><th>Components</th><th>Front/right/top IoU</th></tr>'''+table+'</table></html>'
    (OUT/'index.html').write_text(page)
    print(OUT/'index.html')


if __name__=='__main__':main()
