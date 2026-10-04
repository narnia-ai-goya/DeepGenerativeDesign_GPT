#!/usr/bin/env python3
"""Render the complex image-first chair through the geometric stages."""
from __future__ import annotations

import html
import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from make_chair_domain import ROOT
from report_chair_prefea_stages import silhouette_scores

sys.path.insert(0,str(ROOT/'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim,render_lit

OUT=ROOT/'experiments/chair/complex_truss_image_to_3d_2026-09-29'
CASE=OUT/'main'
INPUT=ROOT/'experiments/chair/image_concepts_2026-09-27/complex_truss_armchair/multiview_registered'
VIEWS=(('hero',28,40),('front',15,0),('right',15,90),('top',85,0))
IMAGES={'hero':INPUT.parent/'generated_native.png','front':INPUT/'input/v00_front_lo.png',
        'right':INPUT/'input/v02_right_lo.png','top':INPUT/'input/v_top.png'}
STAGES=(('prototype',CASE/'prototype.obj'),('dense base',CASE/'dense/mesh_dense.obj'),
        ('dense pw2',CASE/'dense_pw2/mesh_dense.obj'),
        ('dense stronger image',CASE/'dense_strong_image/mesh_dense.obj'),
        ('sparse base',CASE/'sparse_d13/generation/mesh.obj'),
        ('sparse pw2',CASE/'sparse_pw2_d13/generation/mesh.obj'))


def main():
    env=trimesh.load(ROOT/'data_real/chair/original_DesignSpace.stl',force='mesh')
    center=env.bounds.mean(axis=0)
    radius=float(np.linalg.norm(env.extents))*1.5
    fit=float(env.extents.max())/2
    rows=[(name,path) for name,path in STAGES if path.exists()]
    size,header=440,36
    sheet=Image.new('RGB',(size*len(VIEWS),(size+header)*(len(rows)+1)),'white')
    draw=ImageDraw.Draw(sheet)
    for j,(view,_,_) in enumerate(VIEWS):
        img=Image.open(IMAGES[view]).convert('RGB').resize((size,size),Image.Resampling.LANCZOS)
        sheet.paste(img,(j*size,header))
        draw.text((j*size+10,10),f'input: {view}',fill='#20252b')
    metrics=[]
    for i,(name,path) in enumerate(rows,start=1):
        mesh=trimesh.load(path,force='mesh')
        record={'stage':name,'mesh':str(path),'vertices':len(mesh.vertices),
                'faces':len(mesh.faces),'components':len(mesh.split(only_watertight=False)),
                'watertight':bool(mesh.is_watertight),'silhouettes':{}}
        audit_path=CASE/('dense_bc_audit.json' if name=='dense base' else
                         'dense_pw2_bc_audit.json' if name=='dense pw2' else
                         'dense_strong_image_bc_audit.json' if name=='dense stronger image' else
                         'sparse_d13/bc_audit.json' if name=='sparse base' else
                         'sparse_pw2_d13/bc_audit.json' if name=='sparse pw2' else
                         'prototype_metrics.json')
        if audit_path.exists():
            check=json.loads(audit_path.read_text())
            record['bc_pass']=check.get('bc_geometry_pass')
            record['shape_gate_pass']=check.get('shape_gate_pass')
        for j,(view,elev,azim) in enumerate(VIEWS):
            eye,up=camera_from_elev_azim(center,radius,elev,azim)
            rendered=Image.fromarray(render_lit(mesh,eye,center,up,size=size,
                fit_extent=fit,margin=1.15,color=(.47,.52,.56))).convert('RGB')
            x,y=j*size,i*(size+header)
            sheet.paste(rendered,(x,y+header))
            draw.text((x+10,y+10),f'{name}: {view}',fill='#20252b')
            if view!='hero':
                target=Image.open(IMAGES[view]).convert('RGB').resize((size,size))
                record['silhouettes'][view]=silhouette_scores(rendered,target)
        metrics.append(record)
    sheet.save(OUT/'comparison.png')
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    table='\n'.join('<tr><td>'+html.escape(r['stage'])+'</td><td><a href="/'+
        str((r['mesh']).removeprefix(str(ROOT)+'/'))+'">OBJ</a></td><td>'+str(r.get('bc_pass'))+
        '</td><td>'+str(r.get('shape_gate_pass'))+'</td><td>'+
        '/'.join(f'{r["silhouettes"][v]["iou"]:.3f}' for v in ('front','right','top'))+
        '</td></tr>' for r in metrics)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Complex image-first chair</title>
<style>body{font:16px system-ui;margin:30px;background:#f5f7f9;color:#18232d}img{max-width:100%}table{border-collapse:collapse;background:white}td,th{border:1px solid #ccd3dc;padding:8px}a{color:#075e9f}</style>
<h1>Complex truss armchair: image → dense → sparse</h1><p>BC geometry and multi-view silhouette comparison. FEA off. First hero column is illustrative; quantitative IoU is front/right/top only.</p>
<p><strong>Result:</strong> The selected sparse mesh passes the six-region BC/shape gate, but visually loses the backrest triangles and side diagonals, and has thin dangling fragments. Passing the BC gate does not imply image fidelity. This run is an unsuccessful image-to-3D reconstruction of the detailed concept.</p>
<a href="comparison.png"><img src="comparison.png"></a><table><tr><th>Stage</th><th>Mesh</th><th>BC</th><th>Shape gate</th><th>Front/right/top IoU</th></tr>'''+table+'</table></html>'
    (OUT/'index.html').write_text(page)
    print(OUT/'index.html')


if __name__=='__main__':main()
