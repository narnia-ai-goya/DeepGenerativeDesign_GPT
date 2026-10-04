#!/usr/bin/env python3
"""Render and audit the chair MMC scaffold and sparse diagnostic variants."""
from __future__ import annotations

import html
import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from make_chair_domain import ROOT
from validate_chair_bc_geometry import audit

sys.path.insert(0,str(ROOT/'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit

SOURCE=ROOT/'experiments/chair/sofa_style_2026-09-28/backrest_bc_multi_examples_2026-09-29/diagonal_braced'
OUT=ROOT/'experiments/chair/sofa_style_2026-09-28/mmc_dense_pilot_2026-09-29'
BC=ROOT/'experiments/chair/sofa_style_2026-09-28/consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
VIEWS=(('oblique',25,35),('front',15,0),('side',15,90),('top',85,0))
ROWS=(
    ('MMC component scaffold',OUT/'mmc_scaffold.obj'),
    ('previous dense',SOURCE/'dense_pw2/mesh_dense.obj'),
    ('MMC-guided dense',OUT/'dense/mesh_dense.obj'),
    ('MMC-guided dense soft',OUT/'dense_soft/mesh_dense.obj'),
    ('previous sparse',SOURCE/'sparse_pw2_d13/generation/mesh.obj'),
    ('baseline before refiner',OUT/'baseline/generation/mesh_pre_refiner_world.obj'),
    ('baseline after refiner',OUT/'baseline/generation/mesh.obj'),
    ('no thickness before refiner',OUT/'no_thickness/generation/mesh_pre_refiner_world.obj'),
    ('no thickness after refiner',OUT/'no_thickness/generation/mesh.obj'),
)


def main():
    bc=np.load(BC)
    for name in ('baseline','no_thickness'):
        folder=OUT/name/'generation'
        source=folder/'mesh_pre_refiner.obj'
        dest=folder/'mesh_pre_refiner_world.obj'
        if source.exists() and not dest.exists():
            pre=trimesh.load(source,force='mesh')
            voxel_idx=(pre.vertices+1.)*512./2.
            pre.vertices=bc['origin']+(voxel_idx+.5)*(bc['pitch_xyz']/8.)
            pre.export(dest)
    envelope=trimesh.load(ROOT/'data_real/chair/original_DesignSpace.stl',force='mesh')
    center=envelope.bounds.mean(axis=0)
    radius=float(np.linalg.norm(envelope.extents))*1.5
    fit=float(envelope.extents.max())/2
    size=340; header=36
    available=[(name,path) for name,path in ROWS if path.exists()]
    image=Image.new('RGB',(size*len(VIEWS),(size+header)*len(available)),'white')
    draw=ImageDraw.Draw(image)
    records=[]
    for row,(name,path) in enumerate(available):
        mesh=trimesh.load(path,force='mesh')
        result=audit(path,BC)
        records.append({'name':name,'path':str(path),'vertices':len(mesh.vertices),
                        'faces':len(mesh.faces),'bc_pass':result['bc_geometry_pass'],
                        'bc_coverage':{k:v['coverage'] for k,v in result['regions'].items()},
                        'components':result['mesh_components'],
                        'largest_fraction':result['largest_component_volume_fraction']})
        for col,(view,elev,azim) in enumerate(VIEWS):
            eye,up=camera_from_elev_azim(center,radius,elev,azim)
            tile=Image.fromarray(render_lit(mesh,eye,center,up,size=size,
                    fit_extent=fit,margin=1.15,color=(.49,.54,.58))).convert('RGB')
            x,y=col*size,row*(size+header)
            image.paste(tile,(x,y+header))
            draw.text((x+7,y+8),f'{name} / {view}',fill='#20252b')
    image.save(OUT/'comparison.png')
    (OUT/'metrics.json').write_text(json.dumps(records,indent=2)+'\n')
    body='\n'.join(f'<tr><td>{html.escape(r["name"])}</td><td><a href="/{r["path"].removeprefix(str(ROOT)+"/")}">mesh</a></td><td>{r["bc_pass"]}</td><td>{r["components"]}</td><td>{r["largest_fraction"]:.4f}</td></tr>' for r in records)
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair MMC pilot</title>
<style>body{{font:16px Arial;margin:30px;background:#f7f8fa;color:#17202a}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccd3da;padding:8px}}a{{color:#0065ae}}</style>
<h1>Chair MMC scaffold and sparse refiner diagnostic</h1>
<p>Same diagonal-braced images and boundary conditions. Geometry pilot only; FEA is off. The MMC row is a fixed straight-component scaffold used as a dense shape prior, not an optimized MMC solver.</p>
<p><strong>Finding:</strong> The straight MMC scaffold covers all six BC regions and is connected. Both MMC-guided dense runs cover BC, but the strong prior creates protrusions at the backrest/seat joints; the softer prior loses much of the backrest frame. Sparse pre/post-refiner silhouettes are nearly unchanged. Disabling sparse thickness guidance does not remove the organic style and adds a visible side hole and detached components. These results do not support replacing the current dense generator with a simple MMC shape anchor.</p>
<img src="comparison.png" alt="Rendered comparison"><table><tr><th>Variant</th><th>OBJ</th><th>BC pass</th><th>Components</th><th>Largest volume fraction</th></tr>{body}</table></html>'''
    (OUT/'index.html').write_text(page)
    print(OUT/'index.html')
    print(OUT/'comparison.png')


if __name__=='__main__':main()
