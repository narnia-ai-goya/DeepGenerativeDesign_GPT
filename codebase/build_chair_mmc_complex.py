#!/usr/bin/env python3
"""Build and render an armchair-like explicit straight-component chair pilot."""
from __future__ import annotations

import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from skimage.measure import marching_cubes

from make_chair_domain import ROOT
from run_chair_mmc_dense_pilot import BC, bar
from validate_chair_bc_geometry import audit

sys.path.insert(0, str(ROOT/'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit

OUT=ROOT/'experiments/chair/sofa_style_2026-09-28/mmc_complex_chair_2026-09-29'


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    data=np.load(BC)
    ijk=np.indices((64,64,64)).reshape(3,-1).T
    world=data['origin']+(ijk+.5)*data['pitch_xyz']
    mask=np.zeros(len(world),dtype=bool)
    components=[]

    def add(name,start,end,width,depth):
        nonlocal mask
        mask |= bar(world,start,end,width,depth)
        components.append({'name':name,'start':start,'end':end,'width':width,'depth':depth})

    for x in (-.225,.225):
        for y in (-.205,.205):
            add(f'leg_{x:+.3f}_{y:+.3f}',(x,y,.030),(x,y,.505),.068,.068)
    add('seat_plate',(-.215,0,.480),(.215,0,.480),.430,.048)
    add('front_apron',(-.225,-.205,.438),(.225,-.205,.438),.073,.070)
    add('rear_apron',(-.225,.205,.438),(.225,.205,.438),.073,.070)
    for x in (-.225,.225):
        add(f'seat_side_{x:+.3f}',(x,-.205,.450),(x,.205,.450),.065,.070)
        add(f'back_post_{x:+.3f}',(x,.205,.475),(x,.205,.865),.068,.068)
        # Side triangular struts tie the front and rear legs to the seat frame.
        add(f'front_side_diagonal_{x:+.3f}',(x,-.205,.235),(x,.205,.455),.044,.048)
        add(f'rear_side_diagonal_{x:+.3f}',(x,.205,.235),(x,-.205,.455),.044,.048)
        # Armrest and the front vertical support create a second horizontal tier.
        add(f'armrest_{x:+.3f}',(x,-.17,.635),(x,.205,.635),.065,.055)
        add(f'arm_post_{x:+.3f}',(x,-.17,.485),(x,-.17,.650),.053,.053)
    add('back_top',(-.225,.205,.825),(.225,.205,.825),.082,.072)
    add('back_mid',(-.225,.205,.690),(.225,.205,.690),.055,.055)
    add('back_lower',(-.225,.205,.555),(.225,.205,.555),.055,.055)
    # Two shallow braces make the back visually richer without filling it in.
    add('back_diagonal_left',(-.225,.205,.580),(0,.205,.795),.043,.045)
    add('back_diagonal_right',(.225,.205,.580),(0,.205,.795),.043,.045)

    mask=mask.reshape((64,64,64))
    mask |= data['bc'].astype(bool)
    mask &= data['bracket'].astype(bool)|data['bc'].astype(bool)
    np.savez_compressed(OUT/'components_occupancy.npz',prototypes=mask[None].astype(np.float32))
    vertices,faces,_,_=marching_cubes(np.pad(mask,1),.5)
    vertices=data['origin']+(vertices-.5)*data['pitch_xyz']
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    mesh.export(OUT/'mmc_complex_chair.obj')
    (OUT/'components.json').write_text(json.dumps(components,indent=2)+'\n')
    check=audit(OUT/'mmc_complex_chair.obj',BC)
    (OUT/'bc_audit.json').write_text(json.dumps(check,indent=2)+'\n')
    return mesh,check


def render(mesh):
    env=trimesh.load(ROOT/'data_real/chair/original_DesignSpace.stl',force='mesh')
    center=env.bounds.mean(axis=0)
    radius=float(np.linalg.norm(env.extents))*1.5
    extent=float(env.extents.max())/2
    views=(('OBLIQUE',25,35),('FRONT',15,0),('SIDE',15,90),('TOP',85,0))
    size,header=720,42
    sheet=Image.new('RGB',(2*size,2*(size+header)),'white')
    draw=ImageDraw.Draw(sheet)
    for i,(name,elev,azim) in enumerate(views):
        eye,up=camera_from_elev_azim(center,radius,elev,azim)
        tile=Image.fromarray(render_lit(mesh,eye,center,up,size=size,
            fit_extent=extent,margin=1.15,color=(.48,.54,.60))).convert('RGB')
        x=(i%2)*size;y=(i//2)*(size+header)
        sheet.paste(tile,(x,y+header))
        draw.text((x+20,y+14),name,fill='#1a2430')
    sheet.save(OUT/'mmc_complex_chair_render.png')


if __name__=='__main__':
    result,check=build()
    render(result)
    print(json.dumps({'obj':str(OUT/'mmc_complex_chair.obj'),
       'png':str(OUT/'mmc_complex_chair_render.png'),
       'bc_pass':check['bc_geometry_pass'],'components':check['mesh_components']},indent=2))
