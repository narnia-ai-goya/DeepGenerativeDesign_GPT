#!/usr/bin/env python3
"""Render the current physical chair BC and explicit force vectors for paper figures."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault('PYVISTA_OFF_SCREEN','true')
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv
import trimesh

from make_chair_domain import ROOT

BASE=ROOT/'experiments/chair/sofa_style_2026-09-28'
SPEC=BASE/'single_view_spec_2026-10-03'
DEST=BASE/'tapered_examples_2026-10-03/qd_round_01/figure/source_images'
OUTPUT=DEST/'01_bc_preview.png'
LOAD_COLOR='#7856a0'


def mesh(name: str)->pv.PolyData:
    m=trimesh.load(SPEC/name,force='mesh',process=False)
    return pv.wrap(m)


def main()->None:
    DEST.mkdir(parents=True,exist_ok=True)
    pl=pv.Plotter(off_screen=True,window_size=(1000,760))
    pl.set_background('white')
    pl.enable_anti_aliasing('msaa')
    # The historical chair domain supplies a recognizable silhouette; the
    # current broader envelope and BC meshes come from the registered spec.
    reference=trimesh.load(ROOT/'data_real/chair/original_DesignSpace.stl',force='mesh',process=False)
    pl.add_mesh(pv.wrap(reference),color='#9baab1',opacity=.18,smooth_shading=True)
    pl.add_mesh(mesh('envelope.stl'),color='#7eabb7',opacity=.075,
                show_edges=True,edge_color='#9bb7bf',line_width=.45)
    pl.add_mesh(mesh('fixed.stl'),color='#a94c42',opacity=1,smooth_shading=True)
    pl.add_mesh(mesh('seat_load.stl'),color=LOAD_COLOR,opacity=1,smooth_shading=True)
    pl.add_mesh(mesh('back_load.stl'),color=LOAD_COLOR,opacity=1,smooth_shading=True)
    # Physical coordinates: seat -Z (800 N); backrest +Y (200 N).
    for x in (-.095,0.,.095):
        arrow=pv.Arrow(start=(x,-.0175,.805),direction=(0,0,-1),
                       scale=.225,shaft_radius=.035,tip_radius=.12,tip_length=.28)
        pl.add_mesh(arrow,color=LOAD_COLOR,lighting=False)
    for x in (-.045,.045):
        arrow=pv.Arrow(start=(x,.035,.775),direction=(0,1,0),
                       scale=.17,shaft_radius=.045,tip_radius=.14,tip_length=.29)
        pl.add_mesh(arrow,color=LOAD_COLOR,lighting=False)
    pl.camera_position=[(.84,-1.22,1.0),(0.,.01,.46),(0,0,1)]
    pl.camera.parallel_projection=True
    pl.camera.parallel_scale=.55
    pl.add_light(pv.Light(position=(1,-1,2),focal_point=(0,0,.5),intensity=.8))
    pl.add_light(pv.Light(position=(-1,.5,1.4),focal_point=(0,0,.5),intensity=.45))
    rendered=Image.fromarray(pl.screenshot(return_img=True,transparent_background=True)).convert('RGBA')
    pl.close()
    canvas=Image.new('RGBA',(1000,820),(0,0,0,0))
    canvas.paste(rendered,(0,0))
    draw=ImageDraw.Draw(canvas)
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,25) if Path(fontpath).exists() else ImageFont.load_default()
    for x,color,label in [(25,'#a94c42','4 fixed feet'),(330,LOAD_COLOR,'800 N  -Z'),
                          (610,LOAD_COLOR,'200 N  +Y')]:
        draw.rounded_rectangle((x,772,x+32,802),radius=5,fill=color)
        draw.text((x+43,773),label,font=font,fill='#203440')
    canvas.save(OUTPUT)
    metadata={'source_spec':str(SPEC/'specification.json'),
              'source_meshes':{name:str(SPEC/f'{name}.stl')
                               for name in ('envelope','fixed','seat_load','back_load')},
              'reference_silhouette':str(ROOT/'data_real/chair/original_DesignSpace.stl'),
              'forces_physical':{'seat':{'newtons':800,'direction_xyz':[0,0,-1]},
                                 'backrest':{'newtons':200,'direction_xyz':[0,1,0]}},
              'load_color':LOAD_COLOR,
              'output':str(OUTPUT)}
    (DEST/'01_bc_preview_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(OUTPUT)


if __name__=='__main__':main()
