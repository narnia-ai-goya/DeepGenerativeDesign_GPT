#!/usr/bin/env python3
"""Regenerate the three-view chair mesh preview with native RGBA rendering."""
from __future__ import annotations

import os
import sys

os.environ.setdefault('PYVISTA_OFF_SCREEN','true')
import numpy as np
from PIL import Image, ImageDraw
import pyvista as pv
import trimesh

from make_chair_domain import ROOT

sys.path.insert(0,str(ROOT/'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim  # noqa: E402

BASE=ROOT/'experiments/chair/sofa_style_2026-09-28/tapered_examples_2026-10-03'
MESH=BASE/'final_meshes/tapered_original.obj'
OUT=BASE/'qd_round_01/figure/source_images/05_tapered_original_mesh_preview.png'


def render(mesh:trimesh.Trimesh,elev:float,azim:float,target:np.ndarray,extent:float)->Image.Image:
    eye,up=camera_from_elev_azim(target,2.0,elev,azim)
    pl=pv.Plotter(off_screen=True,window_size=(350,350))
    pl.enable_anti_aliasing('msaa')
    pl.set_background('white')
    pl.remove_all_lights()
    pl.add_light(pv.Light(position=tuple(eye+np.array([0,0,1])),focal_point=tuple(target),
                          intensity=.65,light_type='scene light'))
    pl.add_light(pv.Light(position=tuple(eye),focal_point=tuple(target),
                          intensity=.5,light_type='headlight'))
    pl.add_mesh(pv.wrap(mesh).compute_normals(point_normals=True,cell_normals=False,
                                              auto_orient_normals=True),
                color=(.62,.66,.7),smooth_shading=True,ambient=.25,diffuse=.65,
                specular=.15,specular_power=20)
    pl.camera_position=[eye.tolist(),target.tolist(),up.tolist()]
    pl.camera.parallel_projection=True
    pl.camera.parallel_scale=extent*1.15
    im=Image.fromarray(pl.screenshot(return_img=True,transparent_background=True)).convert('RGBA')
    pl.close()
    return im


def main()->None:
    mesh=trimesh.load(MESH,force='mesh',process=False)
    canvas=Image.new('RGBA',(1080,370),(0,0,0,0))
    center=np.asarray([0.,.01,.46])
    views=[('front',15,0,center,.56),('side',15,90,center,.56),
           ('back detail',10,15,np.asarray([0.,.17,.77]),.20)]
    draw=ImageDraw.Draw(canvas)
    for i,(label,elev,azim,target,extent) in enumerate(views):
        canvas.alpha_composite(render(mesh,elev,azim,target,extent),(i*360,0))
        draw.text((i*360+8,348),label,fill='#17212b')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(OUT)
    print(OUT)


if __name__=='__main__':main()
