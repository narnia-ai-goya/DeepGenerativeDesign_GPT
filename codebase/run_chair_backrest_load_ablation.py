#!/usr/bin/env python3
"""Add a separate backrest load interface to the chair geometry-only BC mask."""
from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import trimesh
from scipy.ndimage import label

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

BASE=ROOT/'experiments/chair/sofa_style_2026-09-28'
SOURCE=BASE/'consistent_reference_2026-09-28'
OUT=SOURCE/'backrest_load_ablation'


def tube(x,y,z,p,q,radius):
    v=np.stack([x-p[0],y-p[1],z-p[2]],axis=-1)
    d=np.asarray(q,float)-p
    t=np.clip(np.sum(v*d,axis=-1)/np.dot(d,d),0,1)
    return np.sum((v-t[...,None]*d)**2,axis=-1)<=radius**2


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    src=np.load(BASE/'coherent_proxy_dense/voxel.npz')
    arrays={k:np.array(src[k]) for k in src.files}
    origin,pitch=arrays['origin'],arrays['pitch_xyz']
    axes=[origin[i]+(np.arange(64)+.5)*pitch[i] for i in range(3)]
    x,y,z=np.meshgrid(*axes,indexing='ij')
    # Separate load interface on the *rear upper crossbar*. The physical
    # direction would be +Y (occupant leaning backward), but no FEA is run here.
    patch=(np.abs(x)<=.16)&(y>=.18)&(y<=.245)&(z>=.785)&(z<=.845)
    if not np.all(~patch|arrays['bracket']):
        raise ValueError('backrest load patch exits envelope')
    if np.any(patch&arrays['fix']) or np.any(patch&arrays['load']):
        raise ValueError('backrest patch overlaps a prior BC')
    proto=np.load(BASE/'coherent_proxy_dense/prototype.npz')['prototypes'][0]>.5
    if not np.all(~patch|proto):
        raise ValueError('backrest patch lies outside coherent reference mesh')
    arrays['back_load']=patch
    arrays['bc']=arrays['bc'].astype(bool)|patch
    arrays['design']=arrays['bracket'].astype(bool)&~arrays['bc']
    np.savez_compressed(OUT/'voxel.npz',**arrays)
    box=trimesh.creation.box(extents=(.32,.065,.060))
    box.apply_translation((0,(.18+.245)/2,(.785+.845)/2))
    box.export(OUT/'backrest_load.stl')
    baseline=json.loads((SOURCE/'shape_guided/config_dense.json').read_text())
    baseline['name']='chair_consistent_reference_backrest_load_geometry_only'
    mesh=baseline['stages']['mesh']
    mesh.update(bc_proper=str(OUT/'voxel.npz'),bracket_occ=str(OUT/'voxel.npz'),
                save_dense_cache=str(OUT/'dense_cache.npz'),load_dense_cache=None,
                skip_sparse=True,fea_w=0.0,sp_fea_w=0.0)
    (OUT/'config_dense.json').write_text(json.dumps(baseline,indent=2)+'\n')
    # Optional support condition: two rear uprights and their top crossbar.
    # This mask is a target path, not additional fixed geometry.
    corridor=np.zeros_like(patch)
    for sx in (-1,1):
        corridor|=tube(x,y,z,(sx*.215,.20,.455),(sx*.215,.20,.825),.052)
        corridor|=tube(x,y,z,(sx*.215,.20,.465),(sx*.125,.095,.475),.045)
    corridor|=tube(x,y,z,(-.215,.21,.815),(.215,.21,.815),.040)
    corridor&=arrays['bracket'].astype(bool)
    conn,n_conn=label(corridor|arrays['load'].astype(bool)|patch)
    load_labels=set(np.unique(conn[arrays['load'].astype(bool)]))-{0}
    back_labels=set(np.unique(conn[patch]))-{0}
    if not load_labels&back_labels:
        raise ValueError('rear support corridor does not join seat and back load patches')
    np.savez_compressed(OUT/'back_support_corridor.npz',mask=corridor)
    leg_path=np.load(SOURCE/'bc_connectivity_ablation/fix_to_seat_corridors.npz')['mask'].astype(bool)
    combined=(corridor|leg_path)&arrays['bracket'].astype(bool)
    np.savez_compressed(OUT/'all_support_corridors.npz',mask=combined)
    for case in ('back_cw_local','back_support_pw1'):
        sub=OUT/case
        sub.mkdir(exist_ok=True)
        cfg=json.loads(json.dumps(baseline))
        cfg['name']='chair_backrest_load_'+case
        m=cfg['stages']['mesh']
        m.update(save_dense_cache=str(sub/'dense_cache.npz'),load_dense_cache=None,
                 cw=5.0 if case=='back_cw_local' else 0.0,
                 cw_warmup=.2,reach_threshold=.5,reach_kernel=9,
                 pw=1.0 if case=='back_support_pw1' else 0.0,
                 load_path_mask=str(OUT/'back_support_corridor.npz') if case=='back_support_pw1' else None)
        (sub/'config_dense.json').write_text(json.dumps(cfg,indent=2)+'\n')
    for case,pw,vw,target in [('all_support_vw0',.5,0.0,.25),
                               ('all_support_v35',1.0,50.0,.35)]:
        sub=OUT/case; sub.mkdir(exist_ok=True)
        cfg=json.loads(json.dumps(baseline))
        cfg['name']='chair_backrest_load_'+case
        m=cfg['stages']['mesh']
        m.update(save_dense_cache=str(sub/'dense_cache.npz'),load_dense_cache=None,
                 cw=0.0,pw=pw,load_path_mask=str(OUT/'all_support_corridors.npz'),
                 vw=vw,vol_target=target)
        (sub/'config_dense.json').write_text(json.dumps(cfg,indent=2)+'\n')
    metrics={'back_load_voxels':int(patch.sum()),
             'previous_bc_voxels':int(src['bc'].sum()),
             'new_bc_voxels':int(arrays['bc'].sum()),
             'all_inside_envelope':True,'all_inside_reference':True,
             'patch_bounds_m':{'x':[-.16,.16],'y':[.18,.245],'z':[.785,.845]},
             'intended_future_force_direction':'+Y',
             'fea_used_in_this_ablation':False,
             'backrest_load_stl':str(OUT/'backrest_load.stl'),
             'voxel':str(OUT/'voxel.npz'),
             'back_support_corridor_voxels':int(corridor.sum()),
             'all_support_corridor_voxels':int(combined.sum())}
    (OUT/'patch_validation.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


def run(gpu,case):
    sub=OUT if case=='base' else OUT/case
    command=[str(PYTHON),str(GENERATOR),'--config',str(sub/'config_dense.json'),
             '--target-dir',str(SOURCE/'input_lr162'),'--out',str(sub/'dense')]
    with (sub/'dense.log').open('w') as log:
        result=subprocess.run(command,cwd=ROOT,env=generation_env(gpu),
                              stdout=log,stderr=subprocess.STDOUT)
    (sub/'run.json').write_text(json.dumps({'command':command,'gpu':gpu,
        'exit_code':result.returncode,'mesh':str(sub/'dense/mesh_dense.obj')},indent=2)+'\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    print(sub/'dense/mesh_dense.obj')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--case',choices=['base','back_cw_local','back_support_pw1',
                                           'all_support_vw0','all_support_v35'],default='base')
    args=parser.parse_args()
    prepare() if args.command=='prepare' else run(args.gpu,args.case)
