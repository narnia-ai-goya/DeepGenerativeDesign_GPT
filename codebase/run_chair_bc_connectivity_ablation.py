#!/usr/bin/env python3
"""Keep chair BC occupancy and test explicit fix-to-load connectivity at dense stage."""
from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
from scipy.ndimage import label

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SOURCE = BASE / 'consistent_reference_2026-09-28'
OUT = SOURCE / 'bc_connectivity_ablation'
CASES = ('image_bc_connected', 'shape_bc_connected',
         'shape_bc_pw1', 'shape_bc_pw1_reach05',
         'shape_bc_local', 'shape_bc_pw03')


def tube(x, y, z, p, q, radius):
    v = np.stack([x-p[0],y-p[1],z-p[2]],axis=-1)
    d = np.asarray(q,float)-p
    t = np.clip(np.sum(v*d,axis=-1)/np.dot(d,d),0,1)
    return np.sum((v-t[...,None]*d)**2,axis=-1) <= radius**2


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    data=np.load(BASE/'coherent_proxy_dense/voxel.npz')
    origin,pitch=data['origin'],data['pitch_xyz']
    coord=[origin[i]+(np.arange(64)+.5)*pitch[i] for i in range(3)]
    x,y,z=np.meshgrid(*coord,indexing='ij')
    corridor=np.zeros((64,64,64),bool)
    for sx in (-1,1):
        for sy in (-1,1):
            cx,cy=sx*.195,sy*.18
            corridor |= tube(x,y,z,(cx,cy,.045),(cx,cy,.465),.042)
            corridor |= tube(x,y,z,(cx,cy,.460),(sx*.125,sy*.095,.475),.042)
    corridor &= data['bracket'].astype(bool)
    mask_path=OUT/'fix_to_seat_corridors.npz'
    np.savez_compressed(mask_path,mask=corridor)
    structure=np.zeros((3,3,3),dtype=np.uint8)
    structure[1,1,:]=1; structure[1,:,1]=1; structure[:,1,1]=1
    connected,n=label(corridor|data['bc'].astype(bool),structure=structure)
    bc_labels=np.unique(connected[data['bc'].astype(bool)])
    bc_labels=bc_labels[bc_labels>0]
    metrics={'corridor_voxels':int(corridor.sum()),
             'inside_envelope':bool(np.all(~corridor|data['bracket'])),
             'corridor_plus_bc_components':int(n),
             'bc_component_labels':bc_labels.tolist(),
             'path':str(mask_path)}
    if len(bc_labels)!=1:
        raise ValueError(f'corridor does not connect all BC islands: {metrics}')
    (OUT/'corridor_validation.json').write_text(json.dumps(metrics,indent=2)+'\n')
    for case in CASES:
        case_dir=OUT/case
        case_dir.mkdir(exist_ok=True)
        baseline='image_only' if case.startswith('image_') else 'shape_guided'
        cfg=json.loads((SOURCE/baseline/'config_dense.json').read_text())
        cfg['name']='chair_consistent_reference_'+case
        mesh=cfg['stages']['mesh']
        local=case=='shape_bc_local'
        path_only=case=='shape_bc_pw03'
        mesh.update(cw=5.0 if local else (0.0 if path_only else 30.0),
                    cw_warmup=0.2 if local else 0.0,
                    pw=0.3 if (local or path_only) else (1.0 if 'pw1' in case else 3.5),
                    load_path_mask=str(mask_path),
                    reach_threshold=.5 if (case.endswith('reach05') or local) else .1,
                    reach_kernel=9 if local else 41,
                    save_dense_cache=str(case_dir/'dense_cache.npz'),
                    load_dense_cache=None,skip_sparse=True,fea_w=0.0)
        (case_dir/'config_dense.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))


def run(case,gpu):
    case_dir=OUT/case
    command=[str(PYTHON),str(GENERATOR),'--config',str(case_dir/'config_dense.json'),
             '--target-dir',str(SOURCE/'input_lr162'),'--out',str(case_dir/'dense')]
    with (case_dir/'dense.log').open('w') as log:
        result=subprocess.run(command,cwd=ROOT,env=generation_env(gpu),
                              stdout=log,stderr=subprocess.STDOUT)
    (case_dir/'run.json').write_text(json.dumps({'command':command,'gpu':gpu,
       'exit_code':result.returncode,'mesh':str(case_dir/'dense/mesh_dense.obj')},indent=2)+'\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    print(case_dir/'dense/mesh_dense.obj')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--case',choices=CASES,default=CASES[0])
    parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    else: run(args.case,args.gpu)
