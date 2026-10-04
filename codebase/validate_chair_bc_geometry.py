#!/usr/bin/env python3
"""Audit physical chair BC coverage, fix-to-load connection and key geometry."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import label

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codebase/code'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402


def audit(mesh_path:Path,bc_path:Path,prototype_path:Path|None=None)->dict:
    data=np.load(bc_path)
    mesh=trimesh.load(mesh_path,force='mesh')
    occ=voxel_centers_inside(mesh,64,data['origin'],data['pitch_xyz'])
    labels,n_voxel=label(occ)
    origin,pitch=data['origin'],data['pitch_xyz']
    axes=[origin[i]+(np.arange(64)+.5)*pitch[i] for i in range(3)]
    x,y,_=np.meshgrid(*axes,indexing='ij')
    masks={'seat_load':data['load'].astype(bool)}
    if 'back_load' in data.files:
        masks['backrest_load']=data['back_load'].astype(bool)
    for sx in (-1,1):
        for sy in (-1,1):
            masks[f'foot_x{sx:+d}_y{sy:+d}']=(data['fix'].astype(bool)&(sx*x>0)&(sy*y>0))
    regions={}
    common=None
    for name,mask in masks.items():
        inside=mask&occ
        vals,counts=np.unique(labels[inside],return_counts=True)
        positive={int(v):int(c) for v,c in zip(vals,counts) if v>0}
        regions[name]={'target_voxels':int(mask.sum()),
                       'covered_voxels':int(inside.sum()),
                       'coverage':round(float(inside.sum()/max(1,mask.sum())),4),
                       'component_labels':positive}
        current=set(positive)
        common=current if common is None else common&current
    parts=mesh.split(only_watertight=False)
    volumes=[abs(float(p.volume)) for p in parts]
    largest_fraction=max(volumes,default=0)/max(sum(volumes),1e-12)
    bc_covered=all(r['coverage']>=.95 for r in regions.values())
    bc_connected=bool(common)
    dominant_body=largest_fraction>=.99
    result={'mesh':str(mesh_path.resolve()),'bc':str(bc_path.resolve()),
            'regions':regions,'all_bc_covered_95pct':bc_covered,
            'all_bc_same_6_connected_component':bc_connected,
            'mesh_components':len(parts),
            'largest_component_volume_fraction':round(float(largest_fraction),6),
            'dominant_body_99pct':dominant_body,
            'bc_geometry_pass':bc_covered and bc_connected and dominant_body}
    if prototype_path:
        prototype=np.load(prototype_path)['prototypes'][0]>.5
        z=axes[2][None,None,:]
        top_bar=(np.abs(x)<.16)&(y>.15)&(y<.27)&(z>.75)&(z<.89)&prototype
        match=int((occ&top_bar).sum())
        feature_coverage=match/max(1,int(top_bar.sum()))
        result['top_bar']={'target_voxels':int(top_bar.sum()),
                           'covered_voxels':match,
                           'coverage':round(float(feature_coverage),4)}
        result['shape_gate_pass']=result['bc_geometry_pass'] and feature_coverage>=.5
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mesh',type=Path)
    parser.add_argument('--bc',type=Path,default=ROOT/'data_real/chair/voxel.npz')
    parser.add_argument('--prototype',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--require-shape',action='store_true')
    args=parser.parse_args()
    result=audit(args.mesh,args.bc,args.prototype)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['bc_geometry_pass'] or (args.require_shape and not result.get('shape_gate_pass',False)):
        raise SystemExit(1)
