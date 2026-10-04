#!/usr/bin/env python3
"""Test whether new image-QD chairs survive transfer to 3D and common BC/FEA."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy.ndimage import label
from skimage.measure import marching_cubes
import trimesh

from make_chair_domain import ROOT
from evaluate_chair_single_view_qd_2026_10_03 import evaluate,OUT,SPEC
from build_chair_single_view_qd_archive_2026_10_03 import features,cell_for,REPAIR_LIMIT
from evaluate_chair_single_view_qd_fea_2026_10_03 import evaluate_case

NAMES=('image_tall_rect_seed42','image_open_arch_seed42')
FENICS='/home/goya/miniconda3/envs/fenics/bin/python'
SOLVER=ROOT/'codebase/code/fenics_fea_bracket.py'


def fine_fea(row:dict,nodes:np.ndarray,indices:np.ndarray)->dict:
    case=OUT/row['id'];folder=case/'fea'
    occ=np.load(case/'realized_occupancy.npz')['occupied']
    rho=np.where(occ[indices[:,0],indices[:,1],indices[:,2]],1.,.001)
    np.save(folder/'nodes.npy',nodes);np.save(folder/'rho.npy',rho)
    result={'id':row['id']}
    for mode,domain,direction,force in [('seat',SPEC/'fea_domain','-z',800),
                                         ('back',SPEC/'fea_domain_back','y',200)]:
        work=case/'fea_refine035'/mode;work.mkdir(parents=True,exist_ok=True)
        command=[FENICS,str(SOLVER),'--domain-dir',str(domain),
                 '--density',str(folder/'rho.npy'),'--nodes',str(folder/'nodes.npy'),
                 '--output',str(work/'dc.npy'),'--mesh-cache',str(SPEC/'fea_domain/chair_035.msh'),
                 '--mesh-size','.035','--penal','2','--E0','1.0','--load-magnitude',str(force)]
        env={**os.environ,'LOAD_MODE':direction,'BC_SURFACE_DIST':'1','BC_DIST':'0.025',
             'FEA_MAX_NODE_MAP_DISTANCE':'0.03','OMP_NUM_THREADS':'1',
             'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        with (work/'fea.log').open('w') as log:
            process=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        text=(work/'fea.log').read_text(errors='replace')
        fixed=re.search(r'Dirichlet: (\d+) nodes',text)
        loaded=re.search(r'Load: (\d+) nodes',text)
        info=work/'dc_info.json'
        result[mode]={'exit_code':process.returncode,
                      'fixed_nodes':int(fixed.group(1)) if fixed else 0,
                      'load_nodes':int(loaded.group(1)) if loaded else 0,
                      'compliance_proxy':float(json.loads(info.read_text())['compliance'])
                      if process.returncode==0 and info.exists() else None}
    return result


def main()->None:
    spec=np.load(SPEC/'voxel.npz')
    envelope=spec['bracket'].astype(bool);bc=spec['bc'].astype(bool)
    baseline=np.load(OUT/'seed_42/realized_occupancy.npz')['occupied']
    structure=np.zeros((3,3,3),bool);structure[1,1,1]=True
    for axis in range(3):
        for sign in (-1,1):
            shift=[1,1,1];shift[axis]+=sign;structure[tuple(shift)]=True
    rows=[]
    for name in NAMES:
        case=OUT/name
        measured=evaluate(name)
        source=np.load(case/'occupancy.npz')['occupied'].astype(bool)
        realized=(source&envelope)|bc
        added=int((realized&~source).sum());removed=int((source&~realized).sum())
        repair=(added+removed)/max(1,int(realized.sum()))
        _,components=label(realized,structure=structure)
        descriptor=features(realized,spec)
        cell=cell_for(descriptor)
        volume=float(realized.sum()*np.prod(spec['pitch_xyz'])*1000)
        intersection=int((realized&baseline).sum());union=int((realized|baseline).sum())
        row={**measured,'realized_mass_liters':volume,'repair_fraction':repair,
             'realized_components_6conn':components,
             'realized_descriptor':{'side_open_fraction':descriptor[0],
                                    'backrest_taper_ratio':descriptor[1]},
             'realized_cell':cell,'realized_iou_to_baseline':intersection/union,
             'geometry_gate':bool(measured['watertight'] and components==1 and
                                  repair<=REPAIR_LIMIT and cell is not None)}
        np.savez_compressed(case/'realized_occupancy.npz',occupied=realized)
        verts,faces,_,_=marching_cubes(np.pad(realized,1),.5)
        verts=spec['origin']+(verts-.5)*spec['pitch_xyz']
        vox=trimesh.Trimesh(vertices=verts,faces=faces,process=False)
        vox.export(case/'realized_voxel.obj');vox.export(case/'realized_voxel.glb')
        trimesh.load(case/'aligned_main.obj',force='mesh',process=False).export(case/'source_mesh.glb')
        rows.append(row)
    indices=np.argwhere(envelope)
    nodes=spec['origin']+(indices+.5)*spec['pitch_xyz']
    valid=[row for row in rows if row['geometry_gate']]
    with ThreadPoolExecutor(max_workers=2) as pool:
        coarse=list(pool.map(lambda row:evaluate_case(row,nodes,indices),valid))
        fine=list(pool.map(lambda row:fine_fea(row,nodes,indices),valid))
    coarse_map={r['id']:r for r in coarse};fine_map={r['id']:r for r in fine}
    old=json.loads((OUT/'summary.json').read_text())
    baseline_ref=old['baseline_compliance_proxy']
    for row in rows:
        if row['id'] in fine_map:
            f=fine_map[row['id']]
            row['fea_035']=f
            if all(f[m]['exit_code']==0 and f[m]['fixed_nodes']>0 and
                   f[m]['load_nodes']>0 and f[m]['compliance_proxy'] for m in ('seat','back')):
                row['worst_compliance_ratio_to_baseline']=max(
                    f['seat']['compliance_proxy']/baseline_ref['seat'],
                    f['back']['compliance_proxy']/baseline_ref['back'])
            row['fea_050']=coarse_map[row['id']]
        (OUT/row['id']/'transfer_metrics.json').write_text(json.dumps(row,indent=2)+'\n')
    dest=ROOT/'experiments/chair/sofa_style_2026-09-28/single_view_image_qd_2026-10-03/transfer_3d.json'
    dest.write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:
        print(row['id'],'cell',row['realized_cell'],'repair',round(row['repair_fraction'],4),
              'gate',row['geometry_gate'],'iou',round(row['realized_iou_to_baseline'],3),
              'quality',row.get('worst_compliance_ratio_to_baseline'))


if __name__=='__main__':main()
