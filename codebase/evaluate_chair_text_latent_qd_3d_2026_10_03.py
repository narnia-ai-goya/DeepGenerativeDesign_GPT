#!/usr/bin/env python3
"""Matched 3D transfer, envelope/BC gate and common two-load FEM proxy."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from scipy.ndimage import label
from scipy.spatial.transform import Rotation
import trimesh

from make_chair_domain import ROOT
from chair_text_latent_qd_pilot_2026_10_03 import OUT,BASE
from build_chair_single_view_qd_archive_2026_10_03 import features,cell_for

sys.path.insert(0,str(ROOT/'codebase/code'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402

SPEC=BASE/'single_view_spec_2026-10-03'
SOLVER=ROOT/'codebase/code/fenics_fea_bracket.py'
FENICS='/home/goya/miniconda3/envs/fenics/bin/python'


def fea(row:dict,mode:str,nodes:np.ndarray,indices:np.ndarray)->dict:
    case=OUT/'mesh_cases'/row['id']
    occ=np.load(case/'realized_occupancy.npz')['occupied']
    folder=case/'fea';folder.mkdir(exist_ok=True)
    rho=np.where(occ[indices[:,0],indices[:,1],indices[:,2]],1.,.001)
    np.save(folder/'nodes.npy',nodes);np.save(folder/'rho.npy',rho)
    domain=SPEC/('fea_domain' if mode=='seat' else 'fea_domain_back')
    direction='-z' if mode=='seat' else 'y'
    force=800 if mode=='seat' else 200
    work=folder/mode;work.mkdir(exist_ok=True)
    command=[FENICS,str(SOLVER),'--domain-dir',str(domain),
             '--density',str(folder/'rho.npy'),'--nodes',str(folder/'nodes.npy'),
             '--output',str(work/'dc.npy'),'--mesh-cache',str(SPEC/'fea_domain/chair_035.msh'),
             '--mesh-size','.035','--penal','2','--E0','1.0','--load-magnitude',str(force)]
    env={**os.environ,'LOAD_MODE':direction,'BC_SURFACE_DIST':'1','BC_DIST':'0.025',
         'FEA_MAX_NODE_MAP_DISTANCE':'0.03','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
         'MKL_NUM_THREADS':'1'}
    with (work/'fea.log').open('w') as stream:
        process=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    log=(work/'fea.log').read_text(errors='replace')
    fix=re.search(r'Dirichlet: (\d+) nodes',log)
    load=re.search(r'Load: (\d+) nodes',log)
    info=work/'dc_info.json'
    result={'exit_code':process.returncode,'fixed_nodes':int(fix.group(1)) if fix else 0,
            'load_nodes':int(load.group(1)) if load else 0,
            'compliance_proxy':float(json.loads(info.read_text())['compliance'])
            if process.returncode==0 and info.exists() else None}
    result['valid']=bool(result['exit_code']==0 and result['fixed_nodes']>0 and
                         result['load_nodes']>0 and result['compliance_proxy'] is not None)
    return result


def main()->None:
    observations=json.loads((OUT/'observations.json').read_text())
    spec=np.load(SPEC/'voxel.npz');env=spec['bracket'].astype(bool);bc=spec['bc'].astype(bool)
    calibration=json.loads((SPEC/'specification.json').read_text())['native_frame_registration']
    rotation=Rotation.from_euler('x',calibration['rotation_x_degrees'],degrees=True).as_matrix()
    baseline=np.load(BASE/'single_view_qd_2026-10-03/seed_42/realized_occupancy.npz')['occupied']
    six=np.zeros((3,3,3),bool);six[1,1,1]=True
    for axis in range(3):
        for sign in (-1,1):
            offset=[1,1,1];offset[axis]+=sign;six[tuple(offset)]=True
    rows=[]
    for obs in observations:
        case=OUT/'mesh_cases'/obs['id']
        raw=trimesh.load(case/'generation/mesh.obj',force='mesh',process=False)
        raw.vertices=((raw.vertices-np.asarray(calibration['source_center_m']))@rotation.T*
                      calibration['uniform_scale']+np.asarray(calibration['physical_center_m']))
        mesh=max(raw.split(only_watertight=False),key=lambda part:abs(part.volume))
        mesh.export(case/'aligned_main.obj');mesh.export(case/'source_mesh.glb')
        occ=voxel_centers_inside(mesh,64,spec['origin'],spec['pitch_xyz']).astype(bool)
        realized=(occ&env)|bc
        added=int((realized&~occ).sum());removed=int((occ&~realized).sum())
        repair=(added+removed)/max(1,int(realized.sum()))
        _,components=label(realized,structure=six)
        pair=features(realized,spec)
        cell=cell_for(pair)
        overlap=int((realized&baseline).sum());union=int((realized|baseline).sum())
        row={**obs,'mesh':str(case/'aligned_main.obj'),'source_watertight':bool(mesh.is_watertight),
             'source_outside_envelope_fraction':float((occ&~env).sum()/max(1,int(occ.sum()))),
             'source_seat_bc_coverage':float((occ&spec['load']).sum()/max(1,int(spec['load'].sum()))),
             'source_back_bc_coverage':float((occ&spec['back_load']).sum()/max(1,int(spec['back_load'].sum()))),
             'repair_fraction':repair,'realized_components_6conn':int(components),
             'realized_mass_liters':float(realized.sum()*np.prod(spec['pitch_xyz'])*1000),
             'realized_3d_descriptor':{'side_open_fraction':pair[0],
                                       'backrest_taper_ratio':pair[1]},
             'realized_3d_cell':cell,'realized_iou_to_baseline':overlap/union,
             'physical_gate':bool(mesh.is_watertight and components==1 and repair<=.05)}
        row['archive_range_gate']=bool(row['physical_gate'] and cell is not None)
        np.savez_compressed(case/'realized_occupancy.npz',occupied=realized)
        rows.append(row)
    indices=np.argwhere(env);nodes=spec['origin']+(indices+.5)*spec['pitch_xyz']
    tasks=[(row,mode) for row in rows if row['physical_gate'] for mode in ('seat','back')]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(lambda item:(item[0]['id'],item[1],fea(item[0],item[1],nodes,indices)),tasks))
    fea_by={(i,mode):result for i,mode,result in results}
    old=json.loads((BASE/'single_view_qd_2026-10-03/summary.json').read_text())
    seat_ref=old['baseline_compliance_proxy']['seat'];back_ref=old['baseline_compliance_proxy']['back']
    for row in rows:
        if row['physical_gate']:
            seat=fea_by[row['id'],'seat'];back=fea_by[row['id'],'back']
            row['fea_035']={'seat':seat,'back':back}
            row['fea_valid']=seat['valid'] and back['valid']
            if row['fea_valid']:
                row['worst_compliance_ratio_to_baseline']=max(
                    seat['compliance_proxy']/seat_ref,back['compliance_proxy']/back_ref)
        else:row['fea_valid']=False
        (OUT/'mesh_cases'/row['id']/'evaluation.json').write_text(json.dumps(row,indent=2)+'\n')
        print(row['method'],row['id'],'image',row['image_cell'],
              '3D',row['realized_3d_cell'],'physical',row['physical_gate'],
              'archive-range',row['archive_range_gate'],
              'repair',round(row['repair_fraction'],3),
              'quality',row.get('worst_compliance_ratio_to_baseline'),flush=True)
    (OUT/'mesh_evaluations.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
