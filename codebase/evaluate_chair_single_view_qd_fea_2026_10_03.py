#!/usr/bin/env python3
"""Matched, post-generation two-load FEM proxy for voxel-realized QD chairs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from make_chair_domain import ROOT

BASE=ROOT/'experiments/chair/sofa_style_2026-09-28'
OUT=BASE/'single_view_qd_2026-10-03'
SPEC=BASE/'single_view_spec_2026-10-03'
SOLVER=ROOT/'codebase/code/fenics_fea_bracket.py'
FENICS='/home/goya/miniconda3/envs/fenics/bin/python'
MESH=SPEC/'fea_domain/chair_050.msh'


def evaluate_case(row: dict, nodes: np.ndarray, indices: np.ndarray) -> dict:
    case=OUT/row['id'];folder=case/'fea';folder.mkdir(exist_ok=True)
    occ=np.load(case/'realized_occupancy.npz')['occupied']
    rho=np.where(occ[indices[:,0],indices[:,1],indices[:,2]],1.,.001)
    np.save(folder/'nodes.npy',nodes);np.save(folder/'rho.npy',rho)
    result={'id':row['id'],'mass_liters':row['realized_mass_liters'],
            'cell':row['realized_cell'],'rho_solid_nodes':int((rho==1).sum())}
    for name,domain,direction,force in [('seat',SPEC/'fea_domain','-z',800),
                                         ('back',SPEC/'fea_domain_back','y',200)]:
        work=folder/name;work.mkdir(exist_ok=True)
        command=[FENICS,str(SOLVER),'--domain-dir',str(domain),
                 '--density',str(folder/'rho.npy'),'--nodes',str(folder/'nodes.npy'),
                 '--output',str(work/'dc.npy'),'--mesh-cache',str(MESH),
                 '--mesh-size','.05','--penal','2','--E0','1.0',
                 '--load-magnitude',str(force)]
        env={**os.environ,'LOAD_MODE':direction,'BC_SURFACE_DIST':'1',
             'BC_DIST':'0.025','FEA_MAX_NODE_MAP_DISTANCE':'0.03',
             'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        with (work/'fea.log').open('w') as stream:
            process=subprocess.run(command,cwd=ROOT,env=env,
                                   stdout=stream,stderr=subprocess.STDOUT)
        log=(work/'fea.log').read_text(errors='replace')
        fixed=re.search(r'Dirichlet: (\d+) nodes',log)
        loaded=re.search(r'Load: (\d+) nodes',log)
        fixed_n=int(fixed.group(1)) if fixed else 0
        load_n=int(loaded.group(1)) if loaded else 0
        info_path=work/'dc_info.json'
        compliance=(float(json.loads(info_path.read_text())['compliance'])
                    if process.returncode==0 and info_path.exists() else None)
        result[name]={'exit_code':process.returncode,'fixed_nodes':fixed_n,
                      'load_nodes':load_n,'compliance_proxy':compliance,
                      'valid':bool(process.returncode==0 and fixed_n>0 and
                                   load_n>0 and compliance is not None and compliance>0),
                      'log':str(work/'fea.log')}
    (case/'fea_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main() -> None:
    archive=json.loads((OUT/'geometry_archive.json').read_text())
    selected=[r for r in archive['rows'] if r['geometry_gate']]
    spec=np.load(SPEC/'voxel.npz');indices=np.argwhere(spec['bracket'])
    nodes=spec['origin']+(indices+.5)*spec['pitch_xyz']
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(lambda row:evaluate_case(row,nodes,indices),selected))
    (OUT/'fea_results.json').write_text(json.dumps(results,indent=2)+'\n')
    for item in results:
        print(item['id'],'seat',item['seat']['compliance_proxy'],
              'back',item['back']['compliance_proxy'],
              'valid',item['seat']['valid'] and item['back']['valid'],flush=True)


if __name__=='__main__':main()
