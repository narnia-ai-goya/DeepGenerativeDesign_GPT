#!/usr/bin/env python3
"""Same-dense-cache sparse chair ablation: thickness and image projection."""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR,PYTHON,generation_env
from validate_chair_bc_geometry import audit

SOURCE=ROOT/'experiments/chair/sofa_style_2026-09-28/backrest_bc_multi_examples_2026-09-29/diagonal_braced'
BC=ROOT/'experiments/chair/sofa_style_2026-09-28/consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
OUT=ROOT/'experiments/chair/sofa_style_2026-09-28/sparse_style_image_guidance_2026-09-29'
CASES={'thick_3':(3.,0.),'thick_6':(6.,0.),
       'thick_6_proj_0p5':(6.,.5),'thick_6_proj_2':(6.,2.)}


def run(name:str,gpu:int):
    thick,proj=CASES[name]
    folder=OUT/name
    folder.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((SOURCE/'sparse_pw2_d13/config_sparse.json').read_text())
    cfg['name']=f'chair_sparse_style_{name}'
    cfg['stages']['mesh'].update(sp_thick_w=thick,sp_image_proj_w=proj,
        fea_w=0.,sp_fea_w=0.)
    config=folder/'config.json'
    config.write_text(json.dumps(cfg,indent=2)+'\n')
    command=[str(PYTHON),str(GENERATOR),'--config',str(config),
        '--target-dir',str(SOURCE/'input_lr162'),'--out',str(folder/'generation')]
    with (folder/'run.log').open('w') as log:
        result=subprocess.run(command,cwd=ROOT,env=generation_env(gpu),
                              stdout=log,stderr=subprocess.STDOUT)
    check=audit(folder/'generation/mesh.obj',BC,SOURCE/'prototype.npz') if result.returncode==0 else None
    if check is not None:
        (folder/'bc_audit.json').write_text(json.dumps(check,indent=2)+'\n')
    row={'case':name,'thick_w':thick,'sp_image_proj_w':proj,'gpu':gpu,
         'exit_code':result.returncode,'command':command,
         'mesh':str(folder/'generation/mesh.obj'),
         'bc_pass':check['bc_geometry_pass'] if check else None,
         'shape_pass':check['shape_gate_pass'] if check else None}
    (folder/'run.json').write_text(json.dumps(row,indent=2)+'\n')
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('names',nargs='*')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=list(args.names) or list(CASES)
    unknown=set(jobs)-set(CASES)
    if unknown:
        parser.error(f'unknown cases: {sorted(unknown)}')
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        rows=[future.result() for future in
              [executor.submit(run,name,gpu) for gpu,name in enumerate(jobs)]]
    (OUT/'runs.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
