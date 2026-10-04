#!/usr/bin/env python3
"""Run the proven chair multiview+BC recipe on the complex image-first concept."""
from __future__ import annotations

import argparse
import json
import subprocess

import run_chair_backrest_multi_examples as recipe
from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR,PYTHON,generation_env
from validate_chair_bc_geometry import audit

IMAGE=ROOT/'experiments/chair/image_concepts_2026-09-27/complex_truss_armchair/multiview_registered'
OUT=ROOT/'experiments/chair/complex_truss_image_to_3d_2026-09-29'
CASE='main'

# Bind the existing recipe to this image set. Its BC STL, support corridors,
# geometry gate, generator and FEA-off settings remain unchanged.
recipe.OUT=OUT
recipe.image_source=lambda _case: IMAGE
recipe.projection_target=lambda _case: IMAGE/'camera_projection_targets.npz'
recipe.input_dir=lambda _case: IMAGE/'input_lr162'


def run_strong_image(gpu:int):
    case=OUT/CASE
    cfg=json.loads((case/'config_dense_pw2.json').read_text())
    cfg['name']='chair_complex_truss_strong_image_dense'
    cfg['stages']['mesh'].update(pw=2.0,image_proj_w=100.0,
        shape_anchor_w=1500.0,shape_scaffold_w=160.0,
        save_dense_cache=str(case/'dense_strong_image_cache.npz'),
        load_dense_cache=None,skip_sparse=True,fea_w=0.,sp_fea_w=0.)
    config=case/'config_dense_strong_image.json'
    config.write_text(json.dumps(cfg,indent=2)+'\n')
    output=case/'dense_strong_image'
    command=[str(PYTHON),str(GENERATOR),'--config',str(config),
        '--target-dir',str(IMAGE/'input_lr162'),'--out',str(output)]
    with (case/'dense_strong_image.log').open('w') as log:
        result=subprocess.run(command,cwd=ROOT,env=generation_env(gpu),
                              stdout=log,stderr=subprocess.STDOUT)
    check=audit(output/'mesh_dense.obj',recipe.BC_STUDY/'voxel.npz',case/'prototype.npz') if result.returncode==0 else None
    if check is not None:
        (case/'dense_strong_image_bc_audit.json').write_text(json.dumps(check,indent=2)+'\n')
    row={'command':command,'exit_code':result.returncode,'audit':check}
    (case/'dense_strong_image_run.json').write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps({'exit_code':result.returncode,'bc_pass':check['bc_geometry_pass'] if check else None,
        'shape_pass':check['shape_gate_pass'] if check else None,
        'mesh':str(output/'mesh_dense.obj')},indent=2))
    return result.returncode


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','dense','dense_strong_image','sparse'))
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--variant',choices=('base','pw2'),default='base')
    args=parser.parse_args()
    if args.stage=='prepare':recipe.prepare(CASE)
    elif args.stage=='dense':recipe.run_dense(CASE,args.gpu,args.variant)
    elif args.stage=='dense_strong_image':raise SystemExit(run_strong_image(args.gpu))
    else:recipe.run_sparse(CASE,args.gpu,13.,args.variant)


if __name__=='__main__':main()
