#!/usr/bin/env python3
"""Run sparse refinement from the validated chair backrest-BC dense cache."""
from __future__ import annotations

import argparse
import json
import subprocess

import trimesh

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

BASE=ROOT/'experiments/chair/sofa_style_2026-09-28/consistent_reference_2026-09-28'
STUDY=BASE/'backrest_load_ablation'
CASES=('all_support_vw0', 'all_support_v35')


def output_dir(case: str,bc_dilate_mm: float):
    tag='' if bc_dilate_mm==0 else f'_d{bc_dilate_mm:g}'
    return STUDY/case/f'sparse_followup{tag}'


def prepare(case: str,bc_dilate_mm: float) -> None:
    source=STUDY/case
    cache=source/'dense_cache.npz'
    audit=json.loads((source/'bc_audit.json').read_text())
    if not (audit['bc_geometry_pass'] and audit['shape_gate_pass']):
        raise ValueError(f'dense geometry failed its gate: {source}')
    if not cache.exists():
        raise FileNotFoundError(cache)
    out=output_dir(case,bc_dilate_mm)
    out.mkdir(exist_ok=True)
    # The high-resolution BC clamp reads STL, not the 64^3 voxel BC union.
    # Here this merged STL is used only geometrically: FEA weights remain zero.
    seat=trimesh.load(ROOT/'data_real/chair/load_remesh.stl',force='mesh')
    back=trimesh.load(STUDY/'backrest_load.stl',force='mesh')
    merged=trimesh.util.concatenate([seat,back])
    if not merged.is_watertight or len(merged.split())!=2:
        raise ValueError('merged geometric load interface is invalid')
    merged.export(out/'seat_and_back_bc_geometry.stl')
    cfg=json.loads((source/'config_dense.json').read_text())
    cfg['name']+= '_sparse_geometry_only'
    mesh=cfg['stages']['mesh']
    mesh.update(skip_sparse=False,load_dense_cache=str(cache),save_dense_cache=None,
                load_stl=str(out/'seat_and_back_bc_geometry.stl'),
                force_bc_dilate_mm=bc_dilate_mm,
                sp_guide_w_peak=10.0,sp_fea_w=0.0,fea_w=0.0)
    (out/'config_sparse.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (out/'setup.json').write_text(json.dumps({
        'source_dense_cache':str(cache), 'source_dense_audit':str(source/'bc_audit.json'),
        'geometric_bc_stl':str(out/'seat_and_back_bc_geometry.stl'),
        'bc_dilate_mm':bc_dilate_mm,
        'fea_enabled':False,'actual_backrest_force_applied':False,
        'reason':'sparse BC-solid clamp uses load_stl; use both geometric interfaces'
    },indent=2)+'\n')
    print(out/'config_sparse.json')


def run(case: str,gpu: int,bc_dilate_mm: float) -> None:
    out=output_dir(case,bc_dilate_mm)
    if not (out/'config_sparse.json').exists():
        prepare(case,bc_dilate_mm)
    target=BASE/'input_lr162'
    cmd=[str(PYTHON),str(GENERATOR),'--config',str(out/'config_sparse.json'),
         '--target-dir',str(target),'--out',str(out/'generation')]
    with (out/'run.log').open('w') as log:
        result=subprocess.run(cmd,cwd=ROOT,env=generation_env(gpu),
                              stdout=log,stderr=subprocess.STDOUT)
    (out/'run.json').write_text(json.dumps({'command':cmd,'gpu':gpu,
        'exit_code':result.returncode,'output':str(out/'generation/mesh.obj')},indent=2)+'\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    print(out/'generation/mesh.obj')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','run'))
    parser.add_argument('--case',choices=CASES,default='all_support_vw0')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--bc-dilate-mm',type=float,default=0.0)
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args.case,args.bc_dilate_mm)
    else:
        run(args.case,args.gpu,args.bc_dilate_mm)
