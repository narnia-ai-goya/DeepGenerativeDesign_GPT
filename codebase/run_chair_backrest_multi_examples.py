#!/usr/bin/env python3
"""Test the backrest-BC chair recipe on distinct existing image concepts."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_closing,label
from skimage.measure import marching_cubes

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR,PYTHON,generation_env
from validate_chair_bc_geometry import audit

BASE=ROOT/'experiments/chair/sofa_style_2026-09-28'
BC_STUDY=BASE/'consistent_reference_2026-09-28/backrest_load_ablation'
OUT=BASE/'backrest_bc_multi_examples_2026-09-29'
CASES=('open_arm','solid_side','diagonal_braced')


def image_source(case):
    if case=='diagonal_braced':
        return ROOT/'experiments/chair/image_concepts_2026-09-27/diagonal_braced/multiview_registered'
    return BASE/case


def projection_target(case):
    if case=='diagonal_braced':
        return image_source(case).parent/'camera_projection_targets.npz'
    return image_source(case)/'camera_projection_targets.npz'


def input_dir(case):
    if case=='diagonal_braced':return OUT/case/'input_lr162'
    return image_source(case)/'input_lr162'


def visual_hull(case,bc):
    sys.path.insert(0,str(ROOT/'codebase/code/conditioning'))
    from cond_render_pv import camera_from_elev_azim
    targets=np.load(projection_target(case))
    env_mesh=trimesh.load(ROOT/'data_real/chair/original_DesignSpace.stl',force='mesh')
    center=env_mesh.bounds.mean(axis=0)
    radius=float(np.linalg.norm(env_mesh.extents))*1.5
    half_extent=float(env_mesh.extents.max())/2*1.15
    ijk=np.indices((64,64,64)).reshape(3,-1).T
    world=bc['origin']+(ijk+.5)*bc['pitch_xyz']
    hull=np.ones(len(world),dtype=bool)
    matched={}
    for name,elev,azim in (('front',15,0),('right',15,90),('top',85,0)):
        eye,up=camera_from_elev_azim(center,radius,elev,azim)
        forward=(center-eye)/np.linalg.norm(center-eye)
        right=np.cross(forward,up);right/=np.linalg.norm(right)
        true_up=np.cross(right,forward);true_up/=np.linalg.norm(true_up)
        relative=world-center
        xx=np.floor((relative@right/(2*half_extent)+.5)*64).astype(int)
        yy=np.floor((.5-relative@true_up/(2*half_extent))*64).astype(int)
        valid=(xx>=0)&(xx<64)&(yy>=0)&(yy<64)
        selected=np.zeros(len(world),dtype=bool)
        mask=targets[f'target_{name}']>.35
        selected[valid]=mask[yy[valid],xx[valid]]
        hull&=selected
        matched[name]=int(selected.sum())
    shape=hull.reshape((64,64,64))&bc['bracket'].astype(bool)
    shape=binary_closing(shape,iterations=1)&bc['bracket'].astype(bool)
    return shape,matched


def prepare(case):
    sub=OUT/case;sub.mkdir(parents=True,exist_ok=True)
    if case=='diagonal_braced':
        small=input_dir(case);small.mkdir(exist_ok=True)
        for stem in ('v00_front_lo','v02_right_lo','v_top'):
            Image.open(image_source(case)/'input'/f'{stem}.png').convert('RGB').resize(
                (162,162),Image.Resampling.LANCZOS).save(small/f'{stem}.png')
    bc=np.load(BC_STUDY/'voxel.npz')
    mask,per_view=visual_hull(case,bc)
    paths=np.load(BC_STUDY/'all_support_corridors.npz')['mask'].astype(bool)
    mask|=paths|bc['bc'].astype(bool)
    labels,n=label(mask)
    sizes=sorted(np.bincount(labels[labels>0].ravel()).tolist(),reverse=True)
    # Do not silently discard components: they diagnose ambiguous image regions.
    np.savez_compressed(sub/'prototype.npz',prototypes=mask[None].astype(np.float32))
    verts,faces,_,_=marching_cubes(np.pad(mask,1),.5)
    verts=bc['origin']+(verts-.5)*bc['pitch_xyz']
    proxy=trimesh.Trimesh(vertices=verts,faces=faces,process=False)
    proxy.export(sub/'prototype.obj')
    source=json.loads((BC_STUDY/'all_support_vw0/config_dense.json').read_text())
    source['name']='chair_backrest_bc_multi_'+case
    source['views']='v00_front_lo,v02_right_lo,v_top'
    m=source['stages']['mesh']
    m.update(views=source['views'],n_views=3,
             image_proj_target=str(projection_target(case)),
             shape_anchor_bank=str(sub/'prototype.npz'),
             save_dense_cache=str(sub/'dense_cache.npz'),load_dense_cache=None,
             skip_sparse=True,fea_w=0.0,sp_fea_w=0.0)
    (sub/'config_dense.json').write_text(json.dumps(source,indent=2)+'\n')
    (sub/'prototype_metrics.json').write_text(json.dumps({
        'case':case,'prototype_voxels':int(mask.sum()),'components':n,
        'component_sizes':sizes,'per_view_projected_voxels':per_view,
        'bc_voxels':int(bc['bc'].sum()),'bc_all_inside':bool(np.all(mask[bc['bc'].astype(bool)])),
        'paths_all_inside':bool(np.all(mask[paths]))},indent=2)+'\n')
    print(sub/'config_dense.json')


def run_dense(case,gpu,variant='base'):
    sub=OUT/case
    if not (sub/'config_dense.json').exists():prepare(case)
    config_path=sub/'config_dense.json'
    dense_dir=sub/'dense'
    log_path=sub/'dense.log'
    cache=sub/'dense_cache.npz'
    audit_path=sub/'dense_bc_audit.json'
    run_path=sub/'dense_run.json'
    if variant=='pw2':
        cfg=json.loads(config_path.read_text())
        cfg['name']+='_pw2'
        cfg['stages']['mesh'].update(pw=2.0,shape_anchor_w=500.0,
                                     shape_scaffold_w=80.0,
                                     save_dense_cache=str(sub/'dense_pw2_cache.npz'))
        config_path=sub/'config_dense_pw2.json'
        config_path.write_text(json.dumps(cfg,indent=2)+'\n')
        dense_dir=sub/'dense_pw2';log_path=sub/'dense_pw2.log'
        cache=sub/'dense_pw2_cache.npz';audit_path=sub/'dense_pw2_bc_audit.json'
        run_path=sub/'dense_pw2_run.json'
    cmd=[str(PYTHON),str(GENERATOR),'--config',str(config_path),
         '--target-dir',str(input_dir(case)),'--out',str(dense_dir)]
    with log_path.open('w') as log:
        result=subprocess.run(cmd,cwd=ROOT,env=generation_env(gpu),
                              stdout=log,stderr=subprocess.STDOUT)
    run_path.write_text(json.dumps({'command':cmd,'exit_code':result.returncode,
        'mesh':str(dense_dir/'mesh_dense.obj'),'cache':str(cache)},indent=2)+'\n')
    if result.returncode:raise SystemExit(result.returncode)
    result=audit(dense_dir/'mesh_dense.obj',BC_STUDY/'voxel.npz',sub/'prototype.npz')
    audit_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'mesh':str(dense_dir/'mesh_dense.obj'),
        'bc_pass':result['bc_geometry_pass'],'shape_pass':result['shape_gate_pass'],
        'components':result['mesh_components'],'bar':result['top_bar']['coverage']},indent=2))


def run_sparse(case,gpu,bc_dilate_mm,dense_variant='base'):
    sub=OUT/case
    suffix='' if dense_variant=='base' else '_'+dense_variant
    dense=json.loads((sub/f'dense{suffix}_bc_audit.json').read_text())
    if not dense['shape_gate_pass']:
        raise SystemExit('dense BC/shape gate failed; sparse skipped')
    out=sub/f'sparse{suffix}_d{bc_dilate_mm:g}';out.mkdir(exist_ok=True)
    cfg=json.loads((sub/f'config_dense{suffix}.json').read_text())
    m=cfg['stages']['mesh']
    m.update(skip_sparse=False,load_dense_cache=str(sub/f'dense{suffix}_cache.npz'),
             save_dense_cache=None,sp_guide_w_peak=10.0,
             load_stl=str(BC_STUDY/'all_support_vw0/sparse_followup_d13/seat_and_back_bc_geometry.stl'),
             force_bc_dilate_mm=bc_dilate_mm,fea_w=0.0,sp_fea_w=0.0)
    (out/'config_sparse.json').write_text(json.dumps(cfg,indent=2)+'\n')
    cmd=[str(PYTHON),str(GENERATOR),'--config',str(out/'config_sparse.json'),
         '--target-dir',str(input_dir(case)),'--out',str(out/'generation')]
    with (out/'run.log').open('w') as log:
        result=subprocess.run(cmd,cwd=ROOT,env=generation_env(gpu),
                              stdout=log,stderr=subprocess.STDOUT)
    (out/'run.json').write_text(json.dumps({'command':cmd,'exit_code':result.returncode,
        'mesh':str(out/'generation/mesh.obj')},indent=2)+'\n')
    if result.returncode:raise SystemExit(result.returncode)
    check=audit(out/'generation/mesh.obj',BC_STUDY/'voxel.npz',sub/'prototype.npz')
    (out/'bc_audit.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps({'mesh':str(out/'generation/mesh.obj'),
        'bc_pass':check['bc_geometry_pass'],'shape_pass':check['shape_gate_pass'],
        'components':check['mesh_components'],'bar':check['top_bar']['coverage']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','dense','sparse'))
    parser.add_argument('--case',choices=CASES,required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--bc-dilate-mm',type=float,default=13.0)
    parser.add_argument('--dense-variant',choices=('base','pw2'),default='base')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.case)
    elif args.command=='dense':run_dense(args.case,args.gpu,args.dense_variant)
    else:run_sparse(args.case,args.gpu,args.bc_dilate_mm,args.dense_variant)
