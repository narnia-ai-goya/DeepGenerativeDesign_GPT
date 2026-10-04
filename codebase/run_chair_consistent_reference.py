#!/usr/bin/env python3
"""Render one 3D chair into aligned views, then test Direct3D-S2 dense recovery."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, gaussian_filter

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env
from report_chair_prefea_stages import silhouette_scores
from prepare_chair_camera_projection_targets import camera_grid
from validate_chair_bc_geometry import audit

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
PROXY_CASE = BASE / 'coherent_proxy_dense'
OUT = BASE / 'consistent_reference_2026-09-28'
VIEWS = (('front','v00_front_lo',15,0),
         ('right','v02_right_lo',15,90),
         ('rear','v04_back_lo',15,180),
         ('top','v_top',85,0))


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'input').mkdir(exist_ok=True)
    (OUT/'input_lr162').mkdir(exist_ok=True)
    proxy = trimesh.load(PROXY_CASE/'prototype.obj',force='mesh')
    env = trimesh.load(ROOT/'data_real/chair/original_DesignSpace.stl',force='mesh')
    voxel = np.load(PROXY_CASE/'voxel.npz')
    occ = np.load(PROXY_CASE/'prototype.npz')['prototypes'][0].astype(np.float32)
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents))*1.5
    half_extent = float(env.extents.max())/2*1.15
    depth_half = float(np.linalg.norm(env.extents))
    arrays = {'active_threshold':np.float32(.1)}
    checks = {}
    sheet = Image.new('RGB',(512*4,548),'white')
    draw = ImageDraw.Draw(sheet)
    field = torch.from_numpy(occ)[None,None]
    for idx,(name,stem,elev,azim) in enumerate(VIEWS):
        eye,up = camera_from_elev_azim(center,radius,elev,azim)
        rgb = np.asarray(render_lit(proxy,eye,center,up,size=512,
            fit_extent=float(env.extents.max())/2,margin=1.15,
            color=(.38,.42,.46)),dtype=np.uint8)
        grid = camera_grid(center,eye,up,half_extent,voxel['origin'],
                           voxel['pitch_xyz'],depth_half)
        with torch.no_grad():
            sampled=F.grid_sample(field,torch.from_numpy(grid.astype(np.float32))[None],
                                  mode='bilinear',padding_mode='zeros',align_corners=True)[0,0]
            projected=sampled.amax(dim=0).numpy()>.5
        # Use the *same 3D occupancy projection* for input alpha and the
        # guidance target. This removes rasterizer/voxel boundary disagreement.
        alpha_hi=np.asarray(Image.fromarray((projected.astype(np.uint8)*255))
            .resize((512,512),Image.Resampling.NEAREST),dtype=np.uint8)
        styled=rgb.copy()
        styled[(alpha_hi>0)&(rgb.min(axis=2)>245)]=[94,104,114]
        rgba=Image.fromarray(np.dstack([styled,alpha_hi]),'RGBA')
        rgba.save(OUT/'input'/f'{stem}.png')
        rgba.resize((162,162),Image.Resampling.LANCZOS).save(
            OUT/'input_lr162'/f'{stem}.png')
        preview=Image.new('RGB',(512,512),'white')
        preview.paste(rgba,mask=rgba.getchannel('A'))
        sheet.paste(preview,(idx*512,36))
        draw.text((idx*512+12,10),f'{name}: same 3D mesh',fill='#20252b')
        target=np.clip(gaussian_filter(projected.astype(np.float32),sigma=.35),0,1)
        near=binary_dilation(target>.05,iterations=8)
        weight=np.where(near,1.0,.25).astype(np.float32)
        rendered_mask=(np.asarray(preview).min(axis=2)<210)
        rendered_mask=np.asarray(Image.fromarray(rendered_mask.astype(np.uint8)*255)
            .resize((64,64),Image.Resampling.BOX))>127
        alignment=float((projected&rendered_mask).sum()/max((projected|rendered_mask).sum(),1))
        arrays[f'target_{name}']=target.astype(np.float32)
        arrays[f'weight_{name}']=weight
        arrays[f'camera_grid_{name}']=grid
        checks[name]={'projection_iou_from_same_3d':round(alignment,4),
                      'target_pixels_64':int(rendered_mask.sum()),
                      'camera_elev_deg':elev,'camera_azim_deg':azim}
    sheet.save(OUT/'input_contact.png')
    np.savez_compressed(OUT/'camera_projection_targets.npz',**arrays)
    source=json.loads((PROXY_CASE/'config_dense.json').read_text())
    for case in ('image_only','shape_guided','shape_strong','shape_no_volume'):
        case_dir=OUT/case
        case_dir.mkdir(exist_ok=True)
        cfg=json.loads(json.dumps(source))
        cfg['name']=f'chair_consistent_reference_{case}'
        cfg['views']='v00_front_lo,v02_right_lo,v04_back_lo,v_top'
        mesh=cfg['stages']['mesh']
        mesh.update(views=cfg['views'],n_views=4,
                    image_proj_target=str(OUT/'camera_projection_targets.npz'),
                    vol_target=.25,save_dense_cache=str(case_dir/'dense_cache.npz'),
                    load_dense_cache=None,skip_sparse=True,fea_w=0.0,
                    shape_anchor_w={'image_only':0.0,'shape_guided':300.0,
                                    'shape_strong':1200.0,'shape_no_volume':300.0}[case],
                    shape_scaffold_w={'image_only':0.0,'shape_guided':50.0,
                                      'shape_strong':200.0,'shape_no_volume':50.0}[case],
                    shape_qd_warmup=0.0)
        if case=='shape_no_volume':
            mesh['vw']=0.0
        (case_dir/'config_dense.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (OUT/'registration.json').write_text(json.dumps({'reference_mesh':str(PROXY_CASE/'prototype.obj'),
        'bc':str(PROXY_CASE/'voxel.npz'),'views':checks},indent=2)+'\n')
    print(json.dumps(checks,indent=2))


def run(case,gpu,allow_invalid=False):
    case_dir=OUT/case
    command=[str(PYTHON),str(GENERATOR),'--config',str(case_dir/'config_dense.json'),
             '--target-dir',str(OUT/'input_lr162'),'--out',str(case_dir/'dense')]
    with (case_dir/'dense.log').open('w') as log:
        result=subprocess.run(command,cwd=ROOT,env=generation_env(gpu),
                              stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(result.returncode)
    check=audit(case_dir/'dense/mesh_dense.obj',PROXY_CASE/'voxel.npz',
                PROXY_CASE/'prototype.npz')
    (case_dir/'bc_audit.json').write_text(json.dumps(check,indent=2)+'\n')
    (case_dir/'run.json').write_text(json.dumps({'command':command,'gpu':gpu,
        'exit_code':result.returncode,'mesh':str(case_dir/'dense/mesh_dense.obj'),
        'bc_geometry_pass':check['bc_geometry_pass'],
        'shape_gate_pass':check['shape_gate_pass']},indent=2)+'\n')
    print(case_dir/'dense/mesh_dense.obj')
    if not check['bc_geometry_pass'] and not allow_invalid:
        raise SystemExit('BC geometry gate failed; dense result cannot advance to sparse/FEA')


def report():
    env=trimesh.load(ROOT/'data_real/chair/original_DesignSpace.stl',force='mesh')
    center=env.bounds.mean(axis=0)
    radius=float(np.linalg.norm(env.extents))*1.5
    extent=float(env.extents.max())/2
    cases=[('reference 3D',PROXY_CASE/'prototype.obj'),
           ('consistent images only',OUT/'image_only/dense/mesh_dense.obj'),
           ('consistent images + 3D',OUT/'shape_guided/dense/mesh_dense.obj'),
           ('consistent images + strong 3D',OUT/'shape_strong/dense/mesh_dense.obj'),
           ('consistent images + 3D, no volume',OUT/'shape_no_volume/dense/mesh_dense.obj')]
    report_views=(('oblique',25,35),('front',15,0),('right',15,90),
                  ('rear',15,180),('top',85,0))
    size,header=360,34
    sheet=Image.new('RGB',(size*len(report_views),(size+header)*len(cases)),'white')
    draw=ImageDraw.Draw(sheet)
    metrics={}
    for i,(case,path) in enumerate(cases):
        mesh=trimesh.load(path,force='mesh')
        parts=mesh.split(only_watertight=False)
        row={'mesh':str(path),'components':len(parts),
             'component_faces':sorted([len(p.faces) for p in parts],reverse=True),
             'volume_litres':round(abs(float(mesh.volume))*1000,3),'silhouettes':{}}
        for j,(view,elev,azim) in enumerate(report_views):
            eye,up=camera_from_elev_azim(center,radius,elev,azim)
            rendered=Image.fromarray(render_lit(mesh,eye,center,up,size=size,
                fit_extent=extent,margin=1.15,color=(.48,.52,.56))).convert('RGB')
            sheet.paste(rendered,(j*size,i*(size+header)+header))
            draw.text((j*size+9,i*(size+header)+9),f'{case}: {view}',fill='#20252b')
            stem=next((v[1] for v in VIEWS if v[0]==view),None)
            if stem:
                target=Image.open(OUT/'input'/f'{stem}.png').convert('RGBA')
                target_rgb=Image.new('RGB',target.size,'white')
                target_rgb.paste(target,mask=target.getchannel('A'))
                row['silhouettes'][view]=silhouette_scores(rendered,
                    target_rgb.resize((size,size)))
        metrics[case]=row
    sheet.save(OUT/'comparison.png')
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    rows=''.join('<tr><td><a href="/'+str(path.relative_to(ROOT))+'">'+case+'</a></td>'
      + ''.join(f'<td>{metrics[case]["silhouettes"][v]["iou"]:.3f}</td>'
                for v in ('front','right','rear','top'))
      + f'<td>{metrics[case]["components"]}</td></tr>' for case,path in cases)
    (OUT/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
      '<title>Consistent chair reference test</title><style>body{font:16px system-ui;'
      'max-width:1850px;margin:2rem auto;background:#f4f6f8;color:#1d2730}'
      'section{background:#fff;padding:1rem;margin:1rem 0;border-radius:12px}'
      'img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;'
      'padding:.45rem}</style><section><h1>같은 3D 원형에서 만든 4뷰 입력</h1>'
      '<p>FEA·sparse 없이 dense에서 동일한 3D 의자의 영상과 형상 기준을 비교합니다. '
      '입력 마스크와 64³ 투영 목표는 네 방향 모두 IoU 1.0입니다.</p>'
      '<table><tr><th>OBJ</th><th>앞 IoU</th><th>오른쪽 IoU</th><th>뒤 IoU</th>'
      '<th>위 IoU</th><th>성분</th></tr>'+rows+'</table>'
      '<p><strong>판정:</strong> 영상만으로는 7개 조각이며, 3D 기준을 추가하면 1개로 '
      '연결됩니다. 그러나 등받이 윗가로대는 여전히 끊기고 강한 가중치도 이를 '
      '복구하지 못합니다. 재료량 손실을 끄면 4개 조각으로 분리됩니다. '
      'sparse/FEA는 진행하지 않았습니다.</p>'
      '<p><a href="registration.json">투영 정합 확인</a> · '
      '<a href="metrics.json">결과 수치</a> · <a href="STUDY.md">상세 분석</a></p>'
      '<img src="input_contact.png"></section>'
      '<section><h2>3D 렌더 비교</h2><img src="comparison.png"></section></html>')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','run','report'])
    parser.add_argument('--case',choices=['image_only','shape_guided','shape_strong','shape_no_volume'],default='image_only')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--allow-invalid',action='store_true',
                        help='record an invalid ablation without failing the command')
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='run': run(args.case,args.gpu,args.allow_invalid)
    else: report()
