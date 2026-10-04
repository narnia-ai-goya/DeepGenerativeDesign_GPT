#!/usr/bin/env python3
"""Test explicit chair topology at dense stage against the prior image-only run."""
from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.ndimage import label
from skimage.measure import marching_cubes

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env
from report_chair_prefea_stages import silhouette_scores
from cond_render_pv import camera_from_elev_azim, render_lit

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'coherent_proxy_dense'


def tube(x, y, z, p, q, radius):
    # Voxelized round structural member in world coordinates (metres).
    v = np.stack([x-p[0], y-p[1], z-p[2]], axis=-1)
    d = np.asarray(q, float) - p
    t = np.clip(np.sum(v*d, axis=-1) / np.dot(d, d), 0, 1)
    return np.sum((v-t[..., None]*d)**2, axis=-1) <= radius**2


def export_vox(mask, origin, pitch, destination):
    verts, faces, _, _ = marching_cubes(np.pad(mask, 1), .5)
    verts = origin + (verts-.5)*pitch
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.invert()
    mesh.export(destination)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    source = np.load(ROOT / 'data_real/chair/voxel.npz')
    arrays = {k: np.array(source[k]) for k in source.files}
    origin, pitch = arrays['origin'], arrays['pitch_xyz']
    axes = [origin[i] + (np.arange(64)+.5)*pitch[i] for i in range(3)]
    x, y, z = np.meshgrid(*axes, indexing='ij')

    # Extend only the narrow side wings forward to permit the illustrated arms.
    old_envelope = arrays['bracket'].astype(bool)
    arm_permission = ((np.abs(x) >= .175) & (np.abs(x) <= .285)
                      & (y >= -.255) & (y <= .23) & (z >= .47) & (z <= .735))
    envelope = old_envelope | arm_permission
    bc = arrays['bc'].astype(bool)
    if OUT.name.endswith('_v4'):
        # Functional rear clearance is a true non-design void in this ablation.
        hard_void = ((np.abs(x) < .135) & (y > .105) & (y < .245)
                     & (z > .105) & (z < .445))
        if np.any(hard_void & bc):
            raise ValueError('rear clearance intersects BC')
        envelope &= ~hard_void
    arrays['bracket'] = envelope
    arrays['design'] = envelope & ~bc

    # Variant 2 starts from the image visual hull to preserve the broad upper
    # back silhouette; explicit paths/voids resolve only its 3D ambiguities.
    seat = ((np.abs(x) <= .245) & (np.abs(y) <= .235)
            & (z >= .425) & (z <= .485))
    upper_back = ((np.abs(x) <= .23) & (y >= .17) & (y <= .255)
                  & (z >= .765) & (z <= .855))
    if OUT.name.endswith(('_v2', '_v3', '_v4')):
        hull = np.load(BASE / 'visual_hull_dense/visual_hull_prototype.npz')['prototypes'][0] > .5
        proxy = hull | bc
        underseat_void = ((np.abs(x) < .135) & (np.abs(y) < .145)
                          & (z > .095) & (z < .415))
        proxy &= ~underseat_void
    else:
        proxy = seat | upper_back | bc
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = sx*.195, sy*.18
            proxy |= tube(x, y, z, (cx,cy,.045), (cx,cy,.465), .048)
        # Back uprights, curved rising arms, and front arm-seat posts.
        proxy |= tube(x, y, z, (sx*.215,.20,.45), (sx*.215,.20,.83),
                      .072 if OUT.name.endswith(('_v3', '_v4')) else .052)
        proxy |= tube(x, y, z, (sx*.218,-.225,.625), (sx*.218,.195,.715), .045)
        proxy |= tube(x, y, z, (sx*.218,-.205,.465), (sx*.218,-.205,.635), .043)
        proxy |= tube(x, y, z, (sx*.215,.18,.66), (sx*.215,.20,.83), .045)
    # Explicit open rear void. No midspan vertical sheet can satisfy this prototype.
    rear_void = ((np.abs(x) < .13) & (y > .105) & (y < .245)
                 & (z > .105) & (z < .415))
    proxy &= ~rear_void
    proxy = (proxy & envelope) | bc
    labels, count = label(proxy)
    component_sizes = sorted(np.bincount(labels[labels > 0].ravel()).tolist(), reverse=True)
    if count != 1:
        raise ValueError(f'prototype not connected: {component_sizes}')
    if np.any(bc & ~proxy):
        raise ValueError('BC is not fully represented')
    np.savez_compressed(OUT / 'voxel.npz', **arrays)
    np.savez_compressed(OUT / 'prototype.npz', prototypes=proxy[None].astype(np.float32))
    export_vox(envelope, origin, pitch, OUT / 'envelope.stl')
    export_vox(proxy, origin, pitch, OUT / 'prototype.obj')
    baseline = json.loads((BASE / 'visual_hull_dense/anchor100_sc20/config_dense.json').read_text())
    baseline['name'] = OUT.name
    mesh = baseline['stages']['mesh']
    anchor_w = {'coherent_proxy_dense': 300.0, 'coherent_proxy_dense_v2': 200.0,
                'coherent_proxy_dense_v3': 500.0, 'coherent_proxy_dense_v4': 300.0}[OUT.name]
    scaffold_w = {'coherent_proxy_dense': 50.0, 'coherent_proxy_dense_v2': 35.0,
                  'coherent_proxy_dense_v3': 80.0, 'coherent_proxy_dense_v4': 50.0}[OUT.name]
    mesh.update(bc_proper=str(OUT / 'voxel.npz'), bracket_occ=str(OUT / 'voxel.npz'),
                fea_bracket_stl=str(OUT / 'envelope.stl'),
                shape_anchor_bank=str(OUT / 'prototype.npz'),
                shape_anchor_w=anchor_w, shape_scaffold_w=scaffold_w,
                shape_qd_warmup=0.0 if OUT.name.endswith(('_v3','_v4')) else .25,
                save_dense_cache=str(OUT / 'dense_cache.npz'), load_dense_cache=None,
                skip_sparse=True, fea_w=0.0)
    (OUT / 'config_dense.json').write_text(json.dumps(baseline, indent=2) + '\n')
    metrics = dict(proxy_voxels=int(proxy.sum()), envelope_voxels=int(envelope.sum()),
                   added_envelope_voxels=int((envelope & ~old_envelope).sum()),
                   bc_voxels=int(bc.sum()), components=count,
                   component_sizes=component_sizes, rear_void_voxels=int(rear_void.sum()),
                   prototype=str(OUT / 'prototype.obj'), config=str(OUT / 'config_dense.json'))
    (OUT / 'prototype_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


def run(gpu):
    command = [str(PYTHON), str(GENERATOR), '--config', str(OUT / 'config_dense.json'),
               '--target-dir', str(BASE / 'open_arm/input_lr162'), '--out', str(OUT / 'dense')]
    with (OUT / 'dense.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=generation_env(gpu),
                                stdout=log, stderr=subprocess.STDOUT)
    (OUT / 'run.json').write_text(json.dumps(dict(command=command, gpu=gpu,
        exit_code=result.returncode, mesh=str(OUT / 'dense/mesh_dense.obj')),
        indent=2) + '\n')
    if result.returncode:
        raise SystemExit(result.returncode)
    print(OUT / 'dense/mesh_dense.obj')


def report():
    reference = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = reference.bounds.mean(axis=0)
    radius = float(np.linalg.norm(reference.extents))*1.5
    scale = float(reference.extents.max())/2
    cases = [
        ('image-only dense', BASE / 'tuning_open_arm/p50_v35_pw00/dense/mesh_dense.obj'),
        ('explicit 3D prototype', OUT / 'prototype.obj'),
        ('3D-guided dense', OUT / 'dense/mesh_dense.obj'),
    ]
    views = [('oblique',25,35),('front',15,0),('right',15,90),('top',85,0),('rear',15,180)]
    stems = {'front':'v00_front_lo','right':'v02_right_lo','top':'v_top'}
    size, header = 360, 34
    sheet = Image.new('RGB', (size*len(views), (size+header)*len(cases)), 'white')
    draw = ImageDraw.Draw(sheet)
    metrics = {}
    for i, (name, path) in enumerate(cases):
        mesh = trimesh.load(path, force='mesh')
        parts = mesh.split(only_watertight=False)
        row = {'mesh':str(path), 'components':len(parts), 'silhouettes':{},
               'volume_litres':round(abs(float(mesh.volume))*1000,3)}
        for j, (view,elev,azim) in enumerate(views):
            eye, up = camera_from_elev_azim(center,radius,elev,azim)
            rendered = Image.fromarray(render_lit(mesh,eye,center,up,size=size,
                fit_extent=scale,margin=1.15,color=(.47,.51,.55))).convert('RGB')
            sheet.paste(rendered, (j*size,i*(size+header)+header))
            draw.text((j*size+9,i*(size+header)+9),f'{name}: {view}',fill='#20252b')
            if view in stems:
                target = Image.open(BASE/'open_arm/input'/f'{stems[view]}.png').convert('RGB')
                row['silhouettes'][view] = silhouette_scores(rendered,target.resize((size,size)))
        metrics[name] = row
    sheet.save(OUT/'comparison.png')
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    rows = ''.join('<tr><td><a href="/'+str(path.relative_to(ROOT))+'">'+name+'</a></td>'
        + ''.join(f'<td>{metrics[name]["silhouettes"][v]["iou"]:.3f}</td>'
                  for v in ('front','right','top'))
        + f'<td>{metrics[name]["components"]}</td></tr>' for name,path in cases)
    (OUT/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
      '<title>Chair coherent 3D proxy dense test</title><style>body{font:16px system-ui;'
      'max-width:1850px;margin:2rem auto;background:#f4f6f8;color:#1d2730}'
      'article{background:white;padding:1rem}img{max-width:100%}'
      'table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.5rem}</style>'
      '<article><h1>의자: 명시적 3D 기준으로 dense 검증</h1>'
      '<p>팔걸이 허용 영역을 넓히고 네 다리·팔걸이·등받이 연결 및 좌석 뒤 빈 공간을 가진 '
      '3D 기준을 사용했습니다. 같은 입력 영상과 BC로 dense를 비교합니다. FEA·sparse는 제외했습니다.</p>'
      '<table><tr><th>OBJ</th><th>앞 IoU</th><th>오른쪽 IoU</th><th>위 IoU</th>'
      '<th>성분 수</th></tr>'+rows+'</table><p><a href="metrics.json">수치 JSON</a> · '
      '<a href="prototype_metrics.json">3D 기준 검증</a></p>'
      '<a href="comparison.png"><img src="comparison.png"></a></article></html>')
    print(json.dumps({'comparison':str(OUT/'comparison.png'),
                      'html':str(OUT/'index.html'),'metrics':metrics},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'report'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--variant', choices=['v1', 'v2', 'v3', 'v4'], default='v1')
    args = parser.parse_args()
    if args.variant == 'v2':
        OUT = BASE / 'coherent_proxy_dense_v2'
    elif args.variant == 'v3':
        OUT = BASE / 'coherent_proxy_dense_v3'
    elif args.variant == 'v4':
        OUT = BASE / 'coherent_proxy_dense_v4'
    if args.command == 'prepare':
        prepare()
    elif args.command == 'run':
        run(args.gpu)
    else:
        report()
