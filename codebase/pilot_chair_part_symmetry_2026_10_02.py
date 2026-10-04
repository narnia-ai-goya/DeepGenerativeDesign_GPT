#!/usr/bin/env python3
"""Training-free SCULPT/SNAP3D-inspired chair assembly diagnostic.

This is a geometric pilot, not an implementation of either paper's trained model.
The existing whole-chair mesh supplies a shared coordinate frame.  A side band is
copied by symmetry, and the central part is kept from the whole-chair mesh.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage
from trimesh.voxel import VoxelGrid, encoding

from make_chair_domain import ROOT
from validate_chair_bc_geometry import audit

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SOURCE = BASE / 'form_embodies_multi_2026-10-02/open_arm_fea_off_sparse/physical_sparse.obj'
BC = BASE / 'consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
OUT = BASE / 'part_symmetry_pilot_2026-10-02'
PITCH = 0.005
SIDE_EDGE = 0.135  # m from chair center plane; central seat remains unchanged.
SEAM_HALF_WIDTH = 0.010  # m: source and center overlap here.


def as_mesh(mask: np.ndarray, grid: VoxelGrid) -> trimesh.Trimesh:
    mesh = VoxelGrid(encoding.DenseEncoding(mask), transform=grid.transform).marching_cubes
    # trimesh's marching_cubes property returns index-space vertices.
    mesh.apply_transform(grid.transform)
    return mesh


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    whole = trimesh.load(SOURCE, force='mesh', process=False)
    grid = whole.voxelized(PITCH).fill()
    occ = grid.matrix.astype(bool)
    nx = occ.shape[0]
    x = grid.indices_to_points(np.stack([np.arange(nx), np.zeros(nx), np.zeros(nx)], axis=1))[:, 0]
    reflect = np.abs(x[:, None] + x[None, :]).argmin(axis=1)
    xx = x[:, None, None]
    source_right = occ[reflect, :, :]
    source_left = occ[reflect, :, :]

    # Explicit part masks: the two side frames overlap the central body at the seam.
    central = occ & (np.abs(xx) <= SIDE_EDGE + SEAM_HALF_WIDTH)
    right_part = occ & (xx >= SIDE_EDGE - SEAM_HALF_WIDTH)
    left_part = occ & (xx <= -SIDE_EDGE + SEAM_HALF_WIDTH)
    for name, mask in [('central', central), ('right_original', right_part), ('left_original', left_part)]:
        as_mesh(mask, grid).export(OUT / f'{name}.obj')

    variants = {'original_voxelized': occ}
    for name, side in [('right_to_left', 'right'), ('left_to_right', 'left')]:
        if side == 'right':
            copied = source_right & (xx <= -SIDE_EDGE + SEAM_HALF_WIDTH)
            retained = occ & (xx >= -SIDE_EDGE - SEAM_HALF_WIDTH)
        else:
            copied = source_left & (xx >= SIDE_EDGE - SEAM_HALF_WIDTH)
            retained = occ & (xx <= SIDE_EDGE + SEAM_HALF_WIDTH)
        assembled = copied | retained
        # Restrict bridge repair to the junction band.  Never smooth the full chair.
        seam = np.abs(xx + SIDE_EDGE if side == 'right' else xx - SIDE_EDGE) <= 2 * SEAM_HALF_WIDTH
        repaired = ndimage.binary_closing(assembled, structure=ndimage.generate_binary_structure(3, 1), iterations=1)
        assembled = np.where(seam, repaired, assembled)
        variants[name] = assembled
        as_mesh(copied, grid).export(OUT / f'{name}_copied_side.obj')

    domain = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = domain.bounds.mean(axis=0)
    radius = float(np.linalg.norm(domain.extents)) * 1.5
    fit = float(domain.extents.max()) / 2
    views = [('front', 15, 0), ('right', 15, 90), ('top', 85, 0)]
    tile, label_h = 340, 30
    sheet = Image.new('RGB', (tile * 3, (tile + label_h) * len(variants)), 'white')
    draw = ImageDraw.Draw(sheet)
    rows = []
    for row, (name, mask) in enumerate(variants.items()):
        mesh = as_mesh(mask, grid)
        path = OUT / f'{name}.obj'
        mesh.export(path)
        mesh.export(OUT / f'{name}.glb')
        reflected = mask[reflect, :, :]
        symmetry_iou = float(np.count_nonzero(mask & reflected) / max(np.count_nonzero(mask | reflected), 1))
        check = audit(path, BC)
        comp, ncomp = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
        item = {'name': name, 'mesh': str(path), 'volume_liters': float(mask.sum() * PITCH**3 * 1000),
                'symmetry_iou': symmetry_iou, 'voxel_components': int(ncomp),
                'largest_component_fraction': float(np.bincount(comp.ravel())[1:].max() / mask.sum()),
                'bc_pass': bool(check['bc_geometry_pass']),
                'bc_coverage': {k: float(v['coverage']) for k, v in check['regions'].items()}}
        rows.append(item)
        for col, (view, elev, azim) in enumerate(views):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rgb = render_lit(mesh, eye, center, up, size=tile, fit_extent=fit,
                             margin=1.15, color=(.54, .58, .62))
            sheet.paste(Image.fromarray(rgb).convert('RGB'), (tile * col, (tile + label_h) * row + label_h))
            draw.text((tile * col + 8, (tile + label_h) * row + 7), f'{name} / {view}', fill='#20252b')

    sheet.save(OUT / 'comparison.png')
    (OUT / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    trs = ''.join('<tr><td>' + html.escape(v['name']) + '</td><td>' + f"{v['symmetry_iou']:.3f}" +
                  '</td><td>' + f"{v['volume_liters']:.2f}" + '</td><td>' + str(v['voxel_components']) +
                  '</td><td>' + f"{v['largest_component_fraction']:.1%}" + '</td><td>' +
                  f"{v['bc_coverage']['seat_load']:.1%}/{v['bc_coverage']['backrest_load']:.1%}" +
                  '</td><td>' + str(v['bc_pass']) + '</td><td><a href="' + v['name'] + '.obj">OBJ</a></td></tr>'
                  for v in rows)
    fea_path = OUT / 'fea_metrics.json'
    fea_html = ''
    if fea_path.exists():
        fea = json.loads(fea_path.read_text())
        ref = fea[0].get('compliance_proxy')
        fea_rows = ''.join('<tr><td>' + html.escape(v['name']) + '</td><td>' +
                           (f"{v['compliance_proxy']:.3e}" if 'compliance_proxy' in v else '실패') +
                           '</td><td>' + (f"{(v['compliance_proxy']/ref-1)*100:+.2f}%"
                           if ref and 'compliance_proxy' in v else '—') + '</td></tr>' for v in fea)
        fea_html = ('<article><h2>동일 조건 FEM proxy</h2><p>동일 64³ domain, 동일 800 N 좌판 하중. '
                    '5 mm 격자화 메시를 다시 표본화한 비교값이며 독립 메시 FEA가 아니다.</p><table><tr><th>조건</th>'
                    '<th>Compliance proxy</th><th>원본 대비</th></tr>' + fea_rows + '</table></article>')
    (OUT / 'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair part symmetry pilot</title>
<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;background:#f2f4f6;color:#1c2733}article{background:white;border-radius:12px;padding:1.3rem;margin:1rem 0}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.4rem}</style>
<article><h1>의자 부품 대칭 조립 파일럿</h1><p>기존 전체 의자 OBJ를 공통 좌표계로 사용한다. 한쪽 측면을 반사 복제하고 중앙 좌판·등받이 부분과 좁은 겹침 영역에서 결합했다. 새 부품 생성이나 학습은 하지 않았다. 5 mm 격자화 때문에 첫 줄도 원본 OBJ와 완전히 같지는 않다.</p><p><a href="viewer.html">3D 메시 뷰어</a> · <a href="REPORT.md">보고서</a> · <a href="metrics.json">수치</a> · <a href="central.obj">중앙 부품</a> · <a href="right_original.obj">오른쪽 부품</a> · <a href="left_original.obj">왼쪽 부품</a></p></article>
<article><img src="comparison.png"></article><article><table><tr><th>조건</th><th>대칭 IoU</th><th>체적 L</th><th>격자 성분</th><th>최대 성분</th><th>좌판/등받이 BC</th><th>BC 통과</th><th>OBJ</th></tr>''' + trs + '</table></article>' + fea_html + '</html>')
    (OUT / 'viewer.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair symmetry 3D viewer</title>
<style>body{margin:0;font:16px system-ui;background:#e9edf1;color:#17212b}header{padding:12px 20px;background:white;display:flex;gap:15px;align-items:center}button{padding:9px 14px;cursor:pointer}button.active{background:#243d57;color:white}#stage{width:100vw;height:calc(100vh - 65px)}canvas{width:100%;height:100%}</style>
<script type="importmap">{"imports":{"three":"/experiments/bracket/direct3ds2_design_eq_bracket_corrected_feaoff_2026-09-15/viewer/vendor/three.module.js","three/addons/":"/experiments/bracket/direct3ds2_design_eq_bracket_corrected_feaoff_2026-09-15/viewer/vendor/"}}</script>
<header><a href="index.html">← 보고서</a><b>의자 3D 메시 비교</b><button data-name="original_voxelized">격자화 원본</button><button data-name="right_to_left">오른쪽 → 왼쪽</button><button data-name="left_to_right">왼쪽 → 오른쪽</button><span>드래그 회전 · 휠 확대</span></header><div id="stage"></div>
<script type="module">import * as THREE from 'three';import {OrbitControls} from 'three/addons/OrbitControls.js';import {GLTFLoader} from 'three/addons/GLTFLoader.js';
const stage=document.querySelector('#stage'),renderer=new THREE.WebGLRenderer({antialias:true});stage.appendChild(renderer.domElement);renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;
const scene=new THREE.Scene();scene.background=new THREE.Color(0xe9edf1);const camera=new THREE.PerspectiveCamera(40,1,.001,100),controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xffffff,0x6b7785,3));for(const [p,i] of [[[2,-3,4],3],[[-3,2,2],1.7]]){const l=new THREE.DirectionalLight(0xffffff,i);l.position.set(...p);scene.add(l)}
let object;function resize(){renderer.setSize(stage.clientWidth,stage.clientHeight);camera.aspect=stage.clientWidth/stage.clientHeight;camera.updateProjectionMatrix()}window.addEventListener('resize',resize);resize();
function show(name){document.querySelectorAll('button[data-name]').forEach(b=>b.classList.toggle('active',b.dataset.name===name));new GLTFLoader().load(name+'.glb',g=>{if(object)scene.remove(object);object=g.scene;object.traverse(o=>{if(o.isMesh)o.material=new THREE.MeshStandardMaterial({color:0x697582,metalness:.18,roughness:.67,side:THREE.DoubleSide})});scene.add(object);const box=new THREE.Box3().setFromObject(object),c=box.getCenter(new THREE.Vector3()),s=box.getSize(new THREE.Vector3()),r=Math.max(s.x,s.y,s.z);controls.target.copy(c);camera.position.copy(c).add(new THREE.Vector3(r*1.4,-r*1.9,r*1.2));camera.near=r/1000;camera.far=r*100;camera.updateProjectionMatrix();controls.update()})}
document.querySelectorAll('button[data-name]').forEach(b=>b.onclick=()=>show(b.dataset.name));show('original_voxelized');renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera)});
</script></html>''')
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    main()
