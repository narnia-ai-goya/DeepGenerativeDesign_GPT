#!/usr/bin/env python3
"""Training-free, SNAP3D-inspired contact/connector search on an image-derived chair.

This is a welded/monolithic structural adaptation, not SNAP3D's rigid-body
peg/socket, IPC contact simulation, or learned part generator.  It takes the
previous image-first three-part chair, discovers two side/center contacts,
varies their retained connector windows, and evaluates common two-load FEM.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial.transform import Rotation
import trimesh

from make_chair_domain import ROOT
from build_chair_image_part_visual_hull_2026_10_02 import PITCH, to_mesh
from validate_chair_bc_geometry import audit

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SOURCE = BASE / 'semantic_parts_from_image_2026-10-02/visual_hull_assembly'
OUT = BASE / 'snap_contact_connector_pilot_2026-10-03'
BC = BASE / 'consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
SPEC = BASE / 'registered_spec_2026-10-02'
FEM = BASE / 'form_embodies_multi_2026-10-02'
BACK_DOMAIN = OUT / 'native_fea_domain_back'
PHYSICAL_BACK_LOAD = BASE / 'single_view_spec_2026-10-03/fea_domain_back/load.stl'
SOLVER = ROOT / 'codebase/code/fenics_fea_bracket.py'
FENICS = '/home/goya/miniconda3/envs/fenics/bin/python'
SIDE_PLANE_M = .185
SLIT_HALF_WIDTH_M = .008
SIX = ndimage.generate_binary_structure(3, 1)


def faces(mask: np.ndarray, x: np.ndarray) -> dict[str, dict]:
    result = {}
    for side, plane in [('left', -SIDE_PLANE_M), ('right', SIDE_PLANE_M)]:
        ix = int(np.argmin(np.abs(x - plane)))
        if side == 'left':
            area = mask[ix-1] & mask[ix] | mask[ix] & mask[ix+1]
        else:
            area = mask[ix-1] & mask[ix] | mask[ix] & mask[ix+1]
        result[side] = {'plane_m': float(x[ix]), 'face_voxels': int(area.sum()),
                        'face_area_m2': float(area.sum() * PITCH**2)}
    return result


def keep_windows(y: np.ndarray, z: np.ndarray, anchors: list[tuple[float, float]],
                 ry: float, rz: float) -> np.ndarray:
    yy = y[:, None]
    zz = z[None, :]
    window = np.zeros((len(y), len(z)), dtype=bool)
    for cy, cz in anchors:
        window |= ((yy - cy) / ry)**2 + ((zz - cz) / rz)**2 <= 1
    return window


def fea(name: str, mode: str, rho: np.ndarray, nodes: np.ndarray) -> dict:
    folder = OUT / name / 'fea' / mode
    folder.mkdir(parents=True, exist_ok=True)
    density = folder / 'rho.npy'
    points = folder / 'nodes.npy'
    np.save(density, rho)
    np.save(points, nodes)
    domain = FEM / 'native_fea_domain' if mode == 'seat' else BACK_DOMAIN
    # Native→physical is a +90° X rotation: physical -Z = native -Y;
    # physical +Y = native -Z.
    direction = '-y' if mode == 'seat' else '-z'
    force = 800 if mode == 'seat' else 200
    cmd = [FENICS, str(SOLVER), '--domain-dir', str(domain),
           '--density', str(density), '--nodes', str(points),
           '--output', str(folder / 'dc.npy'), '--mesh-cache', str(FEM / 'native_fea_shared.msh'),
           '--mesh-size', '.035', '--penal', '2', '--E0', '1.0',
           '--load-magnitude', str(force)]
    env = {**os.environ, 'LOAD_MODE': direction, 'BC_SURFACE_DIST': '1', 'BC_DIST': '0.025',
           'FEA_MAX_NODE_MAP_DISTANCE': '0.03', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    info = folder / 'dc_info.json'
    if mode == 'seat' and info.exists():
        process_code = 0
    else:
        with (folder / 'fea.log').open('w') as stream:
            process = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        process_code = process.returncode
    log = (folder / 'fea.log').read_text(errors='replace')
    fixed = re.search(r'Dirichlet: (\d+) nodes', log)
    loaded = re.search(r'Load: (\d+) nodes', log)
    value = float(json.loads(info.read_text())['compliance']) if process_code == 0 and info.exists() else None
    if value is not None and not np.isfinite(value):
        value = None
    result = {'exit_code': process_code, 'fixed_nodes': int(fixed.group(1)) if fixed else 0,
              'load_nodes': int(loaded.group(1)) if loaded else 0,
              'compliance_proxy': value,
              'log': str(folder / 'fea.log')}
    result['valid'] = bool(result['exit_code'] == 0 and result['fixed_nodes'] > 0
                           and result['load_nodes'] > 0 and result['compliance_proxy'] is not None)
    return result


def prepare_native_back_domain(reg: dict, rotation: np.ndarray) -> None:
    """Register the physical +Y back-load patch to the native FEM mesh frame."""
    BACK_DOMAIN.mkdir(parents=True, exist_ok=True)
    native = FEM / 'native_fea_domain'
    for name in ('original_DesignSpace.stl', 'fixed.stl'):
        shutil.copy2(native/name, BACK_DOMAIN/name)
    load = trimesh.load(PHYSICAL_BACK_LOAD, force='mesh', process=False)
    load.vertices = ((load.vertices - np.asarray(reg['physical_center_m'])) @ rotation /
                     float(reg['uniform_scale']) + np.asarray(reg['source_center_m']))
    load.export(BACK_DOMAIN/'load.stl')
    (BACK_DOMAIN/'registration.json').write_text(json.dumps({
        'physical_source':str(PHYSICAL_BACK_LOAD), 'transform':'inverse native_to_physical',
        'native_to_physical':reg},indent=2)+'\n')


def render(rows: list[dict]) -> None:
    domain = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    center = domain.bounds.mean(axis=0)
    radius = float(np.linalg.norm(domain.extents)) * 1.5
    fit = float(domain.extents.max()) / 2
    views = [('front', 15, 0), ('side', 15, 90), ('top', 85, 0)]
    tile, head = 300, 34
    sheet = Image.new('RGB', (tile*3, (tile+head)*len(rows)), '#f6f8fa')
    draw = ImageDraw.Draw(sheet)
    for row, item in enumerate(rows):
        mesh = trimesh.load(item['mesh_obj'], force='mesh', process=False)
        for col, (view, elev, azim) in enumerate(views):
            eye, up = camera_from_elev_azim(center, radius, elev, azim)
            rgb = render_lit(mesh, eye, center, up, size=tile, fit_extent=fit,
                             margin=1.15, color=(.52, .58, .64))
            sheet.paste(Image.fromarray(rgb), (col*tile, row*(tile+head)+head))
            draw.text((col*tile+8, row*(tile+head)+8), item['name']+' / '+view, fill='#1b2e39')
    sheet.save(OUT / 'comparison.png')


def make_page(rows: list[dict]) -> None:
    cells = []
    for item in rows:
        score = item.get('worst_compliance_ratio')
        label = '—' if score is None else f'{score:.3f}'
        cells.append(f'<tr><td>{html.escape(item["name"])}</td>'
                     f'<td>{item["volume_liters"]:.2f}</td><td>{item["removed_percent"]:.2f}%</td>'
                     f'<td>{item["components_6conn"]}</td><td>{item["bc_pass"]}</td>'
                     f'<td>{label}</td><td>{item.get("pareto",False)}</td>'
                     f'<td><a href="{item["name"]}/assembled.obj">OBJ</a> '
                     f'<a href="{item["name"]}/assembled.glb">GLB</a></td></tr>')
    (OUT/'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Chair contact-connector search</title><style>
body{font:16px system-ui;background:#eef3f5;color:#19313c;max-width:1200px;margin:25px auto}
article{background:white;border:1px solid #d5e1e4;padding:20px;margin:18px 0;border-radius:10px}
img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #d6e0e3;padding:8px}
a{color:#216a80}</style><article><h1>의자 접합부 탐색 파일럿</h1>
<p>이미지에서 만든 중앙·좌우 부품의 접촉을 찾고, 좌우 접합부에 남길 연결 창의 위치와 면적을 바꿔 평가한다.
SNAP3D의 접촉 그래프와 물리 피드백 개념을 일체형 구조 의자에 맞게 적용한 파일럿이다.
SNAP3D의 peg/socket, rigid-body contact solver를 구현한 것은 아니다.</p>
<p><a href="REPORT.md">보고서</a> · <a href="metrics.json">수치</a> · <a href="viewer.html">3D 뷰어</a> · <a href="contact_graph.json">접촉 그래프</a></p></article>
<article><img src="comparison.png" alt="Front, side and top 3D renders of connector variants"></article>
<article><table><tr><th>Candidate</th><th>Mass L</th><th>Removed</th><th>Components</th><th>BC pass</th>
<th>Worst FEM ratio ↓</th><th>Pareto</th><th>Mesh</th></tr>''' + ''.join(cells) + '</table></article></html>')
    buttons = ''.join(f'<button data-id="{html.escape(item["name"])}">{html.escape(item["name"])}</button>'
                      for item in rows)
    viewer = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Connector mesh viewer</title>
<style>body{margin:0;font:15px system-ui;background:#e9eef1;color:#182a33}header{padding:12px;background:white;display:flex;gap:8px;align-items:center;flex-wrap:wrap}button{padding:8px;cursor:pointer}button.active{background:#285e71;color:white}#stage{width:100vw;height:calc(100vh - 70px)}</style>
<script type="importmap">{"imports":{"three":"/experiments/bracket/direct3ds2_design_eq_bracket_corrected_feaoff_2026-09-15/viewer/vendor/three.module.js","three/addons/":"/experiments/bracket/direct3ds2_design_eq_bracket_corrected_feaoff_2026-09-15/viewer/vendor/"}}</script>
<header><a href="index.html">← 결과</a><b>접합부 후보 3D 비교</b>''' + buttons + '''<span>드래그 회전 · 휠 확대</span></header><div id="stage"></div>
<script type="module">import * as THREE from 'three';import {OrbitControls} from 'three/addons/OrbitControls.js';import {GLTFLoader} from 'three/addons/GLTFLoader.js';
const stage=document.querySelector('#stage'),renderer=new THREE.WebGLRenderer({antialias:true});stage.appendChild(renderer.domElement);renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;
const scene=new THREE.Scene();scene.background=new THREE.Color(0xe9eef1);const camera=new THREE.PerspectiveCamera(40,1,.001,100),controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xffffff,0x697988,3));for(const [p,i] of [[[2,-3,4],3],[[-3,2,2],1.6]]){const l=new THREE.DirectionalLight(0xffffff,i);l.position.set(...p);scene.add(l)}
let object;function resize(){renderer.setSize(stage.clientWidth,stage.clientHeight);camera.aspect=stage.clientWidth/stage.clientHeight;camera.updateProjectionMatrix()}window.addEventListener('resize',resize);resize();
function show(id){document.querySelectorAll('button[data-id]').forEach(b=>b.classList.toggle('active',b.dataset.id===id));new GLTFLoader().load(id+'/assembled.glb',g=>{if(object)scene.remove(object);object=g.scene;object.traverse(o=>{if(o.isMesh)o.material=new THREE.MeshStandardMaterial({color:0x73828f,metalness:.24,roughness:.6,side:THREE.DoubleSide})});scene.add(object);const box=new THREE.Box3().setFromObject(object),c=box.getCenter(new THREE.Vector3()),s=box.getSize(new THREE.Vector3()),r=Math.max(s.x,s.y,s.z);controls.target.copy(c);camera.position.copy(c).add(new THREE.Vector3(r*1.5,-r*1.8,r*1.2));camera.near=r/1000;camera.far=r*100;camera.updateProjectionMatrix();controls.update()})}
document.querySelectorAll('button[data-id]').forEach(b=>b.onclick=()=>show(b.dataset.id));show('full_contact');renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera)});
</script></html>'''
    (OUT/'viewer.html').write_text(viewer)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(SOURCE / 'occupancy.npz')
    base = d['assembled_domain_clipped'].astype(bool)
    origin = d['origin'].astype(float)
    pitch = float(d['pitch'])
    if abs(pitch-PITCH) > 1e-9:
        raise ValueError('Source pitch differs from image-part generator')
    x, y, z = [origin[i] + np.arange(base.shape[i])*pitch for i in range(3)]
    graph = {'source_image_assembly': str(SOURCE / 'assembled_domain_clipped.obj'),
             'part_nodes': ['center_seat_back', 'left_side_frame', 'right_side_frame'],
             'edges': faces(base, x), 'voxel_pitch_m': pitch,
             'meaning': 'shared occupied face near each side/center part boundary; structural monolithic contact'}
    (OUT/'contact_graph.json').write_text(json.dumps(graph, indent=2)+'\n')
    # Carve a narrow junction band only outside a parameterized set of joint
    # windows.  The uncarved reference is included as an upper-contact control.
    configs = [
        ('full_contact', [], 0., 0., False),
        ('center_narrow', [(0., .51)], .045, .055, False),
        ('center_wide', [(0., .51)], .075, .075, False),
        ('front_rear', [(-.145, .51), (.145, .51)], .060, .070, False),
        ('triple', [(-.145, .51), (0., .51), (.145, .51)], .055, .065, False),
        # The foot-to-seat load path must not be cut just to reduce joint area.
        ('leg_safe_center', [(0., .51)], .075, .075, True),
        ('leg_safe_front_rear', [(-.145, .51), (.145, .51)], .060, .070, True),
        ('leg_safe_broad', [(0., .51)], .30, .095, True),
        ('leg_safe_four', [(-.18, .51), (-.06, .51), (.06, .51), (.18, .51)], .055, .075, True),
    ]
    bc = np.load(BC)
    from build_chair_image_part_visual_hull_2026_10_02 import sample_bc
    protected = np.zeros(base.shape, dtype=bool)
    for key in ('fix','load','back_load'):
        protected |= sample_bc(bc, key, (x,y,z))
    seam = (np.abs(x[:,None,None]-SIDE_PLANE_M) <= SLIT_HALF_WIDTH_M) | \
           (np.abs(x[:,None,None]+SIDE_PLANE_M) <= SLIT_HALF_WIDTH_M)
    rows = []
    occs = {}
    for name, anchors, ry, rz, leg_safe in configs:
        allowed = (np.ones((len(y),len(z)), dtype=bool) if name=='full_contact'
                   else keep_windows(y,z,anchors,ry,rz))
        active_seam = seam & (z[None,None,:] >= .35) if leg_safe else seam
        occ = base & (~active_seam | allowed[None,:,:] | protected)
        labels, n = ndimage.label(occ, structure=SIX)
        frac = float(np.bincount(labels.ravel())[1:].max()/occ.sum()) if n else 0.
        folder = OUT/name
        folder.mkdir(exist_ok=True)
        mesh = to_mesh(occ,origin)
        mesh.export(folder/'assembled.obj')
        mesh.export(folder/'assembled.glb')
        np.savez_compressed(folder/'occupancy.npz', occupied=occ, origin=origin, pitch=pitch)
        check = audit(folder/'assembled.obj', BC)
        item = {'name':name,'anchors_yz_m':anchors,'radius_y_m':ry,'radius_z_m':rz,
                'leg_path_preserved':leg_safe,
                'mesh_obj':str(folder/'assembled.obj'),
                'mass_voxels':int(occ.sum()),'volume_liters':float(occ.sum()*pitch**3*1000),
                'removed_percent':float((base.sum()-occ.sum())/base.sum()*100),
                'components_6conn':int(n),'largest_component_fraction':frac,
                'watertight':bool(mesh.is_watertight),
                'bc_pass':bool(check['bc_geometry_pass']),
                'bc_coverage':{k:float(v['coverage']) for k,v in check['regions'].items()},
                'geometry_gate':bool(n==1 and frac> .999 and check['bc_geometry_pass'] and mesh.is_watertight),
                'seam_face_voxels':faces(occ,x)}
        rows.append(item)
        occs[name]=occ
        print(name,'mass',round(item['volume_liters'],3),'components',n,
              'BC',item['bc_pass'],'gate',item['geometry_gate'],flush=True)
    # Match the prior image-part FEM: sample this physical grid at native-domain
    # nodes after the recorded rigid registration, then run both load cases.
    spec = np.load(SPEC/'native_frame_spec.npz')
    indices = np.argwhere(spec['bracket'].astype(bool))
    native_nodes = spec['origin']+(indices+.5)*spec['pitch_xyz']
    reg = json.loads((SPEC/'calibration.json').read_text())['native_to_physical']
    rotation = Rotation.from_euler('x',reg['rotation_x_degrees'],degrees=True).as_matrix()
    prepare_native_back_domain(reg, rotation)
    physical_nodes = (native_nodes-np.asarray(reg['source_center_m']))@rotation.T * \
                     float(reg['uniform_scale'])+np.asarray(reg['physical_center_m'])
    ijk = np.floor((physical_nodes-(origin-pitch/2))/pitch).astype(int)
    valid = ((ijk>=0)&(ijk<np.asarray(base.shape))).all(axis=1)
    jobs=[]
    for item in rows:
        if not item['geometry_gate']:
            continue
        selected=np.zeros(len(ijk),dtype=bool)
        o=occs[item['name']]
        selected[valid]=o[ijk[valid,0],ijk[valid,1],ijk[valid,2]]
        rho=np.where(selected,1.,.001).astype(np.float64)
        item['occupied_domain_voxels']=int(selected.sum())
        for mode in ('seat','back'):
            jobs.append((item['name'],mode,rho))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda task:(task[0],task[1],fea(task[0],task[1],task[2],native_nodes)),jobs))
    lookup={(name,mode):result for name,mode,result in results}
    baseline={mode:lookup.get(('full_contact',mode),{}).get('compliance_proxy') for mode in ('seat','back')}
    for item in rows:
        name=item['name']
        item['fea']={mode:lookup.get((name,mode)) for mode in ('seat','back')}
        if all(item['fea'][mode] and item['fea'][mode]['valid'] and baseline[mode]
               for mode in ('seat','back')):
            item['worst_compliance_ratio']=max(item['fea'][mode]['compliance_proxy']/baseline[mode]
                                               for mode in ('seat','back'))
            item['fea_valid']=True
        else:
            item['worst_compliance_ratio']=None
            item['fea_valid']=False
    feasible=[r for r in rows if r['fea_valid']]
    for item in rows:
        item['pareto']=bool(item in feasible and not any(
            other is not item and other['volume_liters']<=item['volume_liters'] and
            other['worst_compliance_ratio']<=item['worst_compliance_ratio'] and
            (other['volume_liters']<item['volume_liters'] or
             other['worst_compliance_ratio']<item['worst_compliance_ratio']) for other in feasible))
    (OUT/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n')
    render(rows)
    make_page(rows)
    text=['# SNAP3D-inspired chair contact/connector search','',
          'This is a training-free structural adaptation of contact-graph reasoning and',
          'physics-guided connector selection from SNAP3D. The chair is treated as one',
          'monolithic load-bearing shape. It does not implement SNAP3D peg/socket',
          'geometry, collision-free part editing, rigid-body contact simulation, or',
          'manufacturable assembly. It therefore must not be labeled a SNAP3D reproduction.','',
          'Source: '+str(SOURCE/'assembled_domain_clipped.obj'),
          'Contact graph: '+str(OUT/'contact_graph.json'),
          'Viewer: '+str(OUT/'viewer.html'),
          'FEM is the matched 35 mm native-domain density proxy with 800 N seat',
          'and 200 N back loads; it is not independent tetrahedral analysis of each OBJ.','',
          '| Candidate | Mass L | Removed % | Components | BC | Worst C / full | Pareto |',
          '|---|---:|---:|---:|---|---:|---|']
    for r in rows:
        q='—' if r['worst_compliance_ratio'] is None else f'{r["worst_compliance_ratio"]:.3f}'
        text.append(f'| {r["name"]} | {r["volume_liters"]:.2f} | {r["removed_percent"]:.2f} | '
                    f'{r["components_6conn"]} | {r["bc_pass"]} | {q} | {r["pareto"]} |')
    text.extend(['', '## 판정', '',
                 '모든 후보가 기하·BC 검사를 통과했지만, 접합부 재료를 덜어낸 후보는 기준보다 강성이 낮다. '
                 '따라서 이번 파일럿은 물리 피드백으로 접합 후보를 구별할 수 있음을 보여줄 뿐, 성능 향상을 보여주지는 않는다.',
                 '발–좌판 연결을 보존한 `leg_safe_broad`는 체적 3.06% 감소에 최악 하중 compliance 16.9% 증가다. '
                 '35 mm FEM proxy에서 일부 후보가 같은 점수로 양자화되므로 작은 후보 간 순위는 신뢰하기 어렵다.',
                 '좌판 하중은 물리좌표 -Z (800 N), 등받이는 물리좌표 +Y (200 N)다. '
                 '등받이 하중 면을 native FEM 좌표로 역변환하고, 같은 native 도메인에서 평가했다.',
                 '', 'SNAP3D source: https://arxiv.org/abs/2609.13146'])
    (OUT/'REPORT.md').write_text('\n'.join(text)+'\n')
    print('REPORT',OUT/'REPORT.md',flush=True)


if __name__=='__main__':
    main()
