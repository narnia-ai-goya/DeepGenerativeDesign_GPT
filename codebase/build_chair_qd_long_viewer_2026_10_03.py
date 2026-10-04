"""Interactive viewer for strict-eligible 128³ chair QD display meshes."""
from __future__ import annotations

import html
import json

import trimesh

from chair_qd_long_protocol_2026_10_03 import OUT


def main() -> None:
    rows = []
    for directory, prefix in ((OUT, ''), (OUT / 'feedback_round_02', 'feedback_round_02/')):
        result = json.loads((directory / 'result.json').read_text())
        for row in result['rows']:
            if not row['strict_eligible']:
                continue
            mesh_path = directory / 'mesh_cases' / row['id'] / 'combined_highres128.obj'
            glb_path = mesh_path.with_suffix('.glb')
            trimesh.load(mesh_path, force='mesh', process=False).export(glb_path)
            rows.append({'id': row['id'], 'method': row['method'],
                         'glb': prefix + 'mesh_cases/' + row['id'] + '/' + glb_path.name,
                         'obj': prefix + 'mesh_cases/' + row['id'] + '/' + mesh_path.name,
                         'cell': row['cell'], 'mass': row['mass_liters'],
                         'quality': row['worst_compliance_ratio']})
    (OUT / 'viewer_manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    buttons = ''.join(f'<button data-id="{html.escape(row["id"])}">{html.escape(row["id"])}</button>' for row in rows)
    (OUT / 'viewer.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Chair QD mesh viewer</title>
<style>body{margin:0;font:15px/1.5 system-ui;color:#172936;background:#e8eef2}header{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:12px 16px;background:white;border-bottom:1px solid #cbd8df}button{padding:8px 11px;border:1px solid #aebec7;border-radius:7px;background:#f7fafb;cursor:pointer}button.active{background:#24648a;color:white}#stage{width:100vw;height:calc(100vh - 120px);min-height:550px}#info{margin-left:auto}a{color:#155f94}</style>
<script type="importmap">{"imports":{"three":"/experiments/bracket/direct3ds2_design_eq_bracket_corrected_feaoff_2026-09-15/viewer/vendor/three.module.js","three/addons/":"/experiments/bracket/direct3ds2_design_eq_bracket_corrected_feaoff_2026-09-15/viewer/vendor/"}}</script>
<header><a href="index.html">← 보고서</a><b>3D 의자 메쉬</b>''' + buttons + '''<span id="info"></span></header><div id="stage"></div>
<script type="module">import * as THREE from 'three';import {OrbitControls} from 'three/addons/OrbitControls.js';import {GLTFLoader} from 'three/addons/GLTFLoader.js';
const cases=''' + json.dumps(rows, ensure_ascii=False) + ''';const stage=document.querySelector('#stage');const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;stage.appendChild(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color(0xe8eef2);scene.add(new THREE.HemisphereLight(0xffffff,0x687987,2.5));for(const [pos,intensity] of [[[3,-4,5],2.5],[[-3,2,3],1.4]]){const light=new THREE.DirectionalLight(0xffffff,intensity);light.position.set(...pos);scene.add(light)}
const camera=new THREE.PerspectiveCamera(38,1,.001,100);const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;let object;
function resize(){renderer.setSize(stage.clientWidth,stage.clientHeight);camera.aspect=stage.clientWidth/stage.clientHeight;camera.updateProjectionMatrix()}window.addEventListener('resize',resize);resize();
function show(id){const row=cases.find(x=>x.id===id);document.querySelectorAll('button[data-id]').forEach(b=>b.classList.toggle('active',b.dataset.id===id));document.querySelector('#info').innerHTML=`cell [${row.cell}] · mass ${row.mass.toFixed(2)} L · compliance/ref ${row.quality.toFixed(3)} · <a href="${row.obj}">OBJ</a>`;new GLTFLoader().load(row.glb,g=>{if(object)scene.remove(object);object=g.scene;object.traverse(x=>{if(x.isMesh)x.material=new THREE.MeshStandardMaterial({color:0x8c9ca9,metalness:.28,roughness:.6,side:THREE.DoubleSide})});scene.add(object);const box=new THREE.Box3().setFromObject(object),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3()),radius=Math.max(size.x,size.y,size.z);controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(radius*1.5,-radius*2.1,radius*1.1));camera.near=radius/1000;camera.far=radius*100;camera.updateProjectionMatrix();controls.update()})}
document.querySelectorAll('button[data-id]').forEach(b=>b.onclick=()=>show(b.dataset.id));show(cases[0].id);renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera)});
</script></html>''')
    print(OUT / 'viewer.html')


if __name__ == '__main__':
    main()
