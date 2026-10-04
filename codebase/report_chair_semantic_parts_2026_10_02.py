#!/usr/bin/env python3
"""Review page for image-first chair part generation and assembly pilot."""
from __future__ import annotations

import html
import json

from make_chair_domain import ROOT

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'semantic_parts_from_image_2026-10-02'
ASM = OUT / 'visual_hull_assembly'
OLD = BASE / 'form_embodies_multi_2026-10-02'


def main() -> None:
    metrics = json.loads((ASM / 'metrics.json').read_text())
    fea = json.loads((ASM / 'fea_metrics.json').read_text())
    fidelity = json.loads((ASM / 'image_fidelity.json').read_text())
    old = next(x for x in json.loads((OLD / 'metrics.json').read_text())
               if x['case'] == 'open_arm' and x['stage'] == 'sparse' and x['mode'] == 'fea_off')
    current = metrics['parts']['assembled_domain_clipped']
    proxy = fea[1]['compliance_proxy']
    change = 100 * (proxy / old['post_voxel_fea_compliance_proxy'] - 1)
    masks = ''.join('<figure><img src="../inputs/' + name + '"><figcaption>' + title + '</figcaption></figure>'
                    for name, title in [
                        ('side_frame_imagegen.png', '측면 프레임 · 측면'),
                        ('side_frame_front_imagegen.png', '측면 프레임 · 정면'),
                        ('seat_back_imagegen.png', '좌판·등받이 · 정면'),
                        ('seat_back_right_imagegen.png', '좌판·등받이 · 측면')])
    rows = ''.join('<tr><td>' + name + '</td><td>' + f'{v["volume_liters"]:.2f}' +
                   '</td><td>' + str(v['6_connected_components']) + '</td><td><a href="' +
                   name + '.obj">OBJ</a></td></tr>' for name, v in metrics['parts'].items())
    (ASM / 'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Image-first chair parts</title>
<style>body{font:16px/1.5 system-ui;max-width:1250px;margin:2rem auto;background:#f2f4f7;color:#1a2531}article{background:white;border-radius:12px;padding:1.3rem;margin:1rem 0}h1,h2{margin:.2rem 0 .8rem}figure{margin:0}figure img{width:220px;background:white;border:1px solid #ddd}figcaption{font-size:14px}img.wide{width:100%;height:auto}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.45rem .7rem}a{color:#155696}.gallery{display:flex;gap:1rem;flex-wrap:wrap}.note{background:#fff2d8;padding:1rem}</style>
<article><h1>이미지에서 처음부터 만든 의자 부품 · 대칭 조립</h1><p>원본 의자 이미지를 의미별 부품 이미지로 편집하고, 부품마다 3D를 생성했다. Direct3D-S2 단일/2시점 생성은 측면 프레임의 빈 공간을 판으로 메워 최종 조립에 사용하지 않았다. 대신 부품 이미지의 두 윤곽을 교차한 기하를 만들고, 한쪽 측면 프레임을 반사 복제해 결합했다. 기존 3D 의자 메시는 이 조립의 기하 입력으로 사용하지 않았다.</p>
<p><a href="viewer.html">3D 메시 뷰어</a> · <a href="REPORT.md">판정 보고서</a> · <a href="metrics.json">기하·BC JSON</a> · <a href="fea_metrics.json">FEM JSON</a> · <a href="image_fidelity.json">이미지 일치도 JSON</a></p></article>
<article><h2>실제 부품 입력 이미지</h2><div class="gallery">''' + masks + '''</div></article>
<article><h2>3D foundation model 부품 생성 진단</h2><p>첫째 줄은 측면 프레임, 둘째 줄은 좌판·등받이. 좌측부터 세 각도다. 단일 시점과 두 시점 모두 보이지 않는 면을 잘못 채웠다.</p><h3>단일 시점</h3><img class="wide" src="../sparse_parts_diagnostic.png"><h3>두 시점</h3><img class="wide" src="../sparse_two_view_diagnostic.png"></article>
<article><h2>윤곽 교차 → 복제 → 접합 → 영역 절단</h2><img class="wide" src="comparison.png"><p>다섯째 줄은 원래 design envelope로 절단한 결과다. 절단 전 재료의 24.4%가 영역 밖에 있었다. 설계 영역을 유지하면 일부 팔걸이 외형이 잘린다.</p></article>
<article><h2>기준선과 결과</h2><table><tr><th>항목</th><th>기존 일체형 sparse</th><th>부품 조립·영역 절단</th></tr>
<tr><td>체적</td><td>''' + f'{old["volume_liters"]:.2f}' + ''' L</td><td>''' + f'{current["volume_liters"]:.2f}' + ''' L</td></tr>
<tr><td>연결 성분</td><td>''' + str(old['components']) + '''</td><td>''' + str(current['6_connected_components']) + '''</td></tr>
<tr><td>정면/측면/상면 IoU</td><td>''' + '/'.join(f'{old["silhouette_iou"][v]:.3f}' for v in ('front','right','top')) + '''</td><td>''' + '/'.join(f"{fidelity['assembled_domain_clipped'][v]['iou']:.3f}" for v in ('front','right','top')) + '''</td></tr>
<tr><td>좌판 800 N FEM proxy ↓</td><td>''' + f'{old["post_voxel_fea_compliance_proxy"]:,.0f}' + '''</td><td>''' + f'{proxy:,.0f} ({change:+.1f}%)' + '''</td></tr>
<tr><td>BC 검사</td><td>통과</td><td>''' + ('통과' if metrics['bc_audit_clipped']['bc_geometry_pass'] else '실패') + '''</td></tr></table>
<p class="note">FEM은 같은 64³ 도메인·동일 solver를 쓴 후처리 proxy다. 독립 메시 FEA가 아니며 기하 생성 때 FEA를 사용하지 않았다. 새 후보의 FEM 수치 개선은 이 한 사례에서만 관찰됐고, 정면·상면 이미지 일치도는 악화됐다.</p></article>
<article><h2>각 부품 파일</h2><table><tr><th>형상</th><th>체적 L</th><th>6-연결 성분</th><th>파일</th></tr>''' + rows + '''</table></article></html>''')
    vendor = '/experiments/bracket/direct3ds2_design_eq_bracket_corrected_feaoff_2026-09-15/viewer/vendor/'
    (ASM / 'viewer.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Image-first chair 3D viewer</title>
<style>body{margin:0;font:15px system-ui;background:#e9edf1;color:#17212b}header{padding:10px 16px;background:white;display:flex;gap:8px;align-items:center;flex-wrap:wrap}button{padding:8px 12px;cursor:pointer}button.active{background:#233e56;color:white}#stage{width:100vw;height:calc(100vh - 100px)}canvas{width:100%;height:100%}</style>
<script type="importmap">{"imports":{"three":"''' + vendor + '''three.module.js","three/addons/":"''' + vendor + '''"}}</script>
<header><a href="index.html">← 보고서</a><b>의자 부품·조립 3D 뷰어</b><button data-name="center">좌판·등받이</button><button data-name="side_left">왼쪽 프레임</button><button data-name="side_right_mirrored">오른쪽 복제</button><button data-name="assembled_contact_repaired">조립 원형</button><button data-name="assembled_domain_clipped">영역 절단</button><span>드래그 회전 · 휠 확대</span></header><div id="stage"></div>
<script type="module">import * as THREE from 'three';import {OrbitControls} from 'three/addons/OrbitControls.js';import {GLTFLoader} from 'three/addons/GLTFLoader.js';
const stage=document.querySelector('#stage'),renderer=new THREE.WebGLRenderer({antialias:true});stage.appendChild(renderer.domElement);renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;
const scene=new THREE.Scene();scene.background=new THREE.Color(0xe9edf1);const camera=new THREE.PerspectiveCamera(40,1,.001,100),controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xffffff,0x6b7785,3));for(const [p,i] of [[[2,-3,4],3],[[-3,2,2],1.7]]){const l=new THREE.DirectionalLight(0xffffff,i);l.position.set(...p);scene.add(l)}
let object;function resize(){renderer.setSize(stage.clientWidth,stage.clientHeight);camera.aspect=stage.clientWidth/stage.clientHeight;camera.updateProjectionMatrix()}window.addEventListener('resize',resize);resize();
function show(name){document.querySelectorAll('button[data-name]').forEach(b=>b.classList.toggle('active',b.dataset.name===name));new GLTFLoader().load(name+'.glb',g=>{if(object)scene.remove(object);object=g.scene;object.traverse(o=>{if(o.isMesh)o.material=new THREE.MeshStandardMaterial({color:0x6f7b87,metalness:.18,roughness:.66,side:THREE.DoubleSide})});scene.add(object);const box=new THREE.Box3().setFromObject(object),c=box.getCenter(new THREE.Vector3()),s=box.getSize(new THREE.Vector3()),r=Math.max(s.x,s.y,s.z);controls.target.copy(c);camera.position.copy(c).add(new THREE.Vector3(r*1.4,-r*1.9,r*1.2));camera.near=r/1000;camera.far=r*100;camera.updateProjectionMatrix();controls.update()})}
document.querySelectorAll('button[data-name]').forEach(b=>b.onclick=()=>show(b.dataset.name));show('assembled_domain_clipped');renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera)});
</script></html>''')
    (ASM / 'REPORT.md').write_text(f'''# 이미지 기반 의자 부품 생성·대칭 조립 파일럿 (2026-10-02)

## 방법과 결과

기존 open-arm 의자의 정면·측면 **이미지**를 사용해 측면 프레임과 중앙 좌판·등받이 부품 이미지를 imagegen으로 별도 제작했다. 기존 3D 의자 메시는 기하 입력에 사용하지 않았다. 각 부품 이미지를 pretrained Direct3D-S2에 단일 시점과 두 시점으로 넣어 dense+sparse 메시까지 생성했다. 그러나 측면 프레임의 빈 공간을 판으로 메우고 중앙 좌판·등받이의 측면을 잘못 추정했다. 이 결과는 조립에 사용하지 않았으며 진단 비교군으로 보존했다. 학습은 수행하지 않았다.

대신 별도 부품 이미지의 정면·측면 윤곽을 같은 물리 좌표계에서 교차해 중앙 형상과 한쪽 측면 프레임을 만들었다. 한쪽 측면을 x=0 평면에 대해 반사 복제했다. 물리 BC의 발 고정부는 측면 프레임에, 좌판·등받이 하중 면은 중앙 부품에 합쳤다. 접합부에서는 중앙과 양 측면이 각각 {metrics['interface_overlap_voxels']['left']}/{metrics['interface_overlap_voxels']['right']}개의 4 mm voxel을 공유하여 별도 확대 보정 없이 한 개의 6-연결 성분이 됐다. BC geometry audit도 통과했다.

원형 조립안 체적은 {metrics['parts']['assembled_contact_repaired']['volume_liters']:.2f} L이며 그중 {metrics['outside_envelope_fraction']:.1%}가 원본 design envelope 바깥이다. 영역 절단 후 체적은 **{current['volume_liters']:.2f} L**, 한 개 성분, BC 통과다. 외형에서는 팔걸이와 등받이 일부가 절단되므로 설계 영역과 입력 이미지의 충돌을 다시 정의해야 한다.

동일한 64³ 도메인·800 N 좌판 하중·동일 후처리 FEM proxy에서 기존 일체형 sparse는 **{old['post_voxel_fea_compliance_proxy']:,.0f}**, 새 영역 절단 조립안은 **{proxy:,.0f} ({change:+.1f}%)**였다. 체적은 각각 {old['volume_liters']:.2f}/{current['volume_liters']:.2f} L로 비슷하다. 다만 새 후보의 정면/측면/상면 IoU는 {fidelity['assembled_domain_clipped']['front']['iou']:.3f}/{fidelity['assembled_domain_clipped']['right']['iou']:.3f}/{fidelity['assembled_domain_clipped']['top']['iou']:.3f}으로 기존 {old['silhouette_iou']['front']:.3f}/{old['silhouette_iou']['right']:.3f}/{old['silhouette_iou']['top']:.3f}보다 정면·상면이 떨어진다. 구조적 우월성을 주장하기에는 한 사례의 proxy 결과만 있고 독립 tetra FEA와 별도 하중 사례가 부족하다.

## 핵심 판정

복제·대칭 조립은 **기존 3D 메시를 재활용하지 않고 이미지에서 출발해 실제로 수행**했다. 현재 3D foundation model은 부품 단독 입력에서 닫힌 판을 환각하므로, 이 파일럿의 최종 조립은 이미지 윤곽 기반이다. SNAP3D식 접합부 검사를 단순한 겹침·연결성·BC·FEM proxy로 시작했지만 학습된 SCULPT 분할기나 SNAP3D의 peg/socket 및 rigid-body contact solver를 구현한 것은 아니다. 다음 우선순위는 이미지와 설계 영역이 동시에 허용하는 부품 인터페이스를 정의하고, 그 접합 단면을 FEA와 함께 탐색하는 것이다.

## 절대경로

- 보고서 페이지: `{ASM / 'index.html'}`
- 3D 메시 뷰어: `{ASM / 'viewer.html'}`
- 조립 단계 렌더: `{ASM / 'comparison.png'}`
- 영역 절단 최종 메시: `{ASM / 'assembled_domain_clipped.obj'}`
- 절단 전 메시: `{ASM / 'assembled_contact_repaired.obj'}`
- 기하·BC 기록: `{ASM / 'metrics.json'}`
- FEM proxy 기록: `{ASM / 'fea_metrics.json'}`
- 이미지 일치도: `{ASM / 'image_fidelity.json'}`
- 재현 코드: `{ROOT / 'codebase/run_chair_semantic_parts_2026_10_02.py'}`, `{ROOT / 'codebase/build_chair_image_part_visual_hull_2026_10_02.py'}`, `{ROOT / 'codebase/evaluate_chair_image_part_assembly_2026_10_02.py'}`
''')
    print(ASM / 'index.html')


if __name__ == '__main__':
    main()
