"""Summarize the two-round chair QD loop with Sparse in-step FEA."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageFont

from make_chair_domain import ROOT


PARENT = ROOT / 'experiments/chair/sofa_style_2026-09-28/text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03'
OUT = PARENT / 'sparse_fea_loop_2026-10-03'
HISTORY7 = {(0, 0), (0, 2), (1, 0), (1, 1), (2, 1), (0, 1), (1, 2)}


def font(size: int, bold: bool = False):
    face = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype(f'/usr/share/fonts/truetype/dejavu/{face}', size)


def fitted(path: str, width: int, height: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert('RGB')
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    return image


def figure(rows: list[dict]) -> Path:
    page = Image.new('RGB', (1800, 1800), '#edf2f5')
    draw = ImageDraw.Draw(page)
    draw.text((23, 14), 'Chair QD loop | original Sparse mesh (front / right)',
              font=font(28, True), fill='#17344a')
    for index, row in enumerate(rows):
        x, y = (index % 2) * 900 + 12, (index // 2) * 430 + 80
        draw.rounded_rectangle((x, y, x + 875, y + 415), radius=14, fill='white')
        draw.text((x + 16, y + 10), f"R{row['round']} {row['id']} | {row['method']}",
                  font=font(20, True), fill='#17344a')
        raw_preview = str(Path(row['aligned_mesh']).with_name('sparse_preview.png'))
        for source, left, width in ((row['image'], x + 15, 310), (raw_preview, x + 330, 530)):
            pic = fitted(source, width, 330)
            page.paste(pic, (left + (width - pic.width) // 2, y + 40 + (330 - pic.height) // 2))
        compliance = row.get('worst_compliance_ratio')
        value = f'{compliance:.3f}' if compliance is not None else 'invalid'
        line = (f"image {row['image_cell']} -> mesh {row['cell']} | "
                f"valid {row['strict_eligible']} | mass {row['mass_liters']:.1f} L | worst C {value}")
        draw.text((x + 16, y + 386), line, font=font(16), fill='#405766')
    path = OUT / 'selected_sparse_raw_eight.png'
    page.save(path)
    return path


def sparse_viewer(rows: list[dict]) -> Path:
    cases = []
    for row in rows:
        case = f"round_{row['round']:02d}/mesh_cases/{row['id']}"
        cases.append({'id': row['id'], 'round': row['round'], 'method': row['method'],
                      'glb': f'{case}/sparse_aligned.glb',
                      'raw_obj': f"round_{row['round']:02d}/{row['id']}/generation/mesh.obj",
                      'cell': row['cell'], 'mass': row['mass_liters'],
                      'compliance': row.get('worst_compliance_ratio')})
    buttons = ''.join(f'<button data-id="{html.escape(row["id"])}">R{row["round"]} {html.escape(row["id"])}</button>'
                      for row in rows)
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Raw Sparse chair meshes</title>
<style>body{margin:0;font:15px/1.5 system-ui;color:#172936;background:#e8eef2}header{display:flex;align-items:center;gap:7px;flex-wrap:wrap;padding:10px 15px;background:white;border-bottom:1px solid #cbd8df}button{padding:7px 10px;border:1px solid #aebec7;border-radius:7px;background:#f7fafb;cursor:pointer}button.active{background:#24648a;color:white}#stage{width:100%;height:calc(100vh - 112px);min-height:490px}#info{padding:5px 15px;background:white}a{color:#155f94}</style>
<script type="importmap">{"imports":{"three":"/experiments/bracket/direct3ds2_design_eq_bracket_corrected_feaoff_2026-09-15/viewer/vendor/three.module.js","three/addons/":"/experiments/bracket/direct3ds2_design_eq_bracket_corrected_feaoff_2026-09-15/viewer/vendor/"}}</script>
<header><b>원본 Sparse 메쉬 · 512³ stage</b>BUTTONS</header><div id="info">Loading…</div><div id="stage"></div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/OrbitControls.js';
import {GLTFLoader} from 'three/addons/GLTFLoader.js';
const cases=CASES;
const stage=document.querySelector('#stage');
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;stage.appendChild(renderer.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color(0xe8eef2);scene.add(new THREE.HemisphereLight(0xffffff,0x6b7b87,2.5));
for(const [position,power] of [[[3,-4,5],2.8],[[-3,2,3],1.4]]){const light=new THREE.DirectionalLight(0xffffff,power);light.position.set(...position);scene.add(light)}
const camera=new THREE.PerspectiveCamera(38,1,.001,100);const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
let object,request=0;
function resize(){renderer.setSize(stage.clientWidth,stage.clientHeight);camera.aspect=stage.clientWidth/stage.clientHeight;camera.updateProjectionMatrix()}window.addEventListener('resize',resize);resize();
function show(id){
 const row=cases.find(x=>x.id===id);const token=++request;
 document.querySelectorAll('button[data-id]').forEach(b=>b.classList.toggle('active',b.dataset.id===id));
 const c=row.compliance===null?'invalid':row.compliance.toFixed(3);
 document.querySelector('#info').innerHTML=`Loading original Sparse triangles… · cell [${row.cell}] · mass ${row.mass.toFixed(2)} L · C/ref ${c} · <a href="${row.raw_obj}">raw OBJ</a>`;
 new GLTFLoader().load(row.glb,g=>{
  if(token!==request){g.scene.traverse(x=>{if(x.isMesh)x.geometry.dispose()});return}
  if(object){scene.remove(object);object.traverse(x=>{if(x.isMesh)x.geometry.dispose()})}
  object=g.scene;object.traverse(x=>{if(x.isMesh)x.material=new THREE.MeshStandardMaterial({color:0x8899a6,metalness:.25,roughness:.64,side:THREE.DoubleSide})});scene.add(object);
  const box=new THREE.Box3().setFromObject(object),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3()),radius=Math.max(size.x,size.y,size.z);
  controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(radius*1.5,-radius*2.1,radius*1.1));camera.near=radius/1000;camera.far=radius*100;camera.updateProjectionMatrix();controls.update();
  document.querySelector('#info').innerHTML=`Original Sparse mesh · cell [${row.cell}] · mass ${row.mass.toFixed(2)} L · C/ref ${c} · <a href="${row.raw_obj}">raw OBJ</a>`;
 },undefined,error=>{if(token===request)document.querySelector('#info').textContent=`Sparse mesh loading failed: ${error.message}`});
}
document.querySelectorAll('button[data-id]').forEach(b=>b.onclick=()=>show(b.dataset.id));show(cases[0].id);
renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera)});
</script></html>'''.replace('BUTTONS', buttons).replace('CASES', json.dumps(cases))
    path = OUT / 'sparse_viewer.html'
    path.write_text(page)
    return path


def main() -> None:
    old_rows = {row['id']: row for row in json.loads((PARENT / 'result.json').read_text())['selected']}
    rows = [row for number in (1, 2) for row in
            json.loads((OUT / f'round_{number:02d}/evaluation.json').read_text())['rows']]
    history = {method: set(HISTORY7) for method in ('QD pool', 'control pool')}
    progression = {method: [] for method in history}
    records = []
    for row in rows:
        log = (OUT / f"round_{row['round']:02d}" / row['id'] / 'generation.log').read_text()
        matches = re.findall(r'\[sp FEA step (\d+)\].*?grad_norm=([0-9.eE+-]+)', log)
        expected = [15, 20, 25, 30]
        steps = [int(step) for step, _ in matches]
        valid_gradients = steps == expected and all(float(norm) > 0 for _, norm in matches)
        if not valid_gradients or '[sp FEA] FAIL' in log:
            raise RuntimeError(f"Missing Sparse FEA steps for {row['id']}: {matches}")
        row['in_step_fea'] = [{'step': int(step), 'gradient_norm': float(norm)} for step, norm in matches]
        row['in_step_fea_valid'] = valid_gradients
        if row['strict_eligible'] and row['cell'] is not None:
            history[row['method']].add(tuple(row['cell']))
        progression[row['method']].append(len(history[row['method']]))
        prior = old_rows.get(row['id'])
        records.append({'id': row['id'], 'round': row['round'], 'method': row['method'],
                        'guided_mass_liters': row['mass_liters'],
                        'unguided_mass_liters': prior['mass_liters'] if prior else None,
                        'guided_worst_compliance_ratio': row.get('worst_compliance_ratio'),
                        'unguided_worst_compliance_ratio': prior.get('worst_compliance_ratio') if prior else None,
                        'guided_strict_eligible': row['strict_eligible'],
                        'unguided_strict_eligible': prior['strict_eligible'] if prior else None})
    pic = figure(rows)
    viewer = sparse_viewer(rows)
    result = {'protocol': str(OUT / 'protocol.json'), 'rounds': 2, 'candidates': len(rows),
              'sparse_fea_steps': [15, 20, 25, 30],
              'in_step_fea_successes': sum(row['in_step_fea_valid'] for row in rows),
              'strict_eligible': {method: sum(row['strict_eligible'] for row in rows if row['method'] == method)
                                  for method in history},
              'history_aware_archive_progression': progression,
              'image_to_3d_cell_hits': sum(row['image_cell'] == row['cell'] for row in rows),
              'median_image_arm_aperture': median(row['image_metrics']['front_arm_aperture_fraction'] for row in rows),
              'median_mesh_arm_aperture': median(row['descriptor']['front_arm_aperture_fraction'] for row in rows),
              'comparison_to_unguided': records, 'selected': rows,
              'figure': str(pic), 'viewer': str(OUT / 'index.html'), 'sparse_viewer': str(viewer)}
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Sparse FEA chair QD loop', '',
             'Two sequential rounds used the same 30-image pool (15 QD-directed, 15 controls; 21 passed image gates). Every selected image was generated through Sparse with nonzero 800 N seat-load FEA gradients at steps 15, 20, 25 and 30. The finished mesh was evaluated independently under 800 N seat and 200 N back load cases. Round-two QD selection was frozen after round-one measured worst compliance and mass became available.', '',
             f"Sparse FEA gradients succeeded in **{result['in_step_fea_successes']}/{len(rows)}** runs. Independent two-load strict eligibility: QD **{result['strict_eligible']['QD pool']}/4**, control **{result['strict_eligible']['control pool']}/4**. Image-to-mesh descriptor-cell hits: **{result['image_to_3d_cell_hits']}/{len(rows)}**. The historical seven-cell archive progressed as QD {progression['QD pool']} and control {progression['control pool']}; this metric alone determines new historical coverage.", '',
             '| ID | Arm | Image → 3D cell | Strict | Guided mass L | Guided worst C | Unguided worst C |',
             '|---|---|---|---:|---:|---:|---:|']
    for row, record in zip(rows, records):
        guided = row.get('worst_compliance_ratio')
        unguided = record['unguided_worst_compliance_ratio']
        g = f'{guided:.3f}' if guided is not None else '—'
        u = f'{unguided:.3f}' if unguided is not None else '—'
        lines.append(f"| {row['id']} | {row['method']} | {row['image_cell']} → {row['cell']} | {row['strict_eligible']} | {row['mass_liters']:.2f} | {g} | {u} |")
    lines += ['', 'Lower compliance means greater stiffness on the same repaired-voxel 35 mm FEA domain, but mass and geometry validity must be considered together. The matched unguided comparison is unavailable for newly selected qd_09. Two loads are solved separately, not simultaneously. A prior image/unguided-mesh pool was reused, so this is a feedback-loop engineering demonstration rather than an independent prospective QD superiority test.', '',
              'Surface-smoothing caveat: this eight-case loop used `sp_sdf_smooth_sigma=0` (default). The earlier wrinkle-reduction recipe was `sp_sdf_smooth_sigma=3` with volume matching, a pre-marching-cubes SDF filter, not a training loss. Laplacian (`sp_lap_w=0`) and pool-anchor (`sp_pool_anchor_w=0`) losses were also off. A separate same-cache qd_09 FEA+sigma3 ablation is reported at `smoothing_ablation_2026-10-03/REPORT.md`; it must not be merged into this frozen QD archive.', '',
              'Observed outcome: the Sparse FEA hook is operational, but it did not provide a consistent stiffness gain in seven matched cases. Mean worst-compliance ratio was 1.322 for QD and 1.320 for control; both were worse than the same-domain baseline ratio of 1.0. The eight realized meshes look very similar in front and side views, and none reached a new historical descriptor cell. The next experiment should increase or rescale the actual sparse update and verify a measurable geometry/FEA response before claiming QD improvement.', '',
              f'- Interactive original Sparse mesh: `{viewer}`',
              f'- Visual: `{pic}`', f'- Data: `{OUT / "result.json"}`',
              f'- Protocol: `{OUT / "protocol.json"}`', f'- Viewer: `{OUT / "index.html"}`', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    cards = []
    for row in rows:
        rel = f"round_{row['round']:02d}/mesh_cases/{row['id']}"
        comp = row.get('worst_compliance_ratio')
        display = f'{comp:.3f}' if comp is not None else 'invalid'
        cards.append(f'<article><h3>{html.escape(row["id"])} · {html.escape(row["method"])} · R{row["round"]}</h3>'
                     f'<div class="pair"><img src="../images/{row["id"]}.png"><img src="{rel}/sparse_preview.png"></div>'
                     f'<p>image cell {row["image_cell"]} → mesh cell {row["cell"]} · strict {row["strict_eligible"]} · mass {row["mass_liters"]:.2f} L · worst C/baseline {display} · 4/4 Sparse FEA steps</p>'
                     f'<p><a href="{rel}/combined_voxel.obj">Final OBJ</a> · <a href="{rel}/evaluation.json">FEA and geometry data</a> · <a href="round_{row["round"]:02d}/{row["id"]}/generation/mesh.obj">Raw Sparse OBJ</a></p></article>')
    page = ('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Chair Sparse FEA QD loop</title><style>body{font:16px/1.5 system-ui;max-width:1650px;margin:25px auto;padding:0 20px;background:#eef2f5;color:#152d3e}'
            'img{max-width:100%}article{background:white;border-radius:12px;padding:15px;margin:15px 0}.pair{display:flex;gap:12px}.pair img{width:49%;height:360px;object-fit:contain}a{color:#126296}</style>'
            '<h1>Chair QD · Sparse 내부 FEA + 루프의 성능 피드백</h1>'
            '<p>30 image candidates → 2 rounds × 4 meshes. Sparse step 15/20/25/30: 800 N seat FEA; finished meshes: independent seat/back FEA. '
            '<a href="REPORT.md">report</a> · <a href="result.json">full data</a> · <a href="protocol.json">protocol</a></p>'
            '<h2>입력 이미지 → 원본 Sparse 메쉬 정면·측면</h2><p>아래 렌더는 원본 Sparse 삼각형 메쉬를 물리 좌표로 정렬해서 직접 그렸습니다. 격자 변환이나 Boolean 결과가 아닙니다. <a href="sparse_viewer.html">회전형 뷰어</a></p>'
            '<p><strong>주름 억제 설정:</strong> 이 8개에는 예전 σ=3 SDF 평활화가 빠져 있습니다. FEA는 Sparse 내부에서 켜져 있습니다. <a href="smoothing_ablation_2026-10-03/qd09_before_after.png">FEA+평활화 비교 이미지</a> · <a href="smoothing_ablation_2026-10-03/REPORT.md">비교 결과</a></p>'
            '<img src="selected_sparse_raw_eight.png">' + ''.join(cards) + '</html>')
    (OUT / 'index.html').write_text(page)
    print(json.dumps({'in_step_fea': result['in_step_fea_successes'], 'strict': result['strict_eligible'],
                      'archive': progression, 'figure': str(pic), 'viewer': str(OUT / 'index.html')}, indent=2))


if __name__ == '__main__':
    main()
