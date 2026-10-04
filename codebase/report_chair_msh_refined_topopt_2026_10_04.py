"""Final comparison for the 15 mm chair-envelope TO and connection repair."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
STUDY = (BASE / 'text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03'
         / 'llm_text_proposal_round_03_2026-10-03'
         / 'simultaneous_load_dense_sparse_2026-10-04')
OUT = STUDY / 'diagnostics/msh_simp_topopt_015_2026-10-04'
SPEC = (BASE / 'text_reasoned_front_axes_2026-10-03/long_qd_fea_2026-10-03'
        / 'envelope_plus16_spec_2026-10-03')
CHAIR = STUDY / 'sparse_evaluation/round_01/mesh_cases/sparse_d0_s1/aligned_main.obj'


def render(mesh, envelope, center, azim, elevation, color):
    size = 540
    pl = pv.Plotter(off_screen=True, window_size=(size, size))
    pl.set_background('#fbfcfe')
    pl.enable_anti_aliasing('msaa')
    pl.add_mesh(mesh, color=color, smooth_shading=False, ambient=.32,
                diffuse=.65, specular=.1, specular_power=16)
    a = np.deg2rad(azim)
    e = np.deg2rad(elevation)
    eye = center + 2.0*np.array([np.cos(e)*np.sin(a),
                                 -np.cos(e)*np.cos(a), np.sin(e)])
    pl.camera_position = [eye.tolist(), center.tolist(), [0, 0, 1]]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = .59
    img = Image.fromarray(pl.screenshot(return_img=True)).convert('RGB')
    pl.close()
    return img


def main() -> None:
    cont = json.loads((OUT / 'summary.json').read_text())
    base = json.loads((OUT / 'baseline15/dc_info.json').read_text())
    raw = json.loads((OUT / 'raw_binary/raw_binary_audit.json').read_text())
    raw_fea = json.loads((OUT / 'raw_binary/fea/dc_info.json').read_text())
    repaired = json.loads((OUT / 'connected_final/connected_binary_audit.json').read_text())
    repaired_fea = json.loads((OUT / 'connected_final/fea/dc_info.json').read_text())
    envelope = pv.read(SPEC / 'envelope.stl')
    center = np.array(envelope.center)
    chair = pv.read(CHAIR)
    raw_mesh = pv.read(OUT / 'raw_binary/raw_binary.vtu').threshold(.5, scalars='density').extract_surface()
    repaired_mesh = pv.read(OUT / 'connected_final/connected_binary.vtu').threshold(.5, scalars='density').extract_surface()
    cases = [('Generated chair', chair, 38, 22, '#89939e'),
             ('TO raw · matched volume', raw_mesh, 38, 22, '#b8754b'),
             ('TO connected · 3/4', repaired_mesh, 38, 22, '#6c8b78'),
             ('TO connected · right', repaired_mesh, 90, 10, '#6c8b78')]
    tile, header = 540, 106
    sheet = Image.new('RGB', (tile*4, tile+header), '#fbfcfe')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    draw.text((23, 13), 'Chair topology optimization · 15 mm tetra mesh',
              font=font, fill='#203448')
    draw.text((23, 54), 'Same 25.44 L · 800 N seat + 200 N back · four-foot connection audited',
              font=small, fill='#466273')
    for i, (label, mesh, azim, elev, color) in enumerate(cases):
        sheet.paste(render(mesh, envelope, center, azim, elev, color), (i*tile, header))
        draw.text((i*tile+17, header+14), label, font=small, fill='#243c4d')
    sheet.save(OUT / 'refined_comparison.png')
    history = json.loads((OUT / 'history.json').read_text())
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot([x['iteration'] for x in history],
            [x['compliance']/1e6 for x in history], 'o-', lw=2, ms=4, color='#ac6d47')
    ax.set(xlabel='OC iteration', ylabel='Continuous-density compliance (million)',
           title='15 mm FEA, fixed 25.442 L density volume')
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / 'refined_convergence.png', dpi=170)
    plt.close(fig)
    comparison = {'generated_chair_compliance': base['compliance'],
                  'continuous_topopt_compliance': cont['final_compliance'],
                  'raw_binary_compliance': raw_fea['compliance'],
                  'connected_binary_compliance': repaired_fea['compliance'],
                  'raw_binary': raw, 'connected_binary': repaired,
                  'all_same_fem_mesh': str(SPEC / 'fea_domain/chair_015.msh'),
                  'limitations': ['Ersatz-material FEM uses a low but nonzero modulus in void cells.',
                                  'Tetra threshold surfaces are rough and not CAD-ready.',
                                  'Connection repair is a separate post-optimization operation.']}
    (OUT / 'refined_comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>15 mm chair topology optimization</title><style>body{{font:16px/1.55 system-ui;max-width:1500px;margin:28px auto;padding:0 22px;background:#eef3f5;color:#20394b}}article{{background:white;padding:20px;border-radius:12px;margin:18px 0}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse}}td,th{{border:1px solid #cdd9df;padding:8px 12px;text-align:right}}td:first-child,th:first-child{{text-align:left}}code{{overflow-wrap:anywhere}}</style>
<h1>15 mm 의자 envelope 직접 위상최적화</h1><article><p>동일한 envelope에서 388,856개 사면체 요소의 밀도를 25회 SIMP/OC로 최적화했습니다. 목표 밀도 체적 25.442 L, 좌면 800 N −Z와 등받이 200 N +Y를 한 번에 적용, SIMP p=2, 감도 필터 45 mm. 모든 값은 동일한 15 mm FEM 메시에서 재평가했습니다.</p>
<table><tr><th>형상</th><th>체적 L</th><th>동시 하중 compliance</th><th>형상 연결성</th></tr>
<tr><td>생성 의자</td><td>25.442</td><td>{base['compliance']/1e6:.3f}×10⁶</td><td>기존 평가 형상</td></tr>
<tr><td>TO 연속 밀도</td><td>{cont['final_volume_liters']:.3f}</td><td>{cont['final_compliance']/1e6:.3f}×10⁶</td><td>이진 형상이 아님</td></tr>
<tr><td>TO 단순 이진화</td><td>{raw['binary_volume_liters']:.3f}</td><td>{raw_fea['compliance']/1e6:.3f}×10⁶</td><td>{raw['components']}개 성분, 네 발 연결={raw['all_bc_connected']}</td></tr>
<tr><td>TO 연결 보정+이진화</td><td>{repaired['binary_volume_liters']:.3f}</td><td>{repaired_fea['compliance']/1e6:.3f}×10⁶</td><td>{repaired['components']}개 성분, 네 발 연결={repaired['all_bc_connected']}</td></tr></table>
<p>수치는 동일 메시의 ersatz-material FEA 결과입니다. 빈 요소에도 작은 탄성계수가 남아 있으므로 연결성 평가는 별도 그래프 검사로 수행했습니다. 연결 보정은 후처리이며 원래 OC 해와 구별합니다.</p></article>
<article><img src="refined_comparison.png"><p>왼쪽 생성 의자는 기준 형상입니다. 갈색 단순 이진화는 앞발 두 개가 떨어져 있고, 초록색 연결 보정 결과는 네 발이 붙지만 표면이 거칠고 원래 의자 형상과 다릅니다.</p></article><article><h2>수렴과 메쉬 해상도</h2><img src="refined_convergence.png" style="max-width:850px"><p><a href="../msh_resolution_comparison_2026-10-04.png">35/20/15 mm 메쉬 비교</a> · <a href="refined_comparison.json">전체 수치</a> · <a href="optimized_density.vtu">연속 밀도 VTU</a> · <a href="connected_final/connected_binary.vtu">연결된 이진 VTU</a></p></article></html>'''
    (OUT / 'index.html').write_text(page)
    print(OUT / 'index.html')
    print(json.dumps({k: comparison[k] for k in ('generated_chair_compliance',
          'continuous_topopt_compliance', 'raw_binary_compliance',
          'connected_binary_compliance')}, indent=2))


if __name__ == '__main__':
    main()
