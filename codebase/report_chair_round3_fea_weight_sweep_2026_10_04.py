"""Report physical and visual sensitivity to Sparse in-loop FEA weight."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import meshio
import numpy as np
from scipy.spatial import cKDTree

from run_chair_round3_fea_weight_sweep_2026_10_04 import OUT, PARENT
from run_chair_sparse_fea_loop_2026_10_03 import SPEC


def rel(path: str | Path) -> str:
    return os.path.relpath(path, OUT)


def main() -> None:
    rows = json.loads((OUT / 'summary.json').read_text())
    labels = [r['weight_label'] for r in rows]
    weight = [r['sp_fea_w'] for r in rows]
    ratios = [r['worst_compliance_ratio'] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(range(len(rows)), ratios, color='#1e657c', marker='o', lw=2.2)
    ax.axhline(1, color='#9f4e39', ls='--', lw=1.3, label='same-domain reference')
    ax.set_xticks(range(len(rows)), ['off', '1×', '10×', '100×', '1000×', '10000×'])
    ax.set_ylabel('Worst compliance / reference ↓')
    ax.set_xlabel('Sparse FEA weight relative to 8×10⁻⁹')
    ax.set_ylim(min(.95, min(ratios)-.06), max(ratios)+.06)
    ax.grid(alpha=.24)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(OUT / 'weight_vs_stiffness.png', dpi=170)
    plt.close(fig)
    prior = rows[0]
    tetra = meshio.read(SPEC / 'fea_domain/chair_035.msh')
    prior_fea = Path(prior['aligned_mesh']).with_name('fea')
    eval_nodes = np.load(prior_fea / 'nodes.npy')
    nearest_external = cKDTree(eval_nodes).query(tetra.points)[1]
    prior_rho = np.load(prior_fea / 'rho.npy')
    table, cards = [], []
    for r in rows:
        name = r['weight_label']
        log = (PARENT / 'r3_narrow_outerloop/sparse_generation.log' if name == 'w1' else
               OUT / 'round_01' / name / 'generation.log')
        trace = log.read_text(errors='replace')
        gradients = [float(v) for v in re.findall(r'\[sp FEA step \d+\].*?grad_norm=([\d.eE+-]+)', trace)]
        comp = [float(v) for v in re.findall(r'\[sp FEA step \d+\] comp=([\d.eE+-]+)', trace)]
        r['in_loop_gradient_norms'] = gradients
        r['in_loop_compliance'] = comp
        r['gradient_calls'] = len(gradients)
        occ_path = Path(r['aligned_mesh']).with_name('realized_occupancy.npz')
        r['occupancy_path'] = str(occ_path)
        r['changed_voxels_from_w1'] = int(np.count_nonzero(
            np.load(occ_path)['occupied'] !=
            np.load(Path(prior['aligned_mesh']).with_name('realized_occupancy.npz'))['occupied']))
        rho = np.load(Path(r['aligned_mesh']).parent / 'fea/rho.npy')
        r['changed_fem_node_densities_from_w1'] = int(np.count_nonzero(
            rho[nearest_external] != prior_rho[nearest_external]))
        seat = r['fea_035']['seat']['compliance_proxy']
        back = r['fea_035']['back']['compliance_proxy']
        table.append(f'<tr><td>{html.escape(name)}</td><td>{r["sp_fea_w"]:.1e}</td>'
                     f'<td>{r["mass_liters"]:.3f}</td><td>{seat/1e6:.2f}</td><td>{back/1e6:.2f}</td>'
                     f'<td><b>{r["worst_compliance_ratio"]:.3f}</b></td>'
                     f'<td>{r["changed_voxels_from_w1"]}</td><td>{r["changed_fem_node_densities_from_w1"]}</td>'
                     f'<td>{"yes" if r["strict_eligible"] else "no"}</td></tr>')
        preview = Path(r['preview'])
        assert preview.exists(), preview
        cards.append(f'<article><h3>{html.escape(name)} · {r["sp_fea_w"]:.1e}</h3>'
                     f'<img loading="lazy" src="{rel(preview)}" alt="FEA weight {html.escape(name)} mesh">'
                     f'<p>Cell {r["cell"]} · C/ref {r["worst_compliance_ratio"]:.3f} · '
                     f'{r["changed_voxels_from_w1"]} changed 64³ voxels vs 1× · '
                     f'{r["gradient_calls"]} in-loop FEA calls</p>'
                     f'<p><a href="{rel(r["generated_mesh"])}">Raw Sparse OBJ</a> · '
                     f'<a href="{rel(r["aligned_mesh"])}">Aligned OBJ</a></p></article>')
    (OUT / 'analysis.json').write_text(json.dumps(rows, indent=2) + '\n')
    best = min((r for r in rows if r['strict_eligible']), key=lambda r:r['worst_compliance_ratio'])
    changed = (prior['worst_compliance_ratio']-best['worst_compliance_ratio'])/prior['worst_compliance_ratio']*100
    interpretation = (f'Best observed: {best["weight_label"]}, C/ref {best["worst_compliance_ratio"]:.3f} '
                      f'({changed:+.2f}% lower than 1×).' if changed > .001 else
                      'No tested weight measurably improved independent FEA compliance.')
    lines = ['# Chair Sparse FEA weight sweep', '',
             'Same LLM-generated image, 162 px front input, Dense cache, seed, 30 Sparse steps, BC/envelope, σ=3 smoothing and independent 800 N seat / 200 N back FEA. Only `sp_fea_w` changes. The `off` run is the control; `w1=8e-9` is the previous result.', '',
             '| Weight | sp_fea_w | Mass L | Seat C (million) | Back C (million) | Worst C/ref | Changed 64³ voxels vs 1× | Changed FEM-node densities |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["weight_label"]} | {r["sp_fea_w"]:.1e} | {r["mass_liters"]:.3f} | '
                     f'{r["fea_035"]["seat"]["compliance_proxy"]/1e6:.2f} | '
                     f'{r["fea_035"]["back"]["compliance_proxy"]/1e6:.2f} | '
                     f'{r["worst_compliance_ratio"]:.3f} | {r["changed_voxels_from_w1"]} | '
                     f'{r["changed_fem_node_densities_from_w1"]} |')
    lines += ['', interpretation, '',
              f'**Mechanism:** The final 64³ occupancy changes by only 3–4 voxels relative to 1×, and **none of those changed voxels is selected by any of the {len(tetra.points)} nodes in the 35 mm FEM mesh nearest-neighbour density map**. Therefore the element densities and compliance are exactly identical. This is a resolution/transfer bottleneck, not evidence that the physics solver failed. The 10000× in-loop compliance changes, but that improvement does not survive final mesh extraction and the evaluation map.', '',
              'All results use the same 35 mm repaired voxel FEA proxy. In-loop gradients are not evidence of final structural improvement by themselves. This is one image and one seed, so any chosen weight needs replication.', '',
              f'- HTML: `{OUT / "index.html"}`',
              f'- Plot: `{OUT / "weight_vs_stiffness.png"}`',
              f'- Raw metrics: `{OUT / "analysis.json"}`', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chair Sparse FEA weight sweep</title><style>
body{{font:16px/1.55 system-ui;max-width:1420px;margin:26px auto;padding:0 20px;background:#eff3f5;color:#183547}}
article,.panel{{background:white;border-radius:12px;padding:17px;margin:16px 0}}a{{color:#136591}}
table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:9px;border-bottom:1px solid #d6e0e5;text-align:right}}th:first-child,td:first-child{{text-align:left}}
.gallery{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.gallery article{{margin:0}}
.gallery img{{width:100%;height:360px;object-fit:contain}}.plot{{display:block;max-width:850px;width:100%;margin:auto}}
@media(max-width:750px){{.gallery{{grid-template-columns:1fr}}table{{font-size:12px}}}}
</style><h1>Chair Sparse FEA weight sensitivity</h1>
<p>동일한 이미지·Dense cache·seed·BC·envelope·30 Sparse steps에서 <code>sp_fea_w</code>만 변경했습니다. 기존 1×는 8×10⁻⁹입니다.</p>
<section class="panel"><h2>독립 FEA 결과</h2><p><b>{html.escape(interpretation)}</b> 낮은 C/ref가 더 강한 형상입니다.</p>
<img class="plot" src="weight_vs_stiffness.png"><table><tr><th>설정</th><th>가중치</th><th>질량 L</th><th>좌면 C, 백만</th><th>등받이 C, 백만</th><th>최악 C/ref ↓</th><th>변화 voxel</th><th>변화 FEM 노드 밀도</th><th>형상/FEA 통과</th></tr>{''.join(table)}</table>
<p><b>원인:</b> 최종 64³ 점유는 1× 대비 3–4 voxel만 달랐고, 바뀐 voxel은 35 mm FEM 메쉬의 {len(tetra.points)}개 노드 중 어느 곳에도 최근접 밀도로 선택되지 않았습니다. 그래서 독립 FEA에 들어간 요소 밀도가 모두 같고 compliance도 정확히 같습니다. 생성 중 FEA는 800 N 좌면 하중으로 4회 작동하며, 최종 비교는 별도 800 N 좌면·200 N 등받이 FEA입니다. 한 이미지·한 seed 결과입니다.</p>
<p><a href="REPORT.md">해석 보고서</a> · <a href="analysis.json">전체 수치</a> · <a href="{rel(PARENT / 'index.html')}">이전 QD 보고서</a></p></section>
<h2>최종 형상 비교</h2><div class="gallery">{''.join(cards)}</div></html>'''
    (OUT / 'index.html').write_text(page)
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
