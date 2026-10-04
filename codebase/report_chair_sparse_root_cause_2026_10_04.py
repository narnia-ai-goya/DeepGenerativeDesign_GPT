"""Evidence-backed diagnosis of why chair Sparse FEA weight did not improve quality."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

from run_chair_round3_fea_weight_sweep_2026_10_04 import OUT, PARENT


def link(path: Path) -> str:
    return os.path.relpath(path, OUT)


def main() -> None:
    smooth = {r['weight_label']: r for r in json.loads((OUT / 'analysis.json').read_text())}
    unsmooth_rows = json.loads((OUT / 'smoothing_transfer_diagnostic/round_01/evaluation.json').read_text())['rows']
    unsmooth = {r['id']: r for r in unsmooth_rows}
    old = smooth['w1']
    baseline = json.loads((PARENT / 'round_01/evaluation.json').read_text())['baseline_fea']
    ref = next(r for r in json.loads(Path(baseline).read_text())['rows'] if r['id'] == 'vanilla_baseline')
    seat_ratio = old['fea_035']['seat']['compliance_proxy'] / ref['fea_035']['seat']['compliance_proxy']
    back_ratio = old['fea_035']['back']['compliance_proxy'] / ref['fea_035']['back']['compliance_proxy']
    assert abs(seat_ratio-1.0919821544650863) < 1e-8
    assert abs(back_ratio-1.589975151080086) < 1e-8
    assert all(r['changed_fem_node_densities_from_w1'] == 0 for r in smooth.values())
    assert unsmooth['off']['worst_compliance_ratio'] == unsmooth['w10000']['worst_compliance_ratio']
    # Measured with cKDTree nearest vertices. Not an exact surface Hausdorff distance.
    geom = {'pre_refiner_p95_mm': .752, 'post_refiner_no_smoothing_p95_mm': .767,
            'post_refiner_sigma3_p95_mm': .774}
    record = {
        'reference_fea': str(Path(baseline)),
        'seat_compliance_ratio': seat_ratio,
        'back_compliance_ratio': back_ratio,
        'dominant_quality_load': '200 N +Y backrest',
        'in_loop_sparse_load': '800 N -Z seat only',
        'sparse_fea_calls': [15, 20, 25, 30],
        'in_loop_w1_last_compliance': old['in_loop_compliance'][-1],
        'in_loop_w10000_last_compliance': smooth['w10000']['in_loop_compliance'][-1],
        'surface_distance_approx': geom,
        'sigma3_off_worst_ratio': smooth['off']['worst_compliance_ratio'],
        'sigma3_w10000_worst_ratio': smooth['w10000']['worst_compliance_ratio'],
        'sigma0_off_worst_ratio': unsmooth['off']['worst_compliance_ratio'],
        'sigma0_w10000_worst_ratio': unsmooth['w10000']['worst_compliance_ratio'],
        'sigma3_changed_64_voxels': smooth['w10000']['changed_voxels_from_w1'],
        'sigma3_changed_fem_node_densities': smooth['w10000']['changed_fem_node_densities_from_w1'],
        'sigma0_changed_64_voxels_off_vs_w10000': 9,
        'sigma0_changed_fem_node_densities_off_vs_w10000': 0,
        'diagnosis': [
            'The active in-loop physics load optimizes the seat, while final worst-case quality is determined by the backrest load.',
            'The Sparse FEA weight changes the pre-refiner surface only sub-millimetrically, even at 10000x.',
            'The final 64^3 binary occupancy differences are missed by nearest-neighbour transfer to all 6997 FEA nodes, making stiffness exactly equal.',
            'Gaussian SDF smoothing changes absolute final stiffness but does not erase the FEA-weight differential; removal alone does not solve the mismatch.',
            'Volume-neutral guidance and a fixed Dense-derived active token support constrain possible changes, but their individual causal effects were not isolated here.'
        ],
        'next_controls': [
            'Run the same Sparse candidate with backrest +Y FEA in-loop, or a measured two-load worst-case objective.',
            'Match the in-loop density/compliance evaluation to final repaired geometry and test FEM cell-averaged or interpolated density transfer.',
            'After objective and representation match, test more frequent/earlier Sparse FEA updates and support expansion, while measuring raw mesh and final quality.'
        ]
    }
    (OUT / 'root_cause.json').write_text(json.dumps(record, indent=2) + '\n')
    md = f'''# Chair Sparse FEA: root-cause diagnosis

## What is established

1. **The optimized load is not the failing load.** Sparse in-loop FEA uses 800 N downward seat load (`LOAD_MODE=-z`) at steps 15/20/25/30. The final independent two-load check shows seat compliance/reference **{seat_ratio:.3f}**, while backrest 200 N +Y is **{back_ratio:.3f}** and determines the reported worst score. Increasing a seat-only weight does not directly optimize the backrest weakness.
2. **The Sparse shape responds weakly.** From FEA OFF to 10,000×, approximate 95th-percentile nearest-vertex surface separation is {geom['pre_refiner_p95_mm']:.3f} mm before refiner, {geom['post_refiner_no_smoothing_p95_mm']:.3f} mm after refiner without smoothing, and {geom['post_refiner_sigma3_p95_mm']:.3f} mm with σ=3 smoothing. These are sample-wise nearest-vertex distances, not exact Hausdorff distances. The refiner and smoothing do not uniquely erase a large FEA effect; the pre-refiner effect is already small. In-loop final seat compliance changes from {old['in_loop_compliance'][-1]/1e6:.1f} to {smooth['w10000']['in_loop_compliance'][-1]/1e6:.1f} million, but this surrogate result does not survive the final pipeline.
3. **Final FEM sampling cannot see the remaining changes.** With σ=3, the final 64³ occupancies differ by only 3–4 cells across weights. All 6 weight settings have **zero changed densities at the 6,997 FEM nodes** after nearest-neighbour transfer, so the 35 mm FEA element densities and both compliance values are exactly identical. With smoothing disabled, OFF and 10,000× differ by 9 cells, but still zero FEM-node densities. This is a verified representation bottleneck, not a failed FEA solve.
4. **Smoothing affects absolute quality, but is not the reason weight had no effect.** Removing σ=3 smoothing changes worst compliance/reference from {smooth['off']['worst_compliance_ratio']:.3f} to {unsmooth['off']['worst_compliance_ratio']:.3f}, and the 10,000× result follows the same values. It helps this particular sample but the weighted/unweighted pair remains indistinguishable to final FEM.

## Additional constraints, not individually isolated

The Sparse run has only four late manual FEM pushes, decreasing decay, volume-neutral sensitivity projection, and zero support halo around the fixed Dense active tokens. The in-loop objective uses soft density and SIMP penalization 3; the final metric uses binary repaired occupancy and penalization 2. These choices can weaken objective transfer. Their separate causal contribution requires controlled ablations.

## Next controlled experiment

First align the target: use backrest +Y in-loop FEA, ideally together with seat -Z under a specified worst-case objective. Then align the density transfer to the final mesh evaluation; report both the soft in-loop density and repaired binary density at FEM cells. Only then tune weight, update frequency, or support halo. More weight on the present seat-only objective is not supported by this sweep.

## Files

- Weight sweep: `{OUT / 'index.html'}`
- Raw weight data: `{OUT / 'analysis.json'}`
- No-smoothing evaluation: `{OUT / 'smoothing_transfer_diagnostic/round_01/evaluation.json'}`
- Sparse hook implementation: `{Path('/home/goya/SDL/3d_qd/codebase/code/generate_with_physics_guidance.py')}`
- Final FEA mapping implementation: `{Path('/home/goya/SDL/3d_qd/codebase/code/fenics_fea_bracket.py')}`
'''
    (OUT / 'ROOT_CAUSE.md').write_text(md)
    cases = [
        ('σ=3 · FEA OFF', smooth['off']['preview'], smooth['off']['worst_compliance_ratio']),
        ('σ=3 · FEA 10000×', smooth['w10000']['preview'], smooth['w10000']['worst_compliance_ratio']),
        ('σ=0 · FEA OFF', unsmooth['off']['preview'], unsmooth['off']['worst_compliance_ratio']),
        ('σ=0 · FEA 10000×', unsmooth['w10000']['preview'], unsmooth['w10000']['worst_compliance_ratio']),
    ]
    cards = ''.join(f'<article><h3>{html.escape(name)}</h3><img src="{link(Path(preview))}">'
                    f'<p>최악 C/기준 {score:.3f}</p></article>' for name,preview,score in cases)
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sparse FEA 원인 분석</title>
<style>body{{font:16px/1.55 system-ui;max-width:1400px;margin:24px auto;padding:0 20px;background:#eef3f6;color:#18364b}}section,article{{background:white;padding:18px;border-radius:12px;margin:16px 0}}h1{{font-size:2rem}}.cards{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}}.cards article{{margin:0}}.cards img{{width:100%;height:350px;object-fit:contain}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.stats div{{background:#deebf1;padding:14px;border-radius:10px}}.stats b{{display:block;font-size:24px}}a{{color:#146590}}@media(max-width:800px){{.cards,.stats{{grid-template-columns:1fr}}}}</style>
<h1>왜 Sparse FEA weight가 강성을 바꾸지 못했나?</h1>
<div class="stats"><div>좌면 C/기준<b>{seat_ratio:.3f}</b>생성 중 FEA 하중</div><div>등받이 C/기준<b>{back_ratio:.3f}</b>최종 worst-case</div><div>변화된 FEM 노드 밀도<b>0 / 6,997</b>가중치 0~10000×</div></div>
<section><h2>확인된 원인</h2><ol><li><b>목표 하중 불일치:</b> Sparse에서는 좌면 800 N(-Z)만 최적화하지만, 최종 평가를 악화시키는 것은 등받이 200 N(+Y)입니다.</li>
<li><b>작은 Sparse 반응:</b> 10000×에서도 refiner 전 표면 차이는 95% 기준 약 0.75 mm입니다. refiner·평활화 이후에도 유사하므로, 변화가 애초에 작습니다.</li>
<li><b>해상도·밀도 전달:</b> 최종 64³ 점유가 몇 칸 달라도 35 mm FEM의 최근접 노드에 하나도 반영되지 않아 최종 강성이 정확히 같습니다.</li></ol>
<p>평활화 제거는 이번 후보의 절대 강성을 약 3.2% 높이지만, FEA 가중치가 주는 차이는 여전히 0입니다. 볼륨 중립 유도, 고정된 Dense 활성 토큰, 후반 4회 업데이트의 개별 영향은 아직 분리하지 않았습니다.</p>
<p><a href="ROOT_CAUSE.md">상세 분석</a> · <a href="root_cause.json">근거 수치</a> · <a href="index.html">전체 weight sweep</a></p></section>
<h2>평활화 × FEA 가중치 대조</h2><div class="cards">{cards}</div></html>'''
    (OUT / 'root_cause.html').write_text(page)
    print(OUT / 'root_cause.html')


if __name__ == '__main__':
    main()
