"""Calibrate final-mesh descriptors for RA-QD using completed development cases."""
from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import trimesh
from pysdf import SDF

from raqd_core import cell_index

KOREAN_FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(KOREAN_FONT)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=KOREAN_FONT).get_name()
plt.rcParams['axes.unicode_minus'] = False

CANDIDATES = [
    'projection_openness', 'macro_void_fraction_3mm', 'void_clearance_mean_mm',
    'material_centroid_x', 'material_centroid_y', 'material_centroid_z',
    'material_spread_x', 'material_spread_y', 'material_spread_z',
    'material_anisotropy', 'spatial_entropy', 'surface_compactness',
]


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def projection_fill(xyz, occupied, pitch):
    origin = xyz.min(axis=0)
    ijk = np.rint((xyz - origin) / pitch).astype(np.int32)
    values = []
    for axis in range(3):
        keep = [i for i in range(3) if i != axis]
        all_rays = np.unique(ijk[:, keep], axis=0)
        occupied_rays = np.unique(ijk[occupied][:, keep], axis=0)
        values.append(len(occupied_rays) / len(all_rays))
    return values


def features_for_mesh(mesh_path, xyz, pitch):
    mesh = trimesh.load(mesh_path, force='mesh')
    field = SDF(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    signed = field(np.ascontiguousarray(xyz, dtype=np.float32), n_threads=4)
    occupied = signed > 0
    if not occupied.any():
        raise ValueError(f'No occupied reference samples: {mesh_path}')
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    unit = (xyz - lo) / (hi - lo)
    material = unit[occupied]
    covariance = np.cov(material, rowvar=False)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)
    probabilities = []
    octants = (material >= .5).astype(np.int8)
    labels = octants[:, 0] * 4 + octants[:, 1] * 2 + octants[:, 2]
    counts = np.bincount(labels, minlength=8)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum() / np.log(8))
    fills = projection_fill(xyz, occupied, pitch)
    volume = abs(float(mesh.volume))
    sphere_area = (36 * math.pi * volume * volume) ** (1 / 3)
    void = np.maximum(-signed[~occupied], 0)
    values = {
        'material_volume_fraction': float(occupied.mean()),
        'projection_openness': float(1 - np.mean(fills)),
        'projection_fill_x': float(fills[0]),
        'projection_fill_y': float(fills[1]),
        'projection_fill_z': float(fills[2]),
        'macro_void_fraction_3mm': float((signed <= -.003).mean()),
        'void_clearance_mean_mm': float(void.mean() * 1000) if len(void) else 0.,
        'material_anisotropy': float((eigenvalues[-1] - eigenvalues[0]) /
                                     max(eigenvalues[-1], 1e-12)),
        'spatial_entropy': entropy,
        'surface_compactness': float(mesh.area / sphere_area),
    }
    for i, axis in enumerate('xyz'):
        values[f'material_centroid_{axis}'] = float(material[:, i].mean())
        values[f'material_spread_{axis}'] = float(material[:, i].std())
    return values


def padded_bounds(values, padding=.08):
    lo, hi = float(np.min(values)), float(np.max(values))
    width = hi - lo
    if width <= 1e-12:
        width = max(abs(lo), 1.) * .01
    return [lo - padding * width, hi + padding * width]


def ranks(values):
    order = np.argsort(values, kind='mergesort')
    result = np.empty(len(values), dtype=float)
    result[order] = np.arange(len(values))
    return result


def abs_rank_corr(a, b):
    ra, rb = ranks(np.asarray(a)), ranks(np.asarray(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 1.
    return abs(float(np.corrcoef(ra, rb)[0, 1]))


def analyze(rows, dims=(4, 4)):
    volume = np.array([r['material_volume_fraction'] for r in rows])
    bounds = {name: padded_bounds([r[name] for r in rows]) for name in CANDIDATES}
    pairs = []
    for first, second in itertools.combinations(CANDIDATES, 2):
        cells = [cell_index([r[first], r[second]], dims,
                            [bounds[first], bounds[second]]) for r in rows]
        occupied = len(set(cells))
        descriptor_corr = abs_rank_corr([r[first] for r in rows], [r[second] for r in rows])
        volume_corr = max(abs_rank_corr([r[first] for r in rows], volume),
                          abs_rank_corr([r[second] for r in rows], volume))
        score = occupied - .75 * descriptor_corr - .50 * volume_corr
        pairs.append({'descriptors': [first, second], 'occupied_cells': occupied,
                      'coverage': occupied / math.prod(dims),
                      'descriptor_abs_rank_correlation': descriptor_corr,
                      'max_abs_rank_correlation_with_volume': volume_corr,
                      'selection_score': score,
                      'ranges': [bounds[first], bounds[second]],
                      'cells': [list(x) for x in cells]})
    pairs.sort(key=lambda p: (p['selection_score'], p['occupied_cells']), reverse=True)
    return bounds, pairs


def render_plot(rows, selected, out):
    first, second = selected['descriptors']
    fig, ax = plt.subplots(figsize=(8.2, 6.4), layout='constrained')
    style_colors = {'builtin_kagome_trial': '#4169a1', 'bionic_bone': '#16836b',
                    'de_trilattice': '#c66528'}
    for style in style_colors:
        subset = [r for r in rows if r['style'] == style]
        if subset:
            ax.scatter([r[first] for r in subset], [r[second] for r in subset], s=65,
                       alpha=.85, label=style, color=style_colors[style])
    xr, yr = selected['ranges']
    for i in range(5):
        ax.axvline(xr[0] + i * (xr[1]-xr[0])/4, color='#999', lw=.8, alpha=.45)
        ax.axhline(yr[0] + i * (yr[1]-yr[0])/4, color='#999', lw=.8, alpha=.45)
    ax.set(xlim=xr, ylim=yr, xlabel=first, ylabel=second,
           title=f'RA-QD descriptor 개발 분석: {selected["occupied_cells"]}/16 셀 점유')
    ax.grid(alpha=.15); ax.legend(fontsize=8)
    fig.savefig(out / 'selected_descriptor_space.png', dpi=180)
    fig.savefig(out / 'selected_descriptor_space.svg')
    plt.close(fig)


def render_html(rows, analysis, out, source):
    selected = analysis['selected_pair']
    pair_rows = ''.join('<tr><td>{}</td><td>{}</td><td>{}/16</td><td>{:.3f}</td><td>{:.3f}</td></tr>'.format(
        html.escape(p['descriptors'][0]), html.escape(p['descriptors'][1]),
        p['occupied_cells'], p['descriptor_abs_rank_correlation'],
        p['max_abs_rank_correlation_with_volume']) for p in analysis['ranked_pairs'][:12])
    sample_rows = ''.join('<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.4f}</td><td>{:.4f}</td><td>{}</td></tr>'.format(
        html.escape(r['id']), html.escape(r['style']), r['source_method'],
        r[selected['descriptors'][0]], r[selected['descriptors'][1]],
        selected['cells'][i]) for i, r in enumerate(rows))
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RA-QD descriptor 개발 분석</title><style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 22px;line-height:1.6;color:#18212b}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #ccd3da;padding:8px 10px;text-align:left}}th{{background:#f2f5f6}}code{{background:#edf1f3;padding:2px 5px;border-radius:4px}}img{{max-width:100%;height:auto}}.note{{background:#fff7df;border-left:4px solid #d29a16;padding:12px 16px}}</style>
<h1>RA-QD descriptor 개발 분석</h1><p class="note">이 분석은 기존 15개 결과를 개발 자료로 사용한다. 선택된 경계는 다음 독립 실험 전에 고정하며, 기존 MAP-Elites 결과를 사후 재평가해 우수성을 주장하는 용도로 사용하지 않는다.</p>
<p>자동 진단이 선택한 축은 <b>{html.escape(selected['descriptors'][0])}</b> × <b>{html.escape(selected['descriptors'][1])}</b>이다. 기존 15개가 4×4 공간의 <b>{selected['occupied_cells']}개 셀</b>에 분포한다. 이는 descriptor 분리 가능성만 보여주며 RA-QD의 성능 검증은 아니다.</p>
<img src="selected_descriptor_space.png" alt="선택된 descriptor 공간">
<h2>상위 descriptor 조합</h2><table><tr><th>첫 축</th><th>둘째 축</th><th>점유</th><th>축 간 순위상관</th><th>체적과 최대 순위상관</th></tr>{pair_rows}</table>
<h2>후보별 실현 셀</h2><table><tr><th>ID</th><th>스타일</th><th>원래 방법</th><th>첫 축</th><th>둘째 축</th><th>실현 셀</th></tr>{sample_rows}</table>
<h2>판정</h2><ul><li>체적비는 archive 축이 아니라 다음 실험의 고정 제약으로 사용한다.</li><li>자동 순위는 셀 점유, 축 간 중복, 체적과의 중복을 진단한다. 최종 축은 물리적 해석 가능성과 mesh 안정성 검사 후 고정한다.</li><li>목표 셀은 기존 결과에 소급 부여하지 않는다. 기존 결과에는 realized cell만 계산하며 realization gap은 다음 target-directed 실행부터 측정한다.</li></ul>
<p>입력: {html.escape(str(source))}</p><p>절대경로: {html.escape(str(out / 'analysis.json'))}</p></html>'''
    (out / 'report.html').write_text(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    pilot, out = args.pilot.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((pilot / 'summary.json').read_text())
    with np.load(pilot / 'descriptor_reference.npz') as reference:
        xyz = np.ascontiguousarray(reference['xyz'], dtype=np.float32)
        pitch = float(reference['pitch_m'])
    rows = []
    for result in summary['results']:
        if not result['valid']:
            continue
        values = features_for_mesh(result['case_dir'] + '/gen/final.obj', xyz, pitch)
        rows.append({'id': result['id'], 'style': result['genome']['style'],
                     'source_method': result['method'], 'compliance_J': result['compliance_J'],
                     'volume_mm3': result['volume_mm3'], **values})
        print(result['id'], flush=True)
    bounds, pairs = analyze(rows)
    selected = pairs[0]
    analysis = {'status': 'descriptor_development_only', 'source': str(pilot),
                'samples': len(rows), 'dims': [4, 4], 'candidate_bounds': bounds,
                'selected_pair': selected, 'ranked_pairs': pairs,
                'volume_policy': 'fixed constraint in the next independent experiment',
                'target_cells_retrospectively_assigned': False}
    save_json(out / 'analysis.json', analysis)
    save_json(out / 'features.json', rows)
    with (out / 'features.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    render_plot(rows, selected, out)
    render_html(rows, analysis, out, pilot / 'summary.json')
    print(out / 'report.html')


if __name__ == '__main__':
    main()

