"""Checkpoint analysis for the fair online RA-QD versus Sobol experiment."""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from raqd_core import RealizationArchive


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def metrics(rows, protocol):
    archive = RealizationArchive(protocol['dims'], protocol['descriptor_ranges'])
    enriched = []
    for row in rows:
        observed, _ = archive.record(row); enriched.append(observed)
    qualities = [1/(1+elite['compliance_J']/.01) for elite in archive.elites.values()]
    feasible = [row for row in enriched if row.get('constraints_satisfied')]
    out_of_range = [row for row in enriched if row.get('realized_descriptors') is not None
                    and row.get('realized_cell') is None]
    targeted = [row for row in enriched if row.get('target_cell') is not None]
    return {**archive.summary(), 'qd_score': sum(qualities),
            'best_compliance_J': min((row['compliance_J'] for row in archive.elites.values()), default=None),
            'volume_feasible_rate': len(feasible)/len(rows) if rows else None,
            'descriptor_out_of_range_rate': len(out_of_range)/len(rows) if rows else None,
            'target_hits': sum(row.get('realized_cell') == row.get('target_cell') for row in targeted)}


def drift(rows, protocol):
    widths = np.asarray([hi-lo for lo, hi in protocol['descriptor_ranges']])
    output = {}
    for first, second in [('dense', 'sparse'), ('sparse', 'final'), ('dense', 'final')]:
        values, volume = [], []
        for row in rows:
            path = row.get('realization_path', {})
            if first not in path or second not in path: continue
            a = np.asarray([path[first][n] for n in protocol['descriptor_names']])
            b = np.asarray([path[second][n] for n in protocol['descriptor_names']])
            values.append(float(np.linalg.norm((b-a)/widths)))
            volume.append(abs(path[second]['material_volume_fraction']-
                              path[first]['material_volume_fraction']))
        output[f'{first}_to_{second}'] = {
            'samples': len(values), 'mean_normalized_descriptor_drift': float(np.mean(values)) if values else None,
            'mean_absolute_volume_fraction_drift': float(np.mean(volume)) if volume else None}
    return output


def analyze(summary, protocol):
    results = summary['results']
    shared = [row for row in results if row['method'] == 'shared']
    branches = {name: [row for row in results if row['method'] == name] for name in ('raqd', 'random')}
    checkpoints = {}
    for budget in protocol['checkpoints_per_method']:
        own = max(0, budget-len(shared))
        checkpoints[str(budget)] = {name: metrics([*shared, *rows[:own]], protocol)
                                    for name, rows in branches.items()}
    paired = []
    by_index = {name: {int(row['id'].split('_')[-1]): row for row in rows}
                for name, rows in branches.items()}
    for index in sorted(set(by_index['raqd']) & set(by_index['random'])):
        left, right = by_index['raqd'][index], by_index['random'][index]
        paired.append({'index': index, 'seed': left['seed'],
                       'raqd_feasible': left.get('constraints_satisfied', False),
                       'random_feasible': right.get('constraints_satisfied', False),
                       'raqd_compliance_J': left.get('compliance_J'),
                       'random_compliance_J': right.get('compliance_J')})
    return {'status': summary['phase'], 'completed_unique': len(results),
            'checkpoints': checkpoints, 'realization_drift': {
                name: drift([*shared, *rows], protocol) for name, rows in branches.items()},
            'paired_post_initial': paired}


def plot(analysis, out):
    budgets = sorted(int(x) for x in analysis['checkpoints'])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout='constrained')
    for name, color in [('raqd', '#167b69'), ('random', '#b86b27')]:
        values = [analysis['checkpoints'][str(b)][name] for b in budgets]
        axes[0].plot(budgets, [100*v['verified_coverage'] for v in values], 'o-', label=name, color=color)
        axes[1].plot(budgets, [v['qd_score'] for v in values], 'o-', label=name, color=color)
        axes[2].plot(budgets, [100*v['volume_feasible_rate'] if v['volume_feasible_rate'] is not None
                              else np.nan for v in values], 'o-', label=name, color=color)
    axes[0].set(ylabel='Verified coverage (%)', xlabel='Logical evaluations')
    axes[1].set(ylabel='Verified QD score', xlabel='Logical evaluations')
    axes[2].set(ylabel='Volume-feasible rate (%)', xlabel='Logical evaluations')
    for ax in axes: ax.grid(alpha=.2); ax.legend()
    fig.savefig(out/'checkpoint_comparison.png', dpi=180); plt.close(fig)


def report(analysis, out):
    rows = ''
    for budget, methods in analysis['checkpoints'].items():
        for name, value in methods.items():
            feasible = (f'{100*value["volume_feasible_rate"]:.1f}%'
                        if value['volume_feasible_rate'] is not None else '–')
            rows += (f'<tr><td>{budget}</td><td>{name}</td><td>{value["verified_elites"]}/16</td>'
                     f'<td>{value["qd_score"]:.4f}</td><td>{feasible}</td>'
                     f'<td>{value["target_hit_rate"] if value["target_hit_rate"] is not None else "–"}</td></tr>')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RA-QD online checkpoint 분석</title><style>body{{font-family:system-ui,sans-serif;max-width:1050px;margin:36px auto;padding:0 22px;color:#17212b;line-height:1.55}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd4da;padding:8px}}th{{background:#eef2f4}}img{{width:100%}}code{{overflow-wrap:anywhere}}</style><h1>RA-QD online checkpoint 분석</h1><p>상태: <b>{html.escape(analysis['status'])}</b> · 완료된 고유 평가: {analysis['completed_unique']}/52</p><img src="checkpoint_comparison.png"><table><tr><th>예산</th><th>방법</th><th>검증 셀</th><th>QD score</th><th>체적 통과</th><th>목표 적중률</th></tr>{rows}</table><p>분석 절대경로: <code>{html.escape(str(out/'analysis.json'))}</code></p></html>'''
    (out/'analysis.html').write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args(); out = args.experiment.resolve()
    summary = json.loads((out/'summary.json').read_text())
    protocol = json.loads((out/'protocol.json').read_text())
    result = analyze(summary, protocol); save(out/'analysis.json', result)
    plot(result, out); report(result, out); print(out/'analysis.html')


if __name__ == '__main__': main()
