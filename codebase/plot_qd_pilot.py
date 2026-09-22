"""Export pilot behavior-space and evaluation-budget plots with fixed archive axes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

KOREAN_FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(KOREAN_FONT)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=KOREAN_FONT).get_name()
plt.rcParams['axes.unicode_minus'] = False


def plot(out):
    state = json.loads((out / 'summary.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    colors = {'shared': '#667085', 'map_elites': '#147d64', 'random': '#c66528'}
    labels = {'shared': '공통 초기 후보', 'map_elites': 'MAP-Elites', 'random': '랜덤 탐색'}
    for method in colors:
        rs = [r for r in state['results'] if r['method'] == method and r['valid']]
        if rs:
            axes[0].scatter([r['descriptors'][0] for r in rs], [r['descriptors'][1] for r in rs],
                            label=labels[method], c=colors[method], s=48, alpha=.75,
                            marker={'shared': 's', 'map_elites': 'o', 'random': '^'}[method])
    with np.load(out / 'descriptor_reference.npz') as r:
        upper_capacity = float(r['upper'].mean())
    volume = np.linspace(.001, 1, 1000)
    lower = np.maximum(0, 1 - (1-upper_capacity)/volume)
    upper = np.minimum(1, upper_capacity/volume)
    axes[0].fill_between(volume, upper, 1, color='#eeeeee', zorder=-2)
    axes[0].fill_between(volume, 0, lower, color='#eeeeee', zorder=-2)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='설계 영역 체적 점유율',
                ylabel='상부 재료 비율', title='고정된 4 × 4 행동 공간')
    axes[0].set_xticks(np.linspace(0, 1, 5)); axes[0].set_yticks(np.linspace(0, 1, 5))
    axes[0].grid(alpha=.35); axes[0].legend(fontsize=8)
    axes[0].text(.02, .96, '회색: CAD 용량상 도달 불가', transform=axes[0].transAxes,
                 fontsize=7, va='top', color='#58606b')
    for method, result in state['methods'].items():
        hist = result['history']; xs = np.arange(1, len(hist)+1)
        style = '-o' if method == 'map_elites' else '--^'
        axes[1].plot(xs, [h['coverage']*100 for h in hist], style, ms=4,
                     color=colors[method], label=labels[method])
        ys = [h['best_compliance_J']*1000 if h['best_compliance_J'] is not None else np.nan for h in hist]
        axes[2].plot(xs, ys, style, ms=4, color=colors[method], label=labels[method])
    axes[1].set(xlabel='방법별 평가 횟수', ylabel='커버리지 (%)', ylim=(0, 100),
                title='Archive 커버리지 (16개 셀)')
    axes[2].set(xlabel='방법별 평가 횟수', ylabel='최저 compliance (mJ)',
                title='최종 FEA 최저값 (낮을수록 좋음)')
    for ax in axes[1:]:
        ax.set_xlim(1, 9); ax.set_xticks(range(1, 10)); ax.grid(alpha=.25)
        ax.axvline(3, color='#888888', ls=':', lw=1); ax.axvline(6, color='#888888', ls=':', lw=1)
        ax.legend(fontsize=8)
    fig.suptitle('브래킷 QD 파일럿: 방법별 9회 평가, 공통 초기 후보 3개', fontsize=13)
    fig.savefig(out / 'comparison.png', dpi=180)
    fig.savefig(out / 'comparison.svg')
    plt.close(fig)
    print(out / 'comparison.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('out', type=Path)
    plot(ap.parse_args().out.resolve())
