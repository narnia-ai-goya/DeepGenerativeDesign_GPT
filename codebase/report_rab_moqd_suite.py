"""Render the RAB-MOQD development suite as figures and an HTML report."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/bracket/rab_moqd_experiment_suite_2026-09-14"


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    methods = ["posterior_sampled", "posterior_mean", "sobol_random"]
    labels = ["Posterior sampled", "Posterior mean", "Sobol random"]
    colors = ["#2563a6", "#1d8a70", "#d17a2b"]
    gains = [summary["comparison"][name]["qd_hypervolume_gain"] for name in methods]
    valid = [summary["comparison"][name]["valid"] for name in methods]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), layout="constrained")
    axes[0].bar(labels, gains, color=colors)
    axes[0].set(ylabel="QD-HV gain over warm start", title="Three-evaluation method check")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(labels, valid, color=colors)
    axes[1].set(ylim=(0, 3.3), ylabel="Valid realizations / 3", title="Pipeline validity")
    axes[1].tick_params(axis="x", rotation=15)
    fig.savefig(OUT / "method_comparison.png", dpi=190); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.7), layout="constrained")
    for method, label, color in zip(methods, labels, colors):
        rows = [row for row in summary["comparison_results"] if row["method"] == method and row["valid"]]
        ax.scatter([row["realized_descriptors"][0] for row in rows],
                   [row["realized_descriptors"][1] for row in rows],
                   s=90, color=color, label=label, alpha=.85)
    for value in [.020, .02525, .0305, .03575, .041]: ax.axvline(value, color="#888", lw=.7, alpha=.45)
    for value in [.58, .63, .68, .73, .78]: ax.axhline(value, color="#888", lw=.7, alpha=.45)
    ax.set(xlabel="Normalized void scale", ylabel="Strain-energy concentration",
           xlim=(.0195, .0415), ylim=(.575, .785), title="Realized behavior cells")
    ax.legend(); ax.grid(alpha=.10)
    fig.savefig(OUT / "realized_behavior_space.png", dpi=190); plt.close(fig)

    rows = []
    for name in methods:
        item = summary["comparison"][name]
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{item['valid']}/3</td>"
                    f"<td>{item['new_cells']}</td><td>{item['qd_hypervolume_gain']:.5f}</td></tr>")
    cards = []
    for row in summary["comparison_results"]:
        image = ""
        if row.get("preview"):
            image = f'<img src="{html.escape(os.path.relpath(row["preview"], OUT))}">'
        values = (f"C={1000*row['compliance_J']:.3f} mJ · V={row['measured_design_volume_fraction']:.3f}"
                  if row.get("valid") else html.escape("; ".join(row["invalid_reasons"])))
        cards.append(f'<article>{image}<h3>{html.escape(row["id"])}</h3><p>{values}</p></article>')
    verification = "".join(
        f"<li><b>{item['source_result']}</b>: cells {item['realized_cells']}; agreement "
        f"{item['modal_cell_agreement']:.3f}</li>" for item in summary["verification"])
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RAB-MOQD development suite</title><style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:34px auto;padding:0 22px;color:#17212b;line-height:1.58}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd4da;padding:9px}}th{{background:#eef2f4}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:14px}}article{{border:1px solid #d5dce2;border-radius:10px;padding:12px}}article img{{width:100%}}.note{{background:#fff5dc;border-left:4px solid #c88815;padding:12px 16px}}.path{{overflow-wrap:anywhere;font-size:12px}}.plots{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.plots img{{width:100%}}</style><h1>RAB-MOQD development suite</h1><p class="note">각 방법 3회인 method-development 결과다. 통계적 우월성 주장이 아니라 다음 알고리즘 변경을 결정하는 진단으로 사용한다.</p><table><tr><th>방법</th><th>유효</th><th>신규 cell</th><th>QD-HV 증가</th></tr>{''.join(rows)}</table><div class="plots"><img src="method_comparison.png"><img src="realized_behavior_space.png"></div><h2>Seed verification</h2><ul>{verification}</ul><p>두 recipe 모두 세 seed가 서로 다른 cell에 실현됐다. 따라서 single realization elite를 즉시 확정하지 않고, recipe-level cell distribution과 risk-aware Pareto 값을 추정해야 한다.</p><h2>Realizations</h2><div class="cards">{''.join(cards)}</div><p class="path">{OUT / 'summary.json'}<br>{OUT / 'protocol.json'}</p></html>'''
    (OUT / "report.html").write_text(page)
    print(OUT / "report.html")


if __name__ == "__main__": main()
