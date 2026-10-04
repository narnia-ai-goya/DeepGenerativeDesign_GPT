#!/usr/bin/env python3
"""Build the round-0 designer-steered Semantic BO-QD pilot report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ["random", "auto_bo_qd", "initial_intent", "periodic_steering"]
COLORS = {"random": "#777777", "auto_bo_qd": "#2476a6",
          "initial_intent": "#d38b2f", "periodic_steering": "#15836f"}


def mass_bin(value: float, edges: list[float]) -> int | None:
    if value < edges[0] or value > edges[-1]: return None
    return min(len(edges) - 2, int(np.digitize(value, edges[1:-1])))


def archive(rows: list[dict], edges: list[float]) -> dict:
    elites = {}
    for row in rows:
        if not row.get("hard_gate_pass") or row.get("compliance_J") is None: continue
        cell = (row.get("semantic_niche", row.get("semantic_niche_target")),
                mass_bin(row["normalized_mass"], edges))
        if cell[1] is None: continue
        if cell not in elites or row["compliance_J"] < elites[cell]["compliance_J"]:
            elites[cell] = row
    return elites


def semantic_hv(rows: list[dict], niches: list[str], mass_ref=.5, compliance_ref=.15) -> float:
    total = 0.0
    for niche in niches:
        points = sorted((r["normalized_mass"], r["compliance_J"]) for r in rows
                        if r.get("hard_gate_pass") and r.get("compliance_J") is not None
                        and r.get("semantic_niche", r.get("semantic_niche_target")) == niche
                        and r["normalized_mass"] <= mass_ref and r["compliance_J"] <= compliance_ref)
        current_c = compliance_ref
        for mass, compliance in points:
            if compliance < current_c:
                total += (mass_ref - mass) / .30 * (current_c - compliance) / compliance_ref
                current_c = compliance
    return total / max(1, len(niches))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()
    protocol = json.loads((args.experiment / "protocol_snapshot.json").read_text())
    edges = protocol["archive"]["mass_edges"]; niches = list(protocol["archive"]["semantic_niches"])
    warm = json.loads((args.experiment / "shared_warm_start" / f"seed_{args.seed}" /
                       "warm_start_records.json").read_text())["records"]
    verified = json.loads((args.experiment / "shared_evaluations" / f"round_{args.round:02d}" /
                           "verified_results.json").read_text())
    verified_by_id = {r["id"]: r for r in verified}
    summaries, condition_rows = {}, {}
    for condition in CONDITIONS:
        selection = json.loads((args.experiment / "runs" / condition / f"seed_{args.seed}" /
                                "rounds" / f"round_{args.round:02d}" /
                                "sparse_fea_selection.json").read_text())["selected"]
        adaptive = [verified_by_id[r["id"]] for r in selection]
        rows = warm + adaptive; condition_rows[condition] = adaptive
        elites = archive(rows, edges)
        summaries[condition] = {
            "adaptive_valid": sum(r["hard_gate_pass"] for r in adaptive),
            "adaptive_total": len(adaptive), "occupied_cells": len(elites),
            "coverage": len(elites) / protocol["archive"]["cells"],
            "semantic_hv": semantic_hv(list(elites.values()), niches),
            "best_compliance_mJ": min(1000*r["compliance_J"] for r in elites.values()),
            "selected_ids": [r["id"] for r in adaptive],
        }
    report = args.experiment / "report"; assets = report / "assets"; assets.mkdir(parents=True, exist_ok=True)
    (report / "summary.json").write_text(json.dumps({"conditions": summaries,
        "semantic_hv_reference": {"normalized_mass": .5, "compliance_J": .15},
        "note": "Periodic steering equals initial intent in round 0; interaction starts after this checkpoint."}, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=180)
    for condition in CONDITIONS[:3]:
        rows = condition_rows[condition]
        ax.scatter([r["normalized_mass"] for r in rows], [1000*r["compliance_J"] for r in rows],
                   label=condition, s=70, color=COLORS[condition], alpha=.85)
        for r in rows: ax.annotate(r["id"].replace("__", "\n"),
                                   (r["normalized_mass"], 1000*r["compliance_J"]), fontsize=6)
    ax.set(xlabel="Normalized mass", ylabel="Compliance (mJ) ↓",
           title="Round-0 promoted candidates: mass–performance")
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(report / "mass_compliance.png"); plt.close(fig)

    cards = []
    for row in verified:
        input_dst = assets / f"{row['id']}_input.png"; final_dst = assets / f"{row['id']}_final.png"
        shutil.copy2(row["image_512"], input_dst); shutil.copy2(row["final_preview"], final_dst)
        sem = row["semantic_realization"]
        cards.append(f"<article><div><img src='assets/{input_dst.name}'><img src='assets/{final_dst.name}'></div>"
                     f"<h3>{row['id']}</h3><p>hard gate: <b>{row['hard_gate_pass']}</b> · mass {row['normalized_mass']:.3f} · "
                     f"C {1000*row['compliance_J']:.3f} mJ<br>fix {row['fix_coverage_1p5mm']:.3f} · load {row['load_coverage_1p5mm']:.3f} · "
                     f"t10 {row['thickness_p10_mm']:.2f} mm · semantic gap {sem['input_to_final_gap']:.4f}</p>"
                     f"<code>{row['final_mesh']}</code></article>")
    table = "".join(f"<tr><td>{name}</td><td>{s['adaptive_valid']}/{s['adaptive_total']}</td>"
                    f"<td>{s['occupied_cells']}/24 ({100*s['coverage']:.1f}%)</td>"
                    f"<td>{s['semantic_hv']:.4f}</td><td>{s['best_compliance_mJ']:.3f}</td>"
                    f"<td>{', '.join(s['selected_ids'])}</td></tr>" for name,s in summaries.items())
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Designer-steered Semantic BO-QD pilot</title><style>body{{font:15px system-ui;max-width:1500px;margin:30px auto;padding:0 22px;color:#183044;background:#f3f6f8}}section,article{{background:white;border:1px solid #d3dce3;border-radius:10px;padding:14px}}.note{{border-left:5px solid #16816d}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #ced9e0;padding:8px;text-align:left}}th{{background:#eaf1f5}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}article div{{display:grid;grid-template-columns:1fr 1fr}}img{{width:100%}}code{{font-size:10px;overflow-wrap:anywhere}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style><h1>Designer-steered Semantic BO-QD · pilot round 0</h1><section class="note"><b>실제 실행 결과.</b> 18 image candidates → 조건별 dense 6 → 조건별 sparse/FEA 3. 중복 계산은 공유했으며 모든 조건의 논리 예산은 동일하다. Periodic steering은 이 checkpoint 이후부터 initial-intent 조건과 갈라진다.</section><h2>조건별 결과</h2><table><tr><th>condition</th><th>adaptive valid</th><th>archive coverage</th><th>semantic HV</th><th>best C (mJ)</th><th>promoted candidates</th></tr>{table}</table><img src="mass_compliance.png"><h2>Input image → final 3D</h2><div class="grid">{''.join(cards)}</div><p>Protocol: {args.experiment/'protocol_snapshot.json'}<br>Verified data: {args.experiment/'shared_evaluations'/f'round_{args.round:02d}'/'verified_results.json'}<br>Summary: {report/'summary.json'}</p></html>'''
    (report / "index.html").write_text(page)
    print((report / "index.html").resolve())


if __name__ == "__main__": main()
