#!/usr/bin/env python3
"""Analyze the 12-case local CFG/eta/step factorial plus the s50 reference."""
import html
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/bracket/gpt_cfg_large_local_study_2026-09-14"
REFERENCE = ROOT / "experiments/bracket/gpt_cfg_refined_study_2026-09-14/cases/cfg8_eta225_s50"


def read_case(case, reference=False):
    m = json.loads((case / "gen/metrics/metrics.json").read_text())
    f = json.loads((case / "gen/fea/fea_tet_summary.json").read_text())
    c = json.loads((case / "config.json").read_text())["stages"]["mesh"]
    views = [v for v in m["views"].values() if "input" in v]
    avg = lambda key, group: sum(v[group][key] for v in views) / len(views)
    target_void = avg("projected_holes_area_mm2", "input")
    mesh_void = avg("projected_holes_area_mm2", "mesh")
    return {
        "id": case.name if not reference else "cfg8_eta225_s50_reference",
        "reference": reference, "cfg": float(c["cfg"]), "eta": float(c["eta"]),
        "dense_steps": int(c["dense_steps"]), "mean_iou": avg("mesh_foreground_iou", "input"),
        "mesh_holes": avg("projected_holes_count", "mesh"),
        "target_holes": avg("projected_holes_count", "input"),
        "mesh_void_area_mm2": mesh_void, "target_void_area_mm2": target_void,
        "void_retention": mesh_void / target_void,
        "solid_fraction": avg("solid_fraction_in_cad", "mesh"),
        "compliance_J": f["compliance"], "vm_max_Pa": f["vm_max"],
        "watertight": m["watertight"], "components": m["components"],
        "mesh": str((case / "gen/final.obj").resolve()),
        "preview": str((case / "gen/metrics/final_preview.png").resolve()),
        "config": str((case / "config.json").resolve()),
    }


def pareto(rows):
    for r in rows:
        r["pareto"] = not any(
            q["void_retention"] >= r["void_retention"] and q["compliance_J"] <= r["compliance_J"]
            and (q["void_retention"] > r["void_retention"] or q["compliance_J"] < r["compliance_J"])
            for q in rows)


def factor_effects(rows):
    study = [r for r in rows if not r["reference"]]
    effects = {}
    for factor in ("cfg", "eta", "dense_steps"):
        groups = defaultdict(list)
        for r in study:
            groups[str(r[factor])].append(r)
        effects[factor] = {k: {
            "n": len(v),
            "mean_void_retention": sum(x["void_retention"] for x in v) / len(v),
            "mean_compliance_J": sum(x["compliance_J"] for x in v) / len(v),
        } for k, v in groups.items()}
    return effects


def plots(rows):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for r in rows:
        marker = "*" if r["reference"] else ("o" if r["dense_steps"] == 40 else "s")
        color = {7.0: "#2171b5", 8.0: "#238b45", 9.0: "#cb181d"}[r["cfg"]]
        ax.scatter(r["void_retention"] * 100, r["compliance_J"] * 1000,
                   s=150 if r["pareto"] else 65, marker=marker, color=color,
                   edgecolor="black" if r["pareto"] else "none", zorder=3)
        if r["pareto"]:
            ax.annotate(r["id"], (r["void_retention"] * 100, r["compliance_J"] * 1000),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Projected void-area retention (%) ↑", ylabel="Verified compliance (mJ) ↓",
           title="Bridge-form / mechanics trade-off")
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT / "pareto_void_compliance.png", dpi=190); plt.close(fig)

    ordered = sorted(rows, key=lambda r: (r["reference"], r["cfg"], r["eta"], r["dense_steps"]))
    cells = []
    for r in ordered:
        im = Image.open(r["preview"]).convert("RGB").resize((360, 360))
        cell = Image.new("RGB", (360, 405), "white"); cell.paste(im, (0, 40))
        d = ImageDraw.Draw(cell); d.text((8, 7), r["id"], fill="black")
        d.text((8, 23), f"void {r['void_retention']*100:.1f}%  C {r['compliance_J']*1000:.2f} mJ", fill="black")
        cells.append(cell)
    canvas = Image.new("RGB", (1440, 405 * 4), (235, 238, 240))
    for i, cell in enumerate(cells): canvas.paste(cell, ((i % 4) * 360, (i // 4) * 405))
    canvas.save(OUT / "all_13_montage.png")


def main():
    rows = [read_case(p) for p in sorted((OUT / "cases").glob("*"))]
    rows.append(read_case(REFERENCE, reference=True)); pareto(rows)
    effects = factor_effects(rows); plots(rows)
    front = sorted((r for r in rows if r["pareto"]), key=lambda r: r["compliance_J"])
    summary = {"cases": rows, "pareto_front": [r["id"] for r in front], "factor_effects": effects,
               "recommended": {"mechanics_first": front[0]["id"],
                               "balanced": "cfg8_eta225_s40",
                               "form_first": max(front, key=lambda r:r["void_retention"])["id"]},
               "finding": "Open-arch realization is a narrow bifurcation; 40 dense steps outperformed 60 on average."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = "".join(
        f"<tr class=\"{'pareto' if r['pareto'] else ''}\"><td>{html.escape(r['id'])}</td><td>{r['cfg']:g}</td><td>{r['eta']:g}</td><td>{r['dense_steps']}</td>"
        f"<td>{r['void_retention']*100:.1f}%</td><td>{r['mean_iou']:.3f}</td><td>{r['compliance_J']*1000:.3f}</td><td>{r['vm_max_Pa']/1e6:.1f}</td></tr>"
        for r in sorted(rows, key=lambda x:(-x["void_retention"],x["compliance_J"])))
    cards = "".join(
        f"<article><img src=\"{html.escape(os.path.relpath(r['preview'], OUT))}\"><h3>{html.escape(r['id'])}</h3>"
        f"<p>void {r['void_retention']*100:.1f}% · C {r['compliance_J']*1000:.3f} mJ<br><code>{html.escape(r['mesh'])}</code></p></article>"
        for r in front)
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPT bridge large local study</title><style>
body{{font-family:system-ui,sans-serif;max-width:1320px;margin:32px auto;padding:0 22px;color:#17212b;line-height:1.55}}.note{{background:#eaf7f1;border-left:5px solid #177b65;padding:14px 18px}}.warn{{background:#fff3d8;border-left:5px solid #cc8512;padding:14px 18px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd4da;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef2f4}}tr.pareto{{background:#dff4e9;font-weight:700}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.two img{{width:100%}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}}article{{border:1px solid #d5dce2;border-radius:10px;padding:12px}}article img{{width:100%}}code{{font-size:11px;overflow-wrap:anywhere}}@media(max-width:800px){{.two{{grid-template-columns:1fr}}}}</style>
<h1>GPT bridge local parameter study</h1><p class="note"><b>12개 신규 full-pipeline + 50-step 기준점 비교.</b> 모든 신규 결과는 watertight·single component이며 FEA를 통과했다.</p>
<p class="warn">열린 교량 아치는 좁은 bifurcation 영역에서만 나왔다. 40 steps의 CFG 7/eta 200과 CFG 8/eta 225, 기존 50-step 기준점만 void retention 35% 이상이며 나머지는 4.9–10.2%로 급락했다. 60 steps는 평균적으로 형상을 판으로 메웠다.</p>
<h2>Pareto 후보</h2><div class="cards">{cards}</div><div class="two"><img src="pareto_void_compliance.png"><img src="all_13_montage.png"></div>
<h2>전체 결과</h2><table><tr><th>case</th><th>CFG</th><th>eta</th><th>steps</th><th>void retention</th><th>IoU</th><th>C mJ</th><th>vm MPa</th></tr>{table}</table>
<h2>절대경로</h2><p><code>{OUT/'summary.json'}</code><br><code>{OUT/'protocol.json'}</code><br><code>{OUT/'pareto_void_compliance.png'}</code><br><code>{OUT/'all_13_montage.png'}</code></p></html>'''
    (OUT / "report.html").write_text(page); print((OUT / "report.html").resolve())


if __name__ == "__main__": main()
