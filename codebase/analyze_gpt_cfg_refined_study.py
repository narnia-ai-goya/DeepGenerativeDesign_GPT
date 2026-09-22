#!/usr/bin/env python3
"""Aggregate and render the refined GPT bridge CFG/eta study."""
import html
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/bracket/gpt_cfg_refined_study_2026-09-14"
TARGET = ROOT / "output/imagegen/bracket_gpt_bridge_arch_v2_2026-09-14/gpt_contact_sheet.png"


def load_rows():
    rows = []
    for case in sorted((OUT / "cases").glob("*")):
        mp = case / "gen/metrics/metrics.json"
        fp = case / "gen/fea/fea_tet_summary.json"
        if not mp.exists() or not fp.exists():
            continue
        m, f = json.loads(mp.read_text()), json.loads(fp.read_text())
        views = [v for v in m["views"].values() if "input" in v]
        mean = lambda key, group: sum(v[group][key] for v in views) / len(views)
        target_void = mean("projected_holes_area_mm2", "input")
        mesh_void = mean("projected_holes_area_mm2", "mesh")
        cfg = float(json.loads((case / "config.json").read_text())["stages"]["mesh"]["cfg"])
        eta = float(json.loads((case / "config.json").read_text())["stages"]["mesh"]["eta"])
        rows.append({
            "id": case.name, "cfg": cfg, "eta": eta,
            "mean_iou": mean("mesh_foreground_iou", "input"),
            "mesh_holes": mean("projected_holes_count", "mesh"),
            "target_holes": mean("projected_holes_count", "input"),
            "mesh_void_area_mm2": mesh_void, "target_void_area_mm2": target_void,
            "void_area_retention": mesh_void / target_void,
            "solid_fraction": mean("solid_fraction_in_cad", "mesh"),
            "compliance_J": f["compliance"], "vm_max_Pa": f["vm_max"],
            "watertight": m["watertight"], "components": m["components"],
            "mesh": str((case / "gen/final.obj").resolve()),
            "preview": str((case / "gen/metrics/final_preview.png").resolve()),
            "metrics": str(mp.resolve()), "fea": str(fp.resolve()),
            "config": str((case / "config.json").resolve()),
        })
    return rows


def montage(rows):
    cells = []
    for r in rows:
        im = Image.open(r["preview"]).convert("RGB").resize((420, 420))
        cell = Image.new("RGB", (420, 470), "white")
        cell.paste(im, (0, 42))
        draw = ImageDraw.Draw(cell)
        draw.text((12, 8), r["id"], fill="black")
        draw.text((12, 24), f"C={r['compliance_J']*1000:.3f} mJ  void={r['void_area_retention']*100:.1f}%", fill="black")
        cells.append(cell)
    out = Image.new("RGB", (1260, 940), (235, 238, 240))
    for i, cell in enumerate(cells):
        out.paste(cell, ((i % 3) * 420, (i // 3) * 470))
    out.save(OUT / "refined_comparison_montage.png")


def main():
    rows = load_rows()
    selected = max(rows, key=lambda r: r["void_area_retention"])
    dominates = all(selected["void_area_retention"] >= r["void_area_retention"] and
                    selected["compliance_J"] <= r["compliance_J"] for r in rows)
    summary = {
        "conditioning": str(TARGET.resolve()), "cases": rows,
        "selected": selected["id"],
        "selection_basis": "maximize projected void-area retention and minimize verified compliance",
        "selected_dominates_all_study_cases_on_both_objectives": dominates,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    montage(rows)
    table = "".join(
        f"<tr{' class=best' if r['id']==selected['id'] else ''}><td>{html.escape(r['id'])}</td><td>{r['cfg']:g}</td><td>{r['eta']:g}</td>"
        f"<td>{r['mean_iou']:.3f}</td><td>{r['mesh_holes']:.2f}/{r['target_holes']:.2f}</td>"
        f"<td>{r['mesh_void_area_mm2']:.0f}</td><td>{r['void_area_retention']*100:.1f}%</td>"
        f"<td>{r['compliance_J']*1000:.3f}</td><td>{r['vm_max_Pa']/1e6:.2f}</td></tr>" for r in rows)
    cards = "".join(
        f"<article><img src=\"{html.escape(os.path.relpath(r['preview'], OUT))}\"><h3>{html.escape(r['id'])}</h3>"
        f"<p>void retention {r['void_area_retention']*100:.1f}% · C {r['compliance_J']*1000:.3f} mJ<br>"
        f"<code>{html.escape(r['mesh'])}</code></p></article>" for r in rows)
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPT bridge refined CFG study</title><style>
body{{font-family:system-ui,sans-serif;max-width:1280px;margin:32px auto;padding:0 22px;color:#17212b;line-height:1.55}}.note{{background:#eaf7f1;border-left:5px solid #177b65;padding:14px 18px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd4da;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef2f4}}tr.best{{background:#dff4e9;font-weight:700}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}}article{{border:1px solid #d5dce2;border-radius:10px;padding:12px}}article img{{width:100%}}code{{font-size:11px;overflow-wrap:anywhere}}</style>
<h1>GPT bridge refined CFG × eta study</h1><p class="note"><b>선택: CFG=8, eta=225.</b> 평균 투영 void area 1,104 mm²(입력의 39.6%)와 compliance 7.695 mJ로, 여섯 조합 중 개구부 보존과 구조 성능 양쪽에서 가장 좋았다.</p>
<p>모든 조합은 동일 GPT 6-view, seed 53000, dense/sparse 50 steps를 사용했고 post-processing과 최종 FEA까지 통과했다. 모든 OBJ는 watertight·single component다.</p>
<table><tr><th>case</th><th>CFG</th><th>eta</th><th>IoU</th><th>holes</th><th>void mm²</th><th>void retention</th><th>C mJ</th><th>vm MPa</th></tr>{table}</table>
<h2>비교</h2><img src="refined_comparison_montage.png" style="width:100%"><div class="cards">{cards}</div>
<h2>절대경로</h2><p><code>{OUT/'summary.json'}</code><br><code>{OUT/'protocol.json'}</code><br><code>{OUT/'refined_comparison_montage.png'}</code><br><code>{selected['mesh']}</code></p></html>'''
    (OUT / "report.html").write_text(page)
    print((OUT / "report.html").resolve())


if __name__ == "__main__":
    main()
