#!/usr/bin/env python3
"""Compare filled outer silhouettes of fixed-outline and outline-free final meshes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def mask(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    foreground = rgb.mean(2) < 245
    labels, count = ndimage.label(foreground, structure=np.ones((3, 3)))
    if count:
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        foreground = labels == sizes.argmax()
    return ndimage.binary_fill_holes(foreground)


def distances(rows: list[dict]) -> tuple[np.ndarray, dict]:
    # The bracket plate lies in the source CAD's XY plane.  `v_top` is therefore the
    # image-facing plate view; the renderer's `v00_front_lo` is an edge/oblique view.
    masks = [mask(Path(r["geometry_metrics"]).parent / "v_top.png") for r in rows]
    matrix = np.zeros((len(rows), len(rows)))
    values = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            union = np.logical_or(masks[i], masks[j]).sum()
            value = 1 - np.logical_and(masks[i], masks[j]).sum() / max(1, union)
            matrix[i, j] = matrix[j, i] = value; values.append(value)
    return matrix, {"mean_pairwise_outer_distance": float(np.mean(values)) if values else 0,
                    "median_pairwise_outer_distance": float(np.median(values)) if values else 0,
                    "max_pairwise_outer_distance": float(np.max(values)) if values else 0,
                    "pairs": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    sets = {}
    for round_index, name in [(0, "fixed_round0"), (1, "fixed_round1"), (2, "outline_free_round2")]:
        path = args.experiment / "shared_evaluations" / f"round_{round_index:02d}" / "verified_results.json"
        rows = json.loads(path.read_text())
        sets[name] = rows
        sets[name + "_valid"] = [r for r in rows if r["hard_gate_pass"]]
    output = args.experiment / "report_outline_free"; assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    summary, matrices = {}, {}
    for name, rows in sets.items():
        matrices[name], summary[name] = distances(rows)
        summary[name]["candidates"] = len(rows)
        summary[name]["hard_gate_pass"] = sum(r["hard_gate_pass"] for r in rows)
    (output / "outline_diversity.json").write_text(json.dumps(summary, indent=2) + "\n")

    compare_names = ["fixed_round0_valid", "fixed_round1_valid", "outline_free_round2_valid"]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=180)
    values = [summary[name]["mean_pairwise_outer_distance"] for name in compare_names]
    bars = ax.bar(compare_names, values, color=["#7890a0", "#477a9d", "#17826e"])
    ax.bar_label(bars, fmt="%.3f"); ax.set_ylabel("Mean pairwise filled-silhouette distance ↑")
    ax.set_title("Final-3D outer silhouette diversity · hard-gate-valid designs only"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(output / "outline_distance_comparison.png"); plt.close(fig)

    rows = sets["outline_free_round2"]
    sheet = Image.new("RGB", (6 * 320, 690), "white"); draw = ImageDraw.Draw(sheet)
    cards = []
    for index, row in enumerate(rows):
        source = Path(row["image_512"]); render = Path(row["final_preview"])
        input_dst = assets / f"{row['id']}_input.png"; final_dst = assets / f"{row['id']}_isometric.png"
        shutil.copy2(source, input_dst); shutil.copy2(render, final_dst)
        top = Image.open(source).convert("RGB").resize((300, 300))
        bottom = Image.open(render).convert("RGB").resize((300, 300))
        x = index * 320 + 10; sheet.paste(top, (x, 10)); sheet.paste(bottom, (x, 330))
        draw.text((x, 640), row["id"].replace("r2_outline_", ""), fill="black")
        draw.text((x, 660), f"valid={row['hard_gate_pass']}  C={1000*row['compliance_J']:.1f}mJ", fill="black")
        cards.append(f"<article><div><img src='assets/{input_dst.name}'><img src='assets/{final_dst.name}'></div>"
                     f"<h3>{row['id']}</h3><p>valid <b>{row['hard_gate_pass']}</b> · mass {row['normalized_mass']:.3f} · "
                     f"C {1000*row['compliance_J']:.2f} mJ · fix/load {row['fix_coverage_1p5mm']:.2f}/{row['load_coverage_1p5mm']:.2f} · "
                     f"t10 {row['thickness_p10_mm']:.2f} mm</p><code>{row['final_mesh']}</code></article>")
    sheet.save(output / "input_to_final_isometric.png")
    valid_gain = (summary['outline_free_round2_valid']['mean_pairwise_outer_distance'] /
                  summary['fixed_round1_valid']['mean_pairwise_outer_distance'] - 1) * 100
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BC-fixed outline-free probe</title><style>body{{font:15px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;background:#f2f5f7;color:#183044}}.note,article{{background:white;border:1px solid #d3dde4;border-radius:10px;padding:14px}}.note{{border-left:5px solid #16816d}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}}article div{{display:grid;grid-template-columns:1fr 1fr}}img{{width:100%}}code{{font-size:10px;overflow-wrap:anywhere}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}</style><h1>BC-fixed / outline-free probe</h1><section class="note">Design envelope를 목표 perimeter가 아니라 최대 허용 영역으로 사용했다. 6개 final 3D 중 {sum(r['hard_gate_pass'] for r in rows)}/6가 모든 hard gate를 통과했다. Hard-gate-valid 설계의 mean pairwise outer distance는 fixed round 1 <b>{summary['fixed_round1_valid']['mean_pairwise_outer_distance']:.3f}</b>에서 outline-free <b>{summary['outline_free_round2_valid']['mean_pairwise_outer_distance']:.3f}</b>로 <b>{valid_gain:.1f}%</b> 증가했다. 단, outline-free 유효 표본은 두 개뿐이므로 이 수치는 가능성 확인용이다.</section><img src="outline_distance_comparison.png"><img src="input_to_final_isometric.png"><h2>Input → final isometric 3D mesh render</h2><div class="grid">{''.join(cards)}</div><p>{output/'outline_diversity.json'}<br>{args.experiment/'protocol_amendment_round2_outline_free.json'}</p></html>'''
    (output / "index.html").write_text(page)
    print((output / "index.html").resolve())


if __name__ == "__main__": main()
