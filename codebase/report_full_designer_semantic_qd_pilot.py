#!/usr/bin/env python3
"""Build a cumulative report for the designer-steered Semantic BO-QD pilot."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = ["random", "auto_bo_qd", "initial_intent", "periodic_steering"]
LABELS = {
    "random": "Random",
    "auto_bo_qd": "Automatic BO-QD",
    "initial_intent": "Initial intent",
    "periodic_steering": "Periodic designer steering",
}
COLORS = {
    "random": "#777777",
    "auto_bo_qd": "#2476a6",
    "initial_intent": "#d38b2f",
    "periodic_steering": "#15836f",
}


def load(path: Path):
    return json.loads(path.read_text())


def mass_bin(value: float, edges: list[float]) -> int | None:
    if value < edges[0] or value > edges[-1]:
        return None
    return min(len(edges) - 2, int(np.digitize(value, edges[1:-1])))


def archive(rows: list[dict], edges: list[float]) -> dict:
    elites = {}
    for row in rows:
        if not row.get("hard_gate_pass") or row.get("compliance_J") is None:
            continue
        cell = (row.get("semantic_niche", row.get("semantic_niche_target")),
                mass_bin(row["normalized_mass"], edges))
        if cell[1] is None:
            continue
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


def gate_failures(row: dict, gates: dict) -> list[str]:
    failures = []
    if gates.get("watertight") and not row.get("final_watertight", False):
        failures.append("watertight")
    if row.get("final_components", 99) > gates["components_max"]:
        failures.append("components")
    checks = [
        ("fix_coverage_1p5mm", "fix BC"),
        ("load_coverage_1p5mm", "load BC"),
        ("inside_envelope_fraction", "envelope"),
        ("thickness_p10_mm", "thickness"),
    ]
    for key, label in checks:
        threshold = gates.get(key + "_min", gates.get("minimum_thickness_mm") if key == "thickness_p10_mm" else None)
        if threshold is not None and row.get(key, -np.inf) < threshold:
            failures.append(label)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.experiment.resolve()
    protocol = load(root / "protocol_snapshot.json")
    edges = protocol["archive"]["mass_edges"]
    niches = list(protocol["archive"]["semantic_niches"])
    gates = protocol["hard_gates"]
    warm = load(root / "shared_warm_start" / f"seed_{args.seed}" / "warm_start_records.json")["records"]
    verified = {i: load(root / "shared_evaluations" / f"round_{i:02d}" / "verified_results.json") for i in range(3)}
    by_id = {i: {r["id"]: r for r in rows} for i, rows in verified.items()}

    histories = {condition: [warm.copy()] for condition in CONDITIONS}
    selected_by_condition = {condition: [] for condition in CONDITIONS}
    for round_index in (0, 1):
        for condition in CONDITIONS:
            selection = load(root / "runs" / condition / f"seed_{args.seed}" / "rounds" /
                             f"round_{round_index:02d}" / "sparse_fea_selection.json")["selected"]
            rows = [by_id[round_index][r["id"]] for r in selection]
            selected_by_condition[condition].extend(rows)
            histories[condition].append(histories[condition][-1] + rows)

    summaries = {}
    progress = {}
    for condition in CONDITIONS:
        elites = archive(histories[condition][-1], edges)
        adaptive = selected_by_condition[condition]
        gaps = [r["semantic_realization"]["input_to_final_gap"] for r in adaptive if r.get("semantic_realization")]
        summaries[condition] = {
            "valid": sum(r["hard_gate_pass"] for r in adaptive),
            "total": len(adaptive),
            "occupied_cells": len(elites),
            "coverage": len(elites) / protocol["archive"]["cells"],
            "semantic_hv": semantic_hv(list(elites.values()), niches),
            "best_compliance_mJ": 1000 * min(r["compliance_J"] for r in elites.values()),
            "mean_semantic_gap": float(np.mean(gaps)),
        }
        progress[condition] = [
            {"stage": stage, "coverage": len(a := archive(rows, edges)) / protocol["archive"]["cells"],
             "semantic_hv": semantic_hv(list(a.values()), niches)}
            for stage, rows in zip(["warm", "round 0", "round 1"], histories[condition])
        ]

    report = root / "report_full"
    assets = report / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    outline = load(root / "report_outline_free" / "outline_diversity.json")
    payload = {
        "conditions_after_round_1": summaries,
        "cumulative_progress": progress,
        "outline_free_probe": outline,
        "scope_note": "Round 2 is an outline-policy amendment probe and is excluded from the matched four-condition comparison.",
    }
    (report / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
    for condition in CONDITIONS:
        axes[0].plot([p["stage"] for p in progress[condition]],
                     [100 * p["coverage"] for p in progress[condition]], "o-",
                     label=LABELS[condition], color=COLORS[condition])
        axes[1].plot([p["stage"] for p in progress[condition]],
                     [p["semantic_hv"] for p in progress[condition]], "o-",
                     label=LABELS[condition], color=COLORS[condition])
    axes[0].set(title="Cumulative archive coverage", ylabel="Coverage (%)")
    axes[1].set(title="Cumulative semantic QD-HV", ylabel="Normalized HV")
    for ax in axes:
        ax.grid(alpha=.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(report / "condition_progress.png")
    plt.close(fig)

    round2 = verified[2]
    cards = []
    for row in round2:
        image_dst = assets / f"{row['id']}_input.png"
        mesh_dst = assets / f"{row['id']}_isometric.png"
        shutil.copy2(row["image_512"], image_dst)
        shutil.copy2(row["final_preview"], mesh_dst)
        failures = gate_failures(row, gates)
        gap = row.get("semantic_realization", {}).get("input_to_final_gap", float("nan"))
        diagnosis_path = Path(row["final_mesh"]).parents[1] / "bc_preservation_diagnosis.json"
        diagnosis = load(diagnosis_path) if diagnosis_path.exists() else None
        final_containment = (diagnosis["stages"]["delivered_final"]["against"]["evaluation"]
                             if diagnosis else None)
        containment_text = (f" · BC volume containment fix/load "
                            f"{final_containment['fix']['interior_volume_containment_fraction']:.2f}/"
                            f"{final_containment['load']['interior_volume_containment_fraction']:.2f}"
                            if final_containment else "")
        cards.append(
            f"<article><div class='pair'><img src='assets/{image_dst.name}'><img src='assets/{mesh_dst.name}'></div>"
            f"<h3>{html.escape(row['id'].replace('r2_outline_', ''))}</h3>"
            f"<p class='{'ok' if row['hard_gate_pass'] else 'bad'}'>hard gate: <b>{row['hard_gate_pass']}</b>"
            f" · failure: {html.escape(', '.join(failures) if failures else 'none')}</p>"
            f"<p>mass {row['normalized_mass']:.3f} · C {1000*row['compliance_J']:.1f} mJ · "
            f"surface proximity fix/load {row['fix_coverage_1p5mm']:.2f}/{row['load_coverage_1p5mm']:.2f}{containment_text} · "
            f"t10 {row['thickness_p10_mm']:.2f} mm · semantic gap {gap:.3f}</p>"
            f"<code>{html.escape(row['final_mesh'])}</code></article>"
        )

    table = "".join(
        f"<tr><td>{LABELS[c]}</td><td>{s['valid']}/{s['total']}</td>"
        f"<td>{s['occupied_cells']}/24 ({100*s['coverage']:.1f}%)</td>"
        f"<td>{s['semantic_hv']:.4f}</td><td>{s['best_compliance_mJ']:.2f}</td>"
        f"<td>{s['mean_semantic_gap']:.4f}</td></tr>" for c, s in summaries.items()
    )
    fixed = outline["fixed_round1_valid"]["mean_pairwise_outer_distance"]
    free = outline["outline_free_round2_valid"]["mean_pairwise_outer_distance"]
    gain = 100 * (free / fixed - 1)
    valid2 = sum(r["hard_gate_pass"] for r in round2)
    page = f"""<!doctype html><html lang='ko'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Designer-steered Semantic BO-QD · integrated pilot</title>
<style>body{{font:15px system-ui;max-width:1500px;margin:30px auto;padding:0 22px;color:#183044;background:#f3f6f8}}section,article{{background:white;border:1px solid #d3dce3;border-radius:10px;padding:15px}}.lead{{border-left:5px solid #16816d}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #ced9e0;padding:8px;text-align:left}}th{{background:#eaf1f5}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:5px}}img{{width:100%}}code{{font-size:10px;overflow-wrap:anywhere}}.ok{{color:#08734f}}.bad{{color:#a23b2c}}@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>Designer-steered Semantic BO-QD · integrated pilot</h1>
<section class='lead'><b>핵심 결과.</b> BC interface만 고정하고 design envelope를 최대 허용 영역으로 바꾸자, hard-gate-valid final mesh의 평균 pairwise outer-silhouette distance가 <b>{fixed:.3f}</b>에서 <b>{free:.3f}</b>로 <b>{gain:.1f}%</b> 증가했다. 자유 외곽선 probe의 통과율은 <b>{valid2}/6</b>이므로, 다음 라운드는 외곽선을 다시 고정하는 대신 load lug와 얇은 member만 국부 보강해야 한다.</section>
<h2>Matched condition comparison · warm start + rounds 0–1</h2>
<table><tr><th>condition</th><th>adaptive valid</th><th>archive coverage</th><th>semantic QD-HV</th><th>best C (mJ)</th><th>mean semantic gap ↓</th></tr>{table}</table>
<img src='condition_progress.png'>
<p>Round 2 outline-free probe는 실험 중 명시적으로 추가한 protocol amendment이므로 위 네 조건의 공정 비교에는 포함하지 않았다.</p>
<h2>Outline-free probe · input image → final 3D mesh render</h2>
<div class='grid'>{''.join(cards)}</div>
<h2>판정</h2><section><p><b>BC 진단:</b> 여섯 후보 모두 sparse 정렬 후와 post boolean 직후에는 fix/load BC 체적 포함률이 1.00/1.00이었다. 전달용 surface remesh 뒤 load 포함률은 fork 0.91, hourglass 0.60, scalloped 1.00, spoke-island 0.61, swept 1.00, tapered 1.00으로 바뀌었다. 따라서 hourglass와 spoke-island의 실제 결손은 생성이나 boolean이 아니라 마지막 remesh/repair에서 발생한다.</p><p><b>평가 제한:</b> 기존 surface-proximity gate는 BC 전체 표면과 final surface의 거리이므로, 내부에 완전히 채워진 BC 표면도 낮게 평가할 수 있다. 최종 판정에는 volumetric containment를 추가해야 한다.</p><p><b>다음 수정:</b> body remesh를 먼저 끝낸 뒤 exact BC union을 마지막에 수행하고, 그 뒤에는 topology-changing repair를 적용하지 않는다. 동시에 BC gate를 surface proximity와 volume containment로 분리한다.</p></section>
<p>Summary: {report/'summary.json'}<br>Outline metrics: {root/'report_outline_free'/'outline_diversity.json'}<br>Protocol amendment: {root/'protocol_amendment_round2_outline_free.json'}</p></html>"""
    (report / "index.html").write_text(page)
    print((report / "index.html").resolve())


if __name__ == "__main__":
    main()
