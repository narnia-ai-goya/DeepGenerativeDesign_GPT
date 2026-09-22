#!/usr/bin/env python3
"""Analyze and render the RC-BQD v2 four-case microbatch."""
import html
import json
from pathlib import Path

from raqd_core import cell_index

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/bracket/rcbqd_v2_microbatch_2026-09-13"
SOURCE = ROOT / "experiments/bracket/raqd_online_2026-09-13"
STEP001 = ROOT / "experiments/bracket/raqd_v2_repair_pilot_2026-09-13/summary.json"
STEP003 = ROOT / "experiments/bracket/raqd_v2_repair_pilot_step003_2026-09-13/summary.json"


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    elites = {}
    target_hits = 0
    for row in summary["results"]:
        if row["status"] != "complete":
            row["failure_class"] = "repair disconnected load interface"
            continue
        d = [row["repaired"]["void_clearance_mean_mm"], row["repaired"]["material_anisotropy"]]
        cell = cell_index(d, protocol["dims"], protocol["descriptor_ranges"])
        row["realized_cell"] = list(cell) if cell else None
        row["target_hit"] = row["proposal"]["target_cell"] == row["realized_cell"]
        target_hits += row["target_hit"]
        if row["repair_pass"] and cell is not None:
            old = elites.get(cell)
            if old is None or row["repaired"]["compliance_J"] < old["repaired"]["compliance_J"]:
                elites[cell] = row
    s1, s3 = json.loads(STEP001.read_text()), json.loads(STEP003.read_text())
    analysis = {
        "status": summary["status"], "evaluations": 4,
        "verified_repairs": sum(r.get("repair_pass", False) for r in summary["results"]),
        "verified_repair_rate": sum(r.get("repair_pass", False) for r in summary["results"])/4,
        "target_cell_hits_among_complete": target_hits,
        "target_cell_hit_rate_among_complete": target_hits/max(summary["complete"],1),
        "occupied_cells": [list(c) for c in sorted(elites)],
        "coverage": len(elites)/16,
        "qd_score": sum(1/(1+r["repaired"]["compliance_J"]/.01) for r in elites.values()),
        "elites": [{"cell": list(c), "id": r["id"], "compliance_J": r["repaired"]["compliance_J"]}
                   for c,r in sorted(elites.items())],
        "step_size_pair": {
            "step_0.01": s1["repaired"], "step_0.03": s3["repaired"],
            "compliance_change_0.03_vs_0.01_percent": 100*(s3["repaired"]["compliance_J"]/
                                                            s1["repaired"]["compliance_J"]-1),
            "interpretation": "Promising single pair; sparse CUDA path is not fully deterministic."
        }
    }
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    cards=[]
    for row in summary["results"]:
        if row["status"] == "complete":
            r=row["repaired"]; ok="통과" if row["repair_pass"] else "실패"
            cards.append(f'''<article><h3>{row['id']} · {ok}</h3><img src="cases/{row['id']}/repaired/gen/metrics/final_preview.png"><p>target {row['proposal']['target_cell']} → realized <b>{row['realized_cell']}</b><br>V={r['material_volume_fraction']:.4f} · C={r['compliance_J']*1000:.3f} mJ<br>watertight={r['watertight']} · components={r['components']} · containment={r['containment_fraction']:.4f}</p></article>''')
        else:
            cards.append(f'''<article class="fail"><h3>{row['id']} · 실패</h3><img src="cases/{row['id']}/repaired/gen/metrics/final_preview.png"><p>repair 과정에서 body/load 연결이 끊겼고 FEA가 load facet 0개를 검출했다. 이 평가는 예산에 포함하며 archive에는 넣지 않는다.</p></article>''')
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RC-BQD v2 microbatch</title><style>body{{font-family:system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px 70px;color:#17212b;background:#f7f5ef;line-height:1.58}}h1{{font-size:2.7rem;letter-spacing:-.04em}}.verdict{{background:#eaf7f1;border-left:6px solid #177a65;padding:16px 20px}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}}article{{background:white;border:1px solid #d8dddf;border-radius:11px;padding:15px}}article.fail{{border-top:5px solid #b54b2b}}img{{width:100%;border-radius:8px}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #d8dddf;padding:10px;text-align:left}}th{{background:#eaf0f2}}.path{{font-family:ui-monospace,monospace;overflow-wrap:anywhere;color:#586671}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}</style><body><h1>RC-BQD v2 micro-batch</h1><p class="verdict"><b>Repair 3/4 성공.</b> 네 평가에서 2개 cell을 채웠고 QD score는 {analysis['qd_score']:.4f}였다. 목표 cell 적중은 완료 3개 중 1개였다.</p><div class="grid">{''.join(cards)}</div><h2>판정</h2><table><tr><th>항목</th><th>결과</th></tr><tr><td>Repair 검증 통과</td><td>3/4 (75%)</td></tr><tr><td>점유 cell</td><td>{analysis['occupied_cells']} · 2/16 (12.5%)</td></tr><tr><td>목표 cell 적중</td><td>1/3 complete</td></tr><tr><td>실패 원인</td><td>iso-level repair 후 다중 shell, load interface 소실</td></tr><tr><td>step size 0.03 paired pilot</td><td>repair 후 5.294 mJ, 0.01의 5.634 mJ보다 {abs(analysis['step_size_pair']['compliance_change_0.03_vs_0.01_percent']):.1f}% 낮음</td></tr></table><p>체적 repair는 feasibility를 크게 높였지만 descriptor posterior는 아직 약하다. 다음 acquisition은 cell 소속확률 기준을 강화하고, repair 단계는 체적뿐 아니라 BC 연결과 shell topology를 검사해야 한다. <code>sp_fea_step_size=0.03</code>은 다음 후보 설정으로 유지하되 반복 실험 전까지 확정값으로 간주하지 않는다.</p><h2>절대경로</h2><p class="path">{OUT/'report.html'}<br>{OUT/'analysis.json'}<br>{OUT/'summary.json'}<br>{STEP003}</p></body></html>'''
    (OUT / "report.html").write_text(page)
    print(OUT / "analysis.json"); print(OUT / "report.html")


if __name__ == "__main__": main()
