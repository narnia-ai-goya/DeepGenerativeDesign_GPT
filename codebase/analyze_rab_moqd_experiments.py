"""Aggregate RAB-MOQD method, replication, surrogate, and grid-sensitivity studies."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rab_moqd_acquisition import cell_of, hypervolume_2d, nondominated, objective_point

ROOT = Path(__file__).resolve().parents[1]
WARM = ROOT / "experiments/bracket/rab_moqd_warmstart_nu030_2026-09-14.json"
SUITE = ROOT / "experiments/bracket/rab_moqd_experiment_suite_2026-09-14/summary.json"
JOINT = ROOT / "experiments/bracket/rab_moqd_joint_residual_2026-09-14/summary.json"
SURROGATE = ROOT / "experiments/bracket/rab_moqd_surrogate_audit_2026-09-14/surrogate_audit.json"
EXPANDED = ROOT / "experiments/bracket/rab_moqd_expanded_2026-09-14/summary.json"
GPT_IMAGE_VALIDATION = ROOT / "experiments/bracket/gpt_image_bridge_validation_2026-09-14/summary.json"
OUT = ROOT / "experiments/bracket/rab_moqd_combined_analysis_2026-09-14"
OBJECTIVE_BOUNDS = [[.0035, .014], [.35, .70]]


def metrics(rows, thresholds):
    fronts = {}
    for row in rows:
        if not row.get("valid"): continue
        cell = tuple(int(np.searchsorted(edge, value, side="right"))
                     for value, edge in zip(row["realized_descriptors"], thresholds))
        point = objective_point(row["compliance_J"], row["measured_design_volume_fraction"],
                                OBJECTIVE_BOUNDS)
        fronts.setdefault(cell, []).append(point)
    return {"occupied_cells": len(fronts),
            "qd_hypervolume": sum(hypervolume_2d(nondominated(front)) for front in fronts.values())}


def cumulative_metrics(warm, rows, thresholds):
    """Return archive metrics after each attempted evaluation is appended."""
    base = metrics(warm, thresholds)
    trajectory = [{"evaluation": 0, "id": "warm_start", "valid": True,
                   **base, "new_cells": 0, "qd_hypervolume_gain": 0.0}]
    accumulated = list(warm)
    for evaluation, row in enumerate(rows, start=1):
        accumulated.append(row)
        current = metrics(accumulated, thresholds)
        trajectory.append({
            "evaluation": evaluation,
            "id": row["id"],
            "valid": bool(row.get("valid")),
            **current,
            "new_cells": current["occupied_cells"]-base["occupied_cells"],
            "qd_hypervolume_gain": current["qd_hypervolume"]-base["qd_hypervolume"],
        })
    return trajectory


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    warm = json.loads(WARM.read_text())["results"]
    suite = json.loads(SUITE.read_text()); joint_summary = json.loads(JOINT.read_text())
    surrogate = json.loads(SURROGATE.read_text())
    expanded = json.loads(EXPANDED.read_text())
    gpt_image_validation = json.loads(GPT_IMAGE_VALIDATION.read_text())["result"]
    methods = {name: [row for row in suite["comparison_results"] if row["method"] == name]
               for name in ("posterior_sampled", "posterior_mean", "sobol_random")}
    methods["joint_residual"] = joint_summary["attempts"]
    values = np.asarray([row["realized_descriptors"] for row in warm])
    resolution = {}
    thresholds_by_resolution = {}
    for count in (4, 6, 8, 10):
        thresholds = np.quantile(values, np.arange(1, count)/count, axis=0).T
        thresholds_by_resolution[str(count)] = thresholds.tolist()
        base = metrics(warm, thresholds)
        resolution[str(count)] = {"base": base, "methods": {}}
        for name, rows in methods.items():
            combined = metrics([*warm, *rows], thresholds)
            resolution[str(count)]["methods"][name] = {
                **combined, "new_cells": combined["occupied_cells"]-base["occupied_cells"],
                "qd_hypervolume_gain": combined["qd_hypervolume"]-base["qd_hypervolume"],
                "cumulative": cumulative_metrics(warm, rows, thresholds)}

    report = {
        "status": "method_development_not_final_benchmark",
        "new_full_pipeline_evaluations": 3+9+4+3+18+1,
        "methods": {name: {"evaluations": len(rows), "valid": sum(row["valid"] for row in rows)}
                    for name, rows in methods.items()},
        "grid_sensitivity": resolution,
        "recommended_next_grid": {
            "type": "8x8 development-quantile thresholds with unbounded outer cells",
            "thresholds": thresholds_by_resolution["8"],
            "reason": "41/64 warm-start cells occupied, leaving 23 cells while retaining robust open tails",
        },
        "surrogate_cell_accuracy": {name: data["cell_mean_prediction"]
                                    for name, data in surrogate["metrics"].items()},
        "gp_uncertainty_multipliers": {
            output: surrogate["metrics"]["mixed_exact_gp"][output]["std_multiplier_for_grouped_90_coverage"]
            for output in ("normalized_void_scale", "strain_energy_concentration",
                           "volume_fraction", "log_compliance")},
        "replication": suite["verification"],
        "expanded_comparison": {
            "protocol": expanded["protocol"],
            "methods": expanded["methods"],
            "new_posterior_target_hits": sum(
                list(cell_of(row["realized_descriptors"],
                             thresholds=thresholds_by_resolution["8"])) == row["target_cell"]
                for row in expanded["new_results"]
                if row["method"] == "posterior_sampled" and row.get("valid")),
            "new_posterior_evaluations": sum(
                row["method"] == "posterior_sampled" for row in expanded["new_results"]),
        },
        "gpt_image_conditioning_validation": {
            key: gpt_image_validation.get(key) for key in (
                "valid", "invalid_reasons", "compliance_J",
                "measured_design_volume_fraction", "realized_descriptors",
                "containment_fraction", "mesh", "preview")
        },
        "conclusions": [
            "The proposed behavior descriptors produce meaningful spread, but recipe-level cell membership is unstable.",
            "Method ranking changes under post-hoc cell definitions, so grid thresholds must be preregistered.",
            "Independent Gaussian GP draws are under-dispersed; grouped residual bootstrap improves closed-grid QD-HV but misses target cells.",
            "The next method should combine calibrated joint residual sampling, explicit style diversity, and a frozen Sobol exploration fraction.",
            "Archive immutable verified meshes immediately; use replication to learn realization noise rather than to invalidate an already verified mesh.",
            "At 12 evaluations, Gaussian posterior achieved higher final QD-HV gain and cumulative-gain AUC than Sobol, while Sobol occupied more new cells.",
            "None of the nine expanded Gaussian-posterior proposals hit its predicted 8x8 target cell, so archive improvement and behavior control must be reported separately.",
        ],
    }
    (OUT / "combined_analysis.json").write_text(json.dumps(report, indent=2)+"\n")

    labels = {"posterior_sampled": "Gaussian posterior", "posterior_mean": "Posterior mean",
              "sobol_random": "Sobol random", "joint_residual": "Joint residual"}
    colors = {"posterior_sampled": "#2767a5", "posterior_mean": "#25816d",
              "sobol_random": "#d07a2b", "joint_residual": "#8659a5"}
    fig, ax = plt.subplots(figsize=(8.4, 5.3), layout="constrained")
    for name in methods:
        ax.plot([4, 6, 8, 10], [resolution[str(n)]["methods"][name]["qd_hypervolume_gain"]
                                for n in (4, 6, 8, 10)], marker="o", lw=2,
                label=labels[name], color=colors[name])
    ax.set(xlabel="Open quantile grid resolution", ylabel="QD-HV gain over warm start",
           title="Method ranking depends on archive resolution", xticks=[4, 6, 8, 10])
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(OUT / "grid_sensitivity.png", dpi=190); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.3), layout="constrained")
    cumulative_8x8 = resolution["8"]["methods"]
    for name in methods:
        trajectory = cumulative_8x8[name]["cumulative"]
        ax.plot([step["evaluation"] for step in trajectory],
                [step["qd_hypervolume"] for step in trajectory],
                marker="o", lw=2, label=labels[name], color=colors[name])
    ax.set(xlabel="Cumulative full-pipeline evaluations",
           ylabel="Cumulative QD-HV",
           title="Cumulative QD-HV on the preregistered 8×8 grid",
           xticks=range(4))
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(OUT / "cumulative_qd_hv_8x8.png", dpi=190); plt.close(fig)

    cumulative_rows = []
    for name in methods:
        values = cumulative_8x8[name]["cumulative"]
        cumulative_rows.append(
            f'<tr><td>{labels[name]}</td>'
            + ''.join(f'<td>{step["qd_hypervolume"]:.5f}</td>' for step in values)
            + f'<td>{values[-1]["qd_hypervolume_gain"]:+.5f}</td></tr>')

    expanded_rows = []
    for name in ("posterior_sampled", "sobol_random"):
        item = expanded["methods"][name]
        expanded_rows.append(
            f'<tr><td>{labels[name]}</td><td>{item["evaluations"]}</td>'
            f'<td>{item["valid"]}</td><td>{item["new_cells"]:+d}</td>'
            f'<td>{item["qd_hypervolume_gain"]:.5f}</td>'
            f'<td>{item["auc_gain"]:.5f}</td></tr>')

    cards = []
    all_rows = [*suite["comparison_results"], *joint_summary["attempts"]]
    for row in all_rows:
        image = (f'<img src="{html.escape(os.path.relpath(row["preview"], OUT))}">'
                 if row.get("preview") else "")
        detail = (f"C={1000*row['compliance_J']:.3f} mJ · V={row['measured_design_volume_fraction']:.3f}"
                  if row.get("valid") else "invalid")
        cards.append(f'<article>{image}<h3>{html.escape(row["id"])}</h3><p>{detail}</p></article>')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RAB-MOQD combined analysis</title><style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:34px auto;padding:0 22px;color:#17212b;line-height:1.6}}.note{{background:#fff4dc;border-left:4px solid #c68412;padding:13px 17px}}.verdict{{background:#eaf7f1;border-left:4px solid #177b65;padding:13px 17px}}img.plot{{max-width:1000px;width:100%}}table{{border-collapse:collapse;width:100%;max-width:1000px}}th,td{{border:1px solid #ccd4da;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef2f4}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(245px,1fr));gap:14px}}article{{border:1px solid #d5dce2;border-radius:10px;padding:12px}}article img{{width:100%}}code{{overflow-wrap:anywhere}}</style><h1>RAB-MOQD combined development analysis</h1><p class="note">새 full-pipeline 평가 38회의 개발 결과다. 확장 비교는 각 방법 12회이므로 효과 크기를 확인하는 단계이며, 독립 반복을 통한 우월성 검정은 후속 과제다.</p><h2>GPT image → 3D 검증</h2><p class="verdict"><b>GPT 내장 이미지 도구로 만든 6-view conditioning에서 유효한 bracket OBJ를 생성했다.</b> C={1000*gpt_image_validation['compliance_J']:.3f} mJ, V={gpt_image_validation['measured_design_volume_fraction']:.3f}, containment={gpt_image_validation['containment_fraction']:.4f}. 큰 개구부 일부가 3D에서 메워져 image-to-3D 충실도 개선은 남아 있다.</p><div class="cards"><article><img src="../../../output/imagegen/bracket_gpt_structural_2026-09-14/gpt_contact_sheet.png"><h3>GPT 6-view conditioning</h3></article><article><img src="../gpt_image_bridge_validation_2026-09-14/cases/gpt_bridge_loadpath_00/gen/metrics/final_preview.png"><h3>Validated 3D realization</h3></article></div><p><a href="../gpt_image_bridge_validation_2026-09-14/report.html">GPT image 검증 상세 보기</a></p><h2>확장 비교: 방법별 12회</h2><p class="verdict"><b>Gaussian posterior가 최종 QD-HV gain과 누적 AUC에서 Sobol을 앞섰다.</b> 반면 Sobol은 더 많은 신규 cell을 채웠고, 새 posterior 후보의 정확한 목표 cell 적중은 0/9였다. 품질 개선과 behavior control을 분리해 해석해야 한다.</p><img class="plot" src="../rab_moqd_expanded_2026-09-14/cumulative_qd_hv.png"><table><thead><tr><th>방법</th><th>평가</th><th>유효</th><th>신규 cell</th><th>최종 QD-HV 증가</th><th>누적 gain AUC</th></tr></thead><tbody>{''.join(expanded_rows)}</tbody></table><p><a href="../rab_moqd_expanded_2026-09-14/report.html">확장 실험의 전체 realization 보기</a></p><h2>초기 3회 누적 진단</h2><p>각 방법을 동일한 warm-start archive에서 시작해 평가 결과를 실행 순서대로 하나씩 누적했다. 무효 평가는 횟수에는 포함되지만 archive와 QD-HV를 바꾸지 않는다.</p><img class="plot" src="cumulative_qd_hv_8x8.png"><table><thead><tr><th>방법</th><th>Warm start</th><th>+1</th><th>+2</th><th>+3</th><th>최종 증가</th></tr></thead><tbody>{''.join(cumulative_rows)}</tbody></table><h2>Grid sensitivity</h2><img class="plot" src="grid_sensitivity.png"><h2>초기 Realizations</h2><div class="cards">{''.join(cards)}</div><p><code>{OUT / 'combined_analysis.json'}</code></p></html>'''
    (OUT / "report.html").write_text(page)
    print(OUT / "report.html")


if __name__ == "__main__": main()
