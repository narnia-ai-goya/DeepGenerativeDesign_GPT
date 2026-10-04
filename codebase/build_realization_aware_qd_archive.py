#!/usr/bin/env python3
"""Build an intent × realized-shape Pareto-QD audit archive from verified meshes.

This does not train the image or 3D generator.  It separates structural intent
from requested mass, then measures realized shape on the delivered final mesh.
The shape clustering threshold is explicit and a sensitivity table is saved.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from audit_semantic_qd_geometry import occupancy, views, equal_area, view_distance
from run_framecorrected_semantic_pareto_qd import embed_texts, pareto_front

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22"
SOURCE = STUDY / "framecorrected_semantic_pareto_qd/archive.json"
DEFAULT_OUT = STUDY / "realization_aware_qd_archive_v2"
DEFAULT_ADDITIONS = DEFAULT_OUT / "verified_additions.json"
DEFAULT_GEOMETRY_GATES = DEFAULT_OUT / "geometry_gate_results.json"


def canonical_intents(rows: list[dict]) -> dict[str, str]:
    bank = json.loads((STUDY / "prompt_bank.json").read_text())["prompts"]
    result = {}
    for row in bank:
        if row["mass_level_target"] == "medium":
            # Keep only the structural clause.  Shared BC/render boilerplate and
            # low/medium/high mass adjectives must not move a semantic QD cell.
            result[row["semantic_niche_target"]] = row["prompt"].split(":", 1)[1].split(";", 1)[0].strip()
    result["longitudinal_spine"] = (
        "two mostly vertical, slightly asymmetric continuous load paths "
        "connected by a few broad transverse bridges and elongated openings")
    missing = {r["semantic_niche_target"] for r in rows} - set(result)
    if missing:
        raise ValueError(f"No canonical structural intent for {sorted(missing)}")
    return result


def add_verified_variants(rows: list[dict], additions: Path) -> None:
    variants = json.loads(additions.read_text())["records"] if additions.exists() else []
    existing = {row["id"] for row in rows}
    for variant in variants:
        name = variant["id"]
        if name in existing:
            raise ValueError(f"Duplicate verified candidate: {name}")
        mesh_path, fea_path = Path(variant["final_mesh"]), Path(variant["fea_summary"])
        render, image = Path(variant["comparison_render"]), Path(variant["input_image"])
        if not all(p.is_file() for p in (mesh_path, fea_path, render, image)):
            raise FileNotFoundError(f"Incomplete verified candidate {name}")
        mesh = trimesh.load_mesh(mesh_path, force="mesh", process=False)
        fea = json.loads(fea_path.read_text())
        if not mesh.is_watertight or not 0 < fea["compliance"] < 1:
            raise ValueError(f"Invalid final mesh or FEA for {name}")
        rows.append({"id": name, "semantic_niche_target": variant["semantic_niche_target"],
                     "mass_level_target": "measured", "source_round": variant.get("source_round", 5),
                     "compliance_J": float(fea["compliance"]),
                     "volume_mm3": abs(float(mesh.volume)) * 1e9,
                     "vm_max_MPa": float(fea["vm_max"]) / 1e6,
                     "final_mesh": str(mesh_path), "comparison_render": str(render),
                     "input_image": str(image)})


def shape_distance(rows: list[dict], resolution: int) -> np.ndarray:
    domain = trimesh.load_mesh(ROOT / "data_real/bracket/original_DesignSpace.stl", process=False)
    lo, hi = domain.bounds
    pitch = float(np.max(hi - lo) / (resolution - 1))
    shape = tuple((np.ceil((hi - lo) / pitch).astype(int) + 1).tolist())
    bc = occupancy(ROOT / "data_real/bracket/fixed.stl", lo, pitch, shape)
    bc |= occupancy(ROOT / "data_real/bracket/load.stl", lo, pitch, shape)
    maps = []
    for row in rows:
        occ = occupancy(Path(row["final_mesh"]), lo, pitch, shape)
        occ[bc] = False
        maps.append(views(occ))
    bc_views = views(bc)
    areas = [int(np.median([v[k].sum() for v in maps])) for k in range(3)]
    normalized = [tuple(equal_area(v[k], ~bc_views[k], areas[k]) for k in range(3))
                  for v in maps]
    n = len(rows)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            d[i, j] = d[j, i] = view_distance(normalized[i], normalized[j])
    return d


def make_report(out: Path, rows: list[dict], summary: dict) -> None:
    def url(path: str) -> str:
        return "/" + str(Path(path).resolve().relative_to(ROOT))

    cards = []
    for row in rows:
        gate = ("pass" if row["geometry_gate_pass"] is True else
                "fail" if row["geometry_gate_pass"] is False else "unverified")
        cards.append(f'''<article class="{'elite' if row['pareto_elite'] else ''}">
        <img src="{html.escape(url(row['comparison_render']))}">
        <h3>{html.escape(row['id'])}</h3>
        <p>intent: {html.escape(row['semantic_niche_target'])} · shape cluster {row['shape_cluster']}<br>
        C={row['compliance_J']:.5f} J · V={row['volume_mm3']:,.0f} mm³<br>
        geometry gate: {gate} · nearest realized shape: {row['nearest_shape_distance']:.3f} ·
        {'Pareto elite' if row['pareto_elite'] else 'dominated'}</p>
        <a href="{html.escape(url(row['final_mesh']))}">OBJ</a></article>''')
    sensitivity = "".join(f"<tr><td>{t}</td><td>{n}</td></tr>" for t, n in summary["shape_cluster_sensitivity"].items())
    audit_path = out / "image_mesh_realization_audit.json"
    audit_note = ""
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit.get("n_records") == len(rows):
            audit_note = (f"<p>현재 자료 {audit['n_pairs']}개 이미지–mesh 쌍에서 입력 이미지 거리와 최종 형상 거리의 "
                          f"순위 상관은 {audit['all_pairs_spearman']:.3f}이다. "
                          "과거 후보들은 생성 설정이 섞여 있어 이 값은 진단용이며, 고정 설정의 전이 성능 추정치는 아니다. "
                          '<a href="image_mesh_realization_audit.json">진단 JSON</a></p>')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Realization-aware QD audit v2</title>
    <style>body{{font:15px/1.5 system-ui;max-width:1500px;margin:24px auto;padding:0 18px;background:#f4f6f8;color:#17212b}}.note,article,table{{background:white;border:1px solid #d9e0e6;border-radius:10px}}.note{{padding:16px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:14px}}article{{padding-bottom:12px}}article.elite{{border:2px solid #26865d}}article img{{width:100%;display:block;border-radius:10px 10px 0 0}}article h3,article p,article a{{margin:8px 12px}}table{{border-collapse:collapse;margin:18px 0}}th,td{{padding:8px 14px;border:1px solid #d9e0e6}}a{{overflow-wrap:anywhere}}</style>
    <h1>Realization-aware QD archive v2 · audit</h1>
    <div class="note">구조 의도 텍스트에서 질량 표현을 분리해 <b>{summary['intent_niches']}개 intent niche</b>를 사용한다. 각 최종 mesh의 BC 영역을 제외하고, 동일 물리 격자의 3-view 형상을 동일 투영 면적으로 정규화해 <b>실현 형상 거리</b>를 계산한다. 각 intent × shape cluster에서는 독립 FEA compliance와 최종 체적의 비지배 해를 보존한다.<br>
    기존 prompt-PCA 방식의 겉보기 8/9 cell 점유는 질량 단어만 달라져도 cell이 바뀌는 문제를 포함한다. 현재 shape cluster는 임계값 {summary['shape_threshold']:.2f}의 탐색적 분할이며, 아래 민감도 확인 전 논문 QD coverage로 확정하면 안 된다.</div>
    <p>최종 mesh {summary['records']}개 중 geometry gate 통과 {summary['hard_gate_pass']}개 · 통과 후보의 형상 cluster {summary['shape_clusters']}개 · intent × shape cell {summary['occupied_cells']}개 · Pareto elite {summary['pareto_elites']}개</p>{audit_note}
    <h2>형상 분할 민감도</h2><table><tr><th>최대 cluster 거리</th><th>형상 cluster 수</th></tr>{sensitivity}</table>
    <h2>후보</h2><div class="grid">{''.join(cards)}</div><p><a href="archive.json">Archive JSON</a> · <a href="protocol.json">Protocol</a></p></html>'''
    (out / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--resolution", type=int, default=48)
    ap.add_argument("--shape-threshold", type=float, default=0.15)
    ap.add_argument("--additions", type=Path, default=DEFAULT_ADDITIONS)
    ap.add_argument("--geometry-gates", type=Path, default=DEFAULT_GEOMETRY_GATES)
    args = ap.parse_args()
    rows = json.loads(args.source.read_text())["records"]
    add_verified_variants(rows, args.additions)
    gate_results = json.loads(args.geometry_gates.read_text()) if args.geometry_gates.exists() else {}
    for row in rows:
        gate = gate_results.get(row["id"])
        if gate is not None and Path(gate.get("final_mesh", "")).resolve() != Path(row["final_mesh"]).resolve():
            raise ValueError(f"Geometry gate belongs to a different mesh: {row['id']}")
        row["geometry_gate_pass"] = gate.get("geometry_gate_pass") if gate else None
        row["geometry_metrics"] = gate.get("geometry_metrics") if gate else None
    intent_text = canonical_intents(rows)
    names = sorted(intent_text)
    embedding = embed_texts([intent_text[n] for n in names])
    intent_distances = np.linalg.norm(embedding[:, None, :] - embedding[None, :, :], axis=2)
    d = shape_distance(rows, args.resolution)
    eligible = [i for i, row in enumerate(rows) if row["geometry_gate_pass"] is True]
    d_eligible = d[np.ix_(eligible, eligible)]
    hierarchy = linkage(squareform(d_eligible), method="complete") if len(eligible) > 1 else None
    thresholds = [0.10, 0.12, 0.15, 0.18]
    sensitivity = {f"{t:.2f}": int(len(set(fcluster(hierarchy, t=t, criterion="distance")))) if hierarchy is not None else len(eligible)
                   for t in thresholds}
    clusters = np.zeros(len(rows), dtype=int)
    if hierarchy is not None:
        clusters[eligible] = fcluster(hierarchy, t=args.shape_threshold, criterion="distance")
    elif len(eligible) == 1:
        clusters[eligible[0]] = 1
    for i, row in enumerate(rows):
        row["shape_cluster"] = int(clusters[i])
        row["intent_text"] = intent_text[row["semantic_niche_target"]]
        peers = [j for j in eligible if j != i] or [j for j in range(len(rows)) if j != i]
        row["nearest_shape_distance"] = float(min(d[i, j] for j in peers))
        nearest_idx = min(peers, key=lambda j: d[i, j])
        row["nearest_shape_id"] = rows[nearest_idx]["id"]
        row["pareto_elite"] = False
    cells = {}
    for row in rows:
        if row["geometry_gate_pass"] is not True:
            continue
        key = (row["semantic_niche_target"], row["shape_cluster"])
        cells.setdefault(key, []).append(row)
    for group in cells.values():
        elites = {r["id"] for r in pareto_front(group)}
        for row in group:
            row["pareto_elite"] = row["id"] in elites
    protocol = {
        "intent_descriptor": "canonical structural intent label; mass and render boilerplate removed",
        "intent_text_distance": "MiniLM distance between canonical clauses is reported for navigation, not used to assign labels",
        "intent_labels": names, "intent_texts": intent_text,
        "intent_embedding_distance_matrix": intent_distances.tolist(),
        "realized_shape_descriptor": "3-view mean Jaccard of final-mesh occupancy at isotropic physical pitch, BC masked and projection area normalized",
        "shape_resolution": args.resolution,
        "shape_clustering": {"algorithm": "complete linkage", "threshold": args.shape_threshold,
                             "status": "pilot; report sensitivity rather than treating coverage as calibrated"},
        "quality": ["minimize independent final-mesh FEA compliance_J",
                    "minimize final-mesh volume_mm3"],
        "cell": "canonical intent label × realized shape cluster",
        "geometry_hard_gate": "watertight, one component, fix coverage ≥0.70, load coverage ≥0.60, envelope containment ≥0.995, thickness p10 ≥3 mm",
        "notes": "No model training. Same intent remains same semantic niche across mass levels. Only measured geometry-pass records enter Pareto cells."
    }
    summary = {"records": len(rows), "hard_gate_pass": len(eligible), "intent_niches": len(names),
               "shape_clusters": len(set(clusters[eligible])), "occupied_cells": len(cells),
               "pareto_elites": sum(r["pareto_elite"] for r in rows),
               "shape_threshold": args.shape_threshold,
               "shape_cluster_sensitivity": sensitivity}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    (out / "archive.json").write_text(json.dumps({"protocol": protocol, "summary": summary,
                                                   "records": rows,
                                                   "cells": {f"{k[0]}__shape_{k[1]}": [r["id"] for r in v]
                                                             for k, v in cells.items()},
                                                   "shape_distance_matrix": d.tolist()}, indent=2,
                                                  ensure_ascii=False) + "\n")
    make_report(out, rows, summary)
    print(out / "index.html")
    print(summary)


if __name__ == "__main__":
    main()
