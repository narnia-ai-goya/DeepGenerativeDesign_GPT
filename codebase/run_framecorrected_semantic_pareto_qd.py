"""Build a verified semantic Pareto-QD archive from final bracket meshes.

The archive descriptor is a frozen text embedding of the designer's structural
intent prompt.  Quality is deliberately *not* collapsed into a raw compliance
score: every semantic niche retains its non-dominated (compliance, volume)
front after independent FEA of the delivered mesh.

This is the archive/selection stage for the next image-conditioned QD loop.
It only consumes frame-corrected FEA results, so a silent generator-grid to
physical-domain mismatch cannot affect archive quality.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def embed_texts(texts: list[str]) -> np.ndarray:
    """Frozen mean-pooled MiniLM embeddings; no learning or API calls."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            tokens = tokenizer(texts[start:start + 16], padding=True, truncation=True,
                               max_length=256, return_tensors="pt")
            state = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(state.dtype)
            value = (state * mask).sum(1) / mask.sum(1).clamp_min(1)
            value = value / value.norm(dim=1, keepdim=True).clamp_min(1e-12)
            batches.append(value.cpu().numpy())
    return np.vstack(batches)


def pca_2d(x: np.ndarray) -> tuple[np.ndarray, dict]:
    mean = x.mean(0)
    _, singular, vt = np.linalg.svd(x - mean, full_matrices=False)
    basis = vt[:2]
    return (x - mean) @ basis.T, {"mean": mean.tolist(), "basis": basis.tolist(),
                                 "singular_values": singular.tolist()}


def pareto_front(rows: list[dict]) -> list[dict]:
    kept = []
    for row in rows:
        dominated = any(
            other is not row and other["compliance_J"] <= row["compliance_J"] and
            other["volume_mm3"] <= row["volume_mm3"] and
            (other["compliance_J"] < row["compliance_J"] or
             other["volume_mm3"] < row["volume_mm3"])
            for other in rows)
        if not dominated:
            kept.append(row)
    return sorted(kept, key=lambda r: (r["volume_mm3"], r["compliance_J"]))


def nearest_centroid(z: np.ndarray, centers: np.ndarray) -> int:
    return int(np.argmin(((centers - z) ** 2).sum(1)))


def make_report(out: Path, records: list[dict], cells: dict[int, list[dict]], protocol: dict):
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    cards = []
    for record in records:
        src = Path(record["comparison_render"])
        image = assets / f"{record['id']}.png"
        shutil.copy2(src, image)
        badge = "Pareto elite" if record["pareto_elite"] else "dominated"
        novelty = record.get("shape_novelty")
        novelty_label = (f"<br>3D shape novelty = {novelty['distance']:.3f} "
                         f"({'pass' if novelty['passes_novelty_gate'] else 'reject'})"
                         if novelty else "")
        cards.append(f'''<article class="{'elite' if record['pareto_elite'] else 'dominated'}">
          <img src="assets/{image.name}"><h3>{html.escape(record['id'])}</h3>
          <p><b>{html.escape(record['semantic_niche_target'])}</b> · CVT cell {record['cell']} · {badge}<br>
          C = {record['compliance_J']:.5g} J · V = {record['volume_mm3']:,.0f} mm³<br>
          stress = {record['vm_max_MPa']:.1f} MPa{novelty_label}</p>
        </article>''')
    cell_rows = "".join(
        f"<tr><td>{idx}</td><td>{len(rows)}</td><td>{', '.join(r['id'] for r in pareto_front(rows))}</td></tr>"
        for idx, rows in sorted(cells.items()))
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Frame-corrected semantic Pareto-QD</title>
    <style>body{{font:15px system-ui;max-width:1450px;margin:30px auto;background:#f4f6f8;color:#17212b;line-height:1.5}}.note{{background:#fff;border-left:5px solid #277da1;padding:14px 17px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}}article{{background:white;border:1px solid #d8e0e6;border-radius:10px;overflow:hidden}}article.elite{{border:3px solid #27835c}}article.dominated{{opacity:.65}}img{{width:100%;display:block}}h3,p{{margin:11px 14px}}table{{border-collapse:collapse;background:#fff;width:100%;margin:18px 0}}td,th{{border:1px solid #d8e0e6;padding:8px;text-align:left}}</style>
    <h1>Frame-corrected semantic Pareto-QD archive</h1>
    <div class="note"><b>Descriptor.</b> Designer prompt → frozen {MODEL} embedding → PCA-2D → nearest CVT semantic niche.<br>
    <b>Quality.</b> Independent FEA compliance × final volume as a two-objective Pareto front. Raw compliance scalarization is not used.<br>
    <b>Verification.</b> Only frame-corrected FEA runs with delivered final meshes are included.</div>
    <h2>Occupied niches</h2><table><tr><th>cell</th><th>candidates</th><th>Pareto elite(s)</th></tr>{cell_rows}</table>
    <h2>Verified candidates</h2><div class="grid">{''.join(cards)}</div></html>'''
    (out / "index.html").write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--niches", type=int, default=9)
    ap.add_argument("--rounds", default="2,3", help="Comma-separated evaluated rounds to archive.")
    args = ap.parse_args()
    study = args.study.resolve()
    out = (args.out or study / "framecorrected_semantic_pareto_qd").resolve()
    # Keep round selection explicit: round 2 has an in-loop FEA re-run under
    # fea_on_bc6_framefix, while later QD rounds are quality-evaluated only
    # after their standard sparse/post pipeline.
    rounds = [int(x) for x in args.rounds.split(",")]
    final_rows = []
    for round_index in rounds:
        for row in load_json(study / f"shared_evaluations/round_{round_index:02d}/final_results.json"):
            final_rows.append({**row, "source_round": round_index})
    prompts = load_json(study / "prompt_bank.json")["prompts"]
    prompt_for = {(p["semantic_niche_target"], p["mass_level_target"]): p["prompt"] for p in prompts}
    bank_text = [p["prompt"] for p in prompts]
    all_embeddings = embed_texts(bank_text)
    embedding_for = {text: emb for text, emb in zip(bank_text, all_embeddings)}
    bank_z, pca = pca_2d(all_embeddings)

    # CVT-like fixed centers: farthest-point selection over the entire designer prompt bank.
    chosen = [int(np.argmax(np.linalg.norm(bank_z - bank_z.mean(0), axis=1)))]
    while len(chosen) < min(args.niches, len(bank_z)):
        d = np.min(((bank_z[:, None, :] - bank_z[chosen][None, :, :]) ** 2).sum(2), axis=1)
        chosen.append(int(np.argmax(d)))
    centers = bank_z[chosen]

    records = []
    for row in final_rows:
        text = row.get("prompt") or prompt_for[(row["semantic_niche_target"], row["mass_level_target"])]
        # The projection and niche centers stay frozen to the original prompt
        # bank. New designer prompts are projected into that same space.
        embedding = embedding_for.get(text)
        if embedding is None:
            embedding = embed_texts([text])[0]
        z = (embedding - np.asarray(pca["mean"])) @ np.asarray(pca["basis"]).T
        case = study / "shared_evaluations" / f"round_{row['source_round']:02d}" / row["id"]
        if row["source_round"] == 2:
            run = case / "fea_on_bc6_framefix" / "post"
            fea_path = run / "fea_independent/fea_tet_summary.json"
            mesh_path = run / "final.obj"
            comparison = study / "report_fea_on_framefix/assets" / f"{row['id'].replace('r2_outline_', '').replace('__medium', '')}_fea_on_framefix.png"
        else:
            run = case
            fea_path = run / "fea_independent/fea_tet_summary.json"
            mesh_path = run / "gen/final.obj"
            comparison = Path(row["final_preview"])
        fea = load_json(fea_path)
        import trimesh
        mesh = trimesh.load_mesh(mesh_path, force="mesh", process=False)
        records.append({"id": row["id"], "semantic_niche_target": row["semantic_niche_target"],
                        "mass_level_target": row["mass_level_target"], "prompt": text,
                        "text_embedding_sha256": hashlib.sha256(embedding.tobytes()).hexdigest(),
                        "descriptor_pca2": z.tolist(), "cell": nearest_centroid(z, centers),
                        "compliance_J": float(fea["compliance"]),
                        "volume_mm3": abs(float(mesh.volume)) * 1e9,
                        "vm_max_MPa": float(fea["vm_max"]) / 1e6,
                        "source_round": row["source_round"],
                        "final_mesh": str(mesh_path.resolve()),
                        "comparison_render": str(comparison.resolve())})
    novelty_path = out / "shape_novelty.json"
    novelty_by_id = {}
    if novelty_path.exists():
        novelty_by_id = {row["id"]: row for row in load_json(novelty_path)["comparisons"]}
    for record in records:
        novelty = novelty_by_id.get(record["id"])
        record["shape_novelty"] = novelty
        record["shape_novelty_pass"] = (novelty is None or novelty["passes_novelty_gate"])
        record["pareto_elite"] = False
    cells: dict[int, list[dict]] = {}
    for record in records:
        if record["shape_novelty_pass"]:
            cells.setdefault(record["cell"], []).append(record)
    for rows in cells.values():
        ids = {r["id"] for r in pareto_front(rows)}
        for row in rows:
            row["pareto_elite"] = row["id"] in ids
    protocol = {"descriptor": {"source": "designer prompt text", "encoder": MODEL,
                                  "projection": "PCA-2D fitted on all 18 prompt-bank entries",
                                  "archive": f"{len(centers)} fixed farthest-point CVT-like centers"},
                "quality": {"objectives": ["minimize independent_FEA_compliance_J",
                                              "minimize final_mesh_volume_mm3"],
                            "selection": "retain non-dominated front within each semantic niche"},
                "feasibility": "final-mesh independent FEA and, where measured, realized-shape novelty gate",
                "centers": centers.tolist(), "pca": pca}
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (out / "archive.json").write_text(json.dumps({"protocol": protocol, "records": records,
                                                    "cells": {str(k): pareto_front(v) for k, v in cells.items()}}, indent=2),
                                      encoding="utf-8")
    make_report(out, records, cells, protocol)
    print(out / "index.html")


if __name__ == "__main__":
    main()
