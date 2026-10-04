#!/usr/bin/env python3
"""Propose an image-space QD batch against a verified final-mesh archive.

The score is intentionally a proposal criterion, not a surrogate prediction of
FEA quality.  New images must still pass dense/sparse generation and final FEA.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_semantic_qd_geometry import equal_area, jaccard

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22"
ARCHIVE = STUDY / "realization_aware_qd_archive_v2/archive.json"
OUT = STUDY / "realization_aware_qd_archive_v2/next_image_batch"


def image_mask(path: Path, resolution: int = 96) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((resolution, resolution), Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    # Red/green BC rings are interfaces, not structural design material.
    neutral = (rgb.max(2) - rgb.min(2)) < 0.24
    return (rgb.mean(2) < 0.70) & neutral


def shape_distance(a: np.ndarray, b: np.ndarray, common_area: int) -> float:
    allowed = np.ones_like(a, dtype=bool)
    return jaccard(equal_area(a, allowed, common_area),
                   equal_area(b, allowed, common_area))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--batch-size", type=int, default=6)
    args = ap.parse_args()
    verified = json.loads(args.archive.read_text())["records"]
    pools = list((STUDY / "shared_image_pool").rglob("candidate_images.json"))
    candidates = {r["id"]: r for p in pools for r in json.loads(p.read_text())["records"]}
    known_ids = {r["id"] for r in verified}
    observed = []
    for row in verified:
        path = row.get("input_image") or candidates.get(row["id"], {}).get("image_512")
        if path and Path(path).is_file():
            observed.append({"id": row["id"], "intent": row["semantic_niche_target"],
                             "geometry_gate_pass": row.get("geometry_gate_pass"),
                             "image": Path(path), "mask": image_mask(Path(path))})
    anchors = [a for a in observed if a["geometry_gate_pass"] is True]
    known_images = {a["image"].resolve() for a in observed}
    if not anchors:
        raise ValueError("No geometry-gate-valid archive candidates; validate final meshes first")
    all_areas = [int(x["mask"].sum()) for x in anchors]
    common_area = int(np.median(all_areas))
    scored = []
    for row in candidates.values():
        if row["id"] in known_ids:
            continue
        path = Path(row["image_512"])
        if not path.is_file() or path.resolve() in known_images:
            continue
        mask = image_mask(path)
        same = [a for a in anchors if a["intent"] == row["semantic_niche_target"]]
        if not same:
            continue
        distances = [(shape_distance(mask, a["mask"], common_area), a["id"]) for a in same]
        nearest_dist, nearest_id = min(distances)
        scored.append({"id": row["id"], "semantic_niche_target": row["semantic_niche_target"],
                       "mass_level_target": row["mass_level_target"],
                       "image_512": str(path), "model_input": row["model_input"],
                       "image_novelty": float(nearest_dist), "nearest_verified_image": nearest_id})
    scored.sort(key=lambda r: (-r["image_novelty"], r["id"]))
    selected, used = [], set()
    for row in scored:
        if row["semantic_niche_target"] in used:
            continue
        selected.append(row)
        used.add(row["semantic_niche_target"])
        if len(selected) >= args.batch_size:
            break
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    payload = {"method": "training-free gradient-free image novelty proposal",
               "archive": str(args.archive.resolve()),
               "image_descriptor": "96x96 neutral dark-body mask, colored BC excluded, equal-area Jaccard",
               "same_intent_only": True, "area_pixels": common_area,
               "valid_anchor_count": len(anchors), "observed_image_count": len(observed),
               "quality_predicted": False, "requires": "dense, sparse, independent final-mesh FEA",
               "ranked": scored, "selected": selected}
    (out / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    def card(row: dict) -> str:
        image = "/" + str(Path(row["image_512"]).relative_to(ROOT))
        return f'''<article><img src="{html.escape(image)}"><h3>{html.escape(row['id'])}</h3>
        <p>{html.escape(row['semantic_niche_target'])} · image novelty {row['image_novelty']:.3f}<br>
        nearest verified input: {html.escape(row['nearest_verified_image'])}</p></article>'''
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QD next image proposals</title><style>body{{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 16px;background:#f4f6f8;color:#17212b}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}}article{{background:#fff;border:1px solid #ddd;border-radius:10px;overflow:hidden}}img{{width:100%;display:block}}h3,p{{margin:10px}}</style><h1>다음 QD 이미지 후보</h1><p>최종 FEA로 검증된 archive를 기준으로 같은 구조 의도 안에서 이미지 형태가 먼 후보를 하나씩 선택했다. 이 순위는 형상 탐색 제안이며, 3D 실현이나 성능을 보장하지 않는다.</p><div class="grid">{''.join(map(card,selected))}</div><p><a href="selection.json">전체 순위 JSON</a></p></html>'''
    (out / "index.html").write_text(page, encoding="utf-8")
    print(out / "index.html")
    for row in selected:
        print(row["id"], row["semantic_niche_target"], f"{row['image_novelty']:.3f}")


if __name__ == "__main__":
    main()
