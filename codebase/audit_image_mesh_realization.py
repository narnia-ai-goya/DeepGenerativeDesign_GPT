#!/usr/bin/env python3
"""Compare image proposal novelty with realized final-mesh shape distance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from select_realization_aware_qd_images import STUDY, image_mask, shape_distance


def main() -> None:
    archive = STUDY / "realization_aware_qd_archive_v2/archive.json"
    data = json.loads(archive.read_text())
    rows = data["records"]
    shape = np.asarray(data["shape_distance_matrix"])
    candidates = {r["id"]: r for file in (STUDY / "shared_image_pool").rglob("candidate_images.json")
                  for r in json.loads(file.read_text())["records"]}
    masks = []
    for row in rows:
        image = row.get("input_image") or candidates.get(row["id"], {}).get("image_512")
        if image is None and row["id"] == "r5_longitudinal_spine__medium":
            image = str(STUDY / "shared_image_pool/round_05/novel_semantic/images_512/r5_longitudinal_spine__medium.png")
        if image is None:
            raise FileNotFoundError(f"No input image for {row['id']}")
        masks.append(image_mask(Path(image)))
    area = int(np.median([mask.sum() for mask in masks]))
    image = np.zeros_like(shape)
    pairs = []
    for i in range(len(rows)):
        for j in range(i):
            image[i, j] = image[j, i] = shape_distance(masks[i], masks[j], area)
            pairs.append((i, j))
    def rho(subset: list[tuple[int, int]]) -> float:
        return float(spearmanr([image[i, j] for i, j in subset],
                               [shape[i, j] for i, j in subset]).statistic)
    same = [(i, j) for i, j in pairs if rows[i]["semantic_niche_target"] == rows[j]["semantic_niche_target"]]
    different = [(i, j) for i, j in pairs if (i, j) not in same]
    nearest_matches = 0
    for i in range(len(rows)):
        valid = np.arange(len(rows)) != i
        image_nearest = int(np.argmin(np.where(valid, image[i], np.inf)))
        shape_nearest = int(np.argmin(np.where(valid, shape[i], np.inf)))
        nearest_matches += image_nearest == shape_nearest
    result = {
        "archive": str(archive), "n_records": len(rows), "n_pairs": len(pairs),
        "image_descriptor": "neutral dark-body 96x96 equal-area Jaccard",
        "shape_descriptor": data["protocol"]["realized_shape_descriptor"],
        "all_pairs_spearman": rho(pairs), "same_intent_pairs": len(same),
        "same_intent_spearman": rho(same), "different_intent_spearman": rho(different),
        "nearest_neighbor_matches": nearest_matches,
        "qualification": "Descriptive audit only: legacy cases used mixed generator settings; pairwise distances are not independent. Do not interpret as a fixed-protocol transfer estimate.",
    }
    out = archive.parent / "image_mesh_realization_audit.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(out)
    print(result)


if __name__ == "__main__":
    main()
