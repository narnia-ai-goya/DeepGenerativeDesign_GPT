#!/usr/bin/env python3
"""Build a training-free semantic QD map from canonical shape-intent text."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer


ROOT = Path("/home/goya/SDL/3d_qd")
OUT = ROOT / "reports/semantic_text_qd"
MODEL = "openai/clip-vit-large-patch14"

INTENTS = {
    "arch_tie": "a broad structural arch connected by a continuous lower tension tie",
    "asymmetric_diagonal": "an asymmetric structural web dominated by one sweeping diagonal brace",
    "fan_rib": "a radial fan of structural ribs spreading from the load interfaces",
    "organic_branching": "an organic branching structural network with curved merging members",
    "triangular_truss": "a coarse structural truss composed of large triangular cells",
    "twin_spine": "two parallel longitudinal structural spines joined by transverse bridges",
    "x_brace": "a structural frame dominated by a central X-shaped crossed brace",
    "y_frame": "a structural frame organized around a central Y-shaped branching member",
    "reinforced_double_arch": "two nested load-bearing arches with a thick cross-tie and redundant diagonal braces",
    "offset_double_y": "two offset Y-shaped structural paths sharing broad reinforced junctions",
    "diamond_ring_web": "a closed diamond ring web crossed by two continuous structural paths",
    "redundant_twin_fan": "paired fan-shaped structural webs connected by redundant cross-members",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = CLIPTextModelWithProjection.from_pretrained(MODEL, local_files_only=True)
    model.eval()
    names = list(INTENTS)
    tokens = tokenizer([INTENTS[n] for n in names], padding=True, truncation=True,
                       max_length=77, return_tensors="pt")
    with torch.inference_mode():
        features = model(**tokens).text_embeds.float().cpu().numpy()
    features /= np.linalg.norm(features, axis=1, keepdims=True).clip(1e-12)
    distance = np.clip(1.0 - features @ features.T, 0.0, 2.0)
    # Deterministic classical MDS for display only. QD uses the full embedding and cosine distance.
    n = len(distance); center = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * center @ (distance ** 2) @ center
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:2]
    coords = vectors[:, order] * np.sqrt(np.maximum(values[order], 0.0))

    group = np.array(["initial" if i < 8 else "BO iteration 01" for i in range(len(names))])
    colors = {"initial": "#2f78a8", "BO iteration 01": "#d57932"}
    fig, ax = plt.subplots(figsize=(12.8, 8.4), dpi=180)
    fig.patch.set_facecolor("#f5f8fa"); ax.set_facecolor("#f5f8fa")
    for g in colors:
        idx = np.where(group == g)[0]
        ax.scatter(coords[idx, 0], coords[idx, 1], s=115, c=colors[g], label=g,
                   edgecolors="white", linewidths=1.7, zorder=3)
    offsets = {
        "arch_tie": (7, 8), "asymmetric_diagonal": (7, -15), "fan_rib": (7, 8),
        "organic_branching": (7, 8), "triangular_truss": (7, 8), "twin_spine": (7, 8),
        "x_brace": (7, 8), "y_frame": (7, 8), "reinforced_double_arch": (7, 8),
        "offset_double_y": (7, -15), "diamond_ring_web": (7, 8),
        "redundant_twin_fan": (7, -15),
    }
    for i, name in enumerate(names):
        ax.annotate(name, coords[i], xytext=offsets[name], textcoords="offset points",
                    fontsize=9, color="#243f53")
    ax.set_title("Semantic design-intent space", loc="left", fontsize=20,
                 fontweight="bold", color="#173247", pad=18)
    ax.text(0, 1.015,
            "Cosine distance of frozen CLIP text embeddings; classical MDS is visualization only",
            transform=ax.transAxes, fontsize=10.5, color="#587184")
    ax.set_xlabel("MDS dimension 1", color="#587184")
    ax.set_ylabel("MDS dimension 2", color="#587184")
    ax.grid(color="#d7e1e8", linewidth=.8, alpha=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#aebfca")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "semantic_intent_map.png", bbox_inches="tight")
    fig.savefig(OUT / "semantic_intent_map.svg", bbox_inches="tight")
    payload = {
        "encoder": MODEL,
        "distance": "cosine distance on L2-normalized frozen text embeddings",
        "optimization_space": "full embedding; MDS coordinates are visualization only",
        "names": names, "intents": INTENTS,
        "distance_matrix": distance.tolist(), "mds_coordinates": coords.tolist(),
    }
    (OUT / "semantic_intent_map.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(OUT / "semantic_intent_map.png")


if __name__ == "__main__":
    main()
