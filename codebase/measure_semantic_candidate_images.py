#!/usr/bin/env python3
"""Measure prompt-conditioned candidate images against frozen semantic text anchors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor


MODEL = "openai/clip-vit-large-patch14"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()
    protocol = json.loads((args.experiment / "protocol_snapshot.json").read_text())
    pool_dir = args.experiment / "shared_image_pool" / f"round_{args.round:02d}"
    records = json.loads((pool_dir / "candidate_images.json").read_text())["records"]
    niches = protocol["archive"]["semantic_niches"]
    names, texts = list(niches), list(niches.values())
    processor = CLIPProcessor.from_pretrained(MODEL, local_files_only=True)
    model = CLIPModel.from_pretrained(MODEL, local_files_only=True).eval()
    images = [Image.open(row["image_512"]).convert("RGB") for row in records]
    with torch.inference_mode():
        image_inputs = processor(images=images, return_tensors="pt")
        text_inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        image_features = model.get_image_features(**image_inputs).float()
        text_features = model.get_text_features(**text_inputs).float()
        image_features /= image_features.norm(dim=1, keepdim=True).clamp_min(1e-12)
        text_features /= text_features.norm(dim=1, keepdim=True).clamp_min(1e-12)
        similarity = (image_features @ text_features.T).cpu().numpy()
    enriched = []
    for row, scores in zip(records, similarity):
        order = np.argsort(scores)[::-1]
        enriched.append({**row, "realized_image_niche": names[int(order[0])],
                         "semantic_margin": float(scores[order[0]] - scores[order[1]]),
                         "semantic_scores": {name: float(value) for name, value in zip(names, scores)}})
    destination = pool_dir / "candidate_image_semantics.json"
    destination.write_text(json.dumps({"encoder": MODEL, "records": enriched}, indent=2) + "\n")
    print(destination.resolve())
    for row in enriched:
        print(row["id"], "->", row["realized_image_niche"], f"margin={row['semantic_margin']:.4f}")


if __name__ == "__main__":
    main()
