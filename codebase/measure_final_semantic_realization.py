#!/usr/bin/env python3
"""Measure input-to-final semantic realization against each candidate's own text intent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor


MODEL = "openai/clip-vit-large-patch14"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path); parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()
    evaluation = args.experiment / "shared_evaluations" / f"round_{args.round:02d}"
    result_path = evaluation / "verified_results.json"
    records = json.loads(result_path.read_text())
    protocol = json.loads((args.experiment / "protocol_snapshot.json").read_text())
    anchors = protocol["archive"]["semantic_niches"]
    processor = CLIPProcessor.from_pretrained(MODEL, local_files_only=True)
    model = CLIPModel.from_pretrained(MODEL, local_files_only=True).eval()
    for row in records:
        text = anchors[row["semantic_niche_target"]]
        geometry_dir = Path(row["geometry_metrics"]).parent
        view_paths = sorted(geometry_dir.glob("v*.png"))
        images = [Image.open(row["image_512"]).convert("RGB")]
        images.extend(Image.open(path).convert("RGB") for path in view_paths)
        with torch.inference_mode():
            image_inputs = processor(images=images, return_tensors="pt")
            text_inputs = processor(text=[text], return_tensors="pt", padding=True)
            image_features = model.get_image_features(**image_inputs).float()
            text_features = model.get_text_features(**text_inputs).float()
            image_features /= image_features.norm(dim=1, keepdim=True).clamp_min(1e-12)
            text_features /= text_features.norm(dim=1, keepdim=True).clamp_min(1e-12)
            scores = (image_features @ text_features.T).squeeze(1).cpu().tolist()
        row["semantic_realization"] = {
            "encoder": MODEL, "target_text": text,
            "input_similarity": scores[0], "final_multiview_mean_similarity": sum(scores[1:]) / len(scores[1:]),
            "final_multiview_min_similarity": min(scores[1:]),
            "input_to_final_gap": scores[0] - sum(scores[1:]) / len(scores[1:]),
            "views": [str(path.resolve()) for path in view_paths],
        }
    result_path.write_text(json.dumps(records, indent=2) + "\n")
    print(result_path.resolve())
    for row in records:
        sem = row["semantic_realization"]
        print(row["id"], f"input={sem['input_similarity']:.4f}",
              f"final={sem['final_multiview_mean_similarity']:.4f}",
              f"gap={sem['input_to_final_gap']:.4f}")


if __name__ == "__main__": main()
