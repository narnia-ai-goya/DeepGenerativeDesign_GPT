#!/usr/bin/env python3
"""Slice a fixed 6×3 semantic/mass contact sheet into low-resolution 3D inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


NICHES = ["arch_tie", "diagonal_truss", "radial_fan", "branching",
           "longitudinal_spine", "ring_lattice"]
MASS_LEVELS = ["low", "medium", "high"]


def foreground_bbox(image: Image.Image, threshold: int = 250) -> tuple[int, int, int, int]:
    rgb = np.asarray(image.convert("RGB"))
    mask = np.min(rgb, axis=2) < threshold
    yy, xx = np.where(mask)
    if not len(xx):
        raise ValueError("Empty contact-sheet cell")
    return int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sheet", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--column-ids", default=",".join(NICHES),
                        help="six comma-separated unique column identifiers")
    parser.add_argument("--semantic-targets", default=",".join(NICHES),
                        help="six comma-separated semantic niche targets")
    parser.add_argument("--id-prefix", default="")
    args = parser.parse_args()
    column_ids = args.column_ids.split(",")
    semantic_targets = args.semantic_targets.split(",")
    if len(column_ids) != 6 or len(semantic_targets) != 6 or len(set(column_ids)) != 6:
        raise ValueError("Exactly six unique column IDs and six semantic targets are required")
    image = Image.open(args.sheet).convert("RGB")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for row, mass in enumerate(MASS_LEVELS):
        y0, y1 = round(row * image.height / 3), round((row + 1) * image.height / 3)
        for col, (column_id, niche) in enumerate(zip(column_ids, semantic_targets)):
            x0, x1 = round(col * image.width / 6), round((col + 1) * image.width / 6)
            cell = image.crop((x0, y0, x1, y1))
            box = foreground_bbox(cell)
            crop = cell.crop(box)
            scale = min(472 / crop.width, 472 / crop.height)
            size = (round(crop.width * scale), round(crop.height * scale))
            crop = crop.resize(size, Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (512, 512), "white")
            canvas.paste(crop, ((512 - size[0]) // 2, (512 - size[1]) // 2))
            ident = f"{args.id_prefix}{column_id}__{mass}"
            full = args.output / "images_512" / f"{ident}.png"
            native = args.output / "images_native162" / f"{ident}.png"
            model_input = args.output / "inputs" / ident / "그림1.png"
            full.parent.mkdir(exist_ok=True); native.parent.mkdir(exist_ok=True)
            model_input.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(full)
            low = canvas.resize((162, 162), Image.Resampling.LANCZOS)
            low.save(native)
            low.resize((512, 512), Image.Resampling.LANCZOS).save(model_input)
            records.append({"id": ident, "semantic_niche_target": niche,
                            "mass_level_target": mass, "image_512": str(full.resolve()),
                            "image_native162": str(native.resolve()),
                            "model_input": str(model_input.resolve()),
                            "sheet_cell": [row, col]})
    (args.output / "candidate_images.json").write_text(
        json.dumps({"source_sheet": str(args.sheet.resolve()), "records": records}, indent=2) + "\n")
    print((args.output / "candidate_images.json").resolve())


if __name__ == "__main__":
    main()
