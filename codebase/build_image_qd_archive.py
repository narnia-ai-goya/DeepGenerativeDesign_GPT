#!/usr/bin/env python3
"""Build the initial image-space QD archive for bracket-to-3D realization."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage
from skimage.morphology import skeletonize


ROOT = Path("/home/goya/SDL/3d_qd")
OUT = ROOT / "experiments/bracket/image_qd_2026-09-22"
RAW = OUT / "images_raw"
NORMAL = OUT / "images_normalized"
INPUTS = OUT / "inputs"
REFERENCE = ROOT / "experiments/bracket/direct3ds2_lowres_2026-09-15/lr162/input/그림1.png"


def object_mask(image: Image.Image, threshold: int = 248) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    mask = rgb.mean(2) < threshold
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if n:
        sizes = np.bincount(labels.ravel()); sizes[0] = 0
        mask = labels == sizes.argmax()
    return ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=1)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    yy, xx = np.where(mask)
    return int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)


def normalize_image(path: Path, target_bbox: tuple[int, int, int, int]) -> tuple[Path, Path]:
    image = Image.open(path).convert("RGB")
    source_box = bbox(object_mask(image))
    crop = image.crop(source_box)
    tw, th = target_bbox[2] - target_bbox[0], target_bbox[3] - target_bbox[1]
    scale = min(tw / crop.width, th / crop.height)
    size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    crop = crop.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (512, 512), "white")
    x = (target_bbox[0] + target_bbox[2] - size[0]) // 2
    y = (target_bbox[1] + target_bbox[3] - size[1]) // 2
    canvas.paste(crop, (x, y))
    native = canvas.resize((162, 162), Image.Resampling.LANCZOS)
    lowres = native.resize((512, 512), Image.Resampling.LANCZOS)
    normal_path = NORMAL / f"{path.stem}_162up.png"
    input_path = INPUTS / path.stem / "그림1.png"
    normal_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    lowres.save(normal_path); lowres.save(input_path)
    native.save(NORMAL / f"{path.stem}_native162.png")
    return normal_path, input_path


def holes(mask: np.ndarray) -> tuple[int, float]:
    filled = ndimage.binary_fill_holes(mask)
    void = filled & ~mask
    labels, n = ndimage.label(void)
    sizes = np.bincount(labels.ravel()) if n else np.zeros(1, dtype=int)
    keep = [i for i in range(1, n + 1) if sizes[i] >= 12]
    area = sum(int(sizes[i]) for i in keep)
    return len(keep), area / max(1, int(filled.sum()))


def branches(mask: np.ndarray) -> int:
    skel = skeletonize(mask)
    degree = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)) - skel
    _, n = ndimage.label(skel & (degree >= 3), structure=np.ones((3, 3), dtype=np.uint8))
    return int(n)


def metrics(path: Path, reference_outer: np.ndarray) -> dict:
    image = Image.open(path).convert("RGB").resize((162, 162), Image.Resampling.LANCZOS)
    mask = object_mask(image)
    outer = ndimage.binary_fill_holes(mask)
    union = (outer | reference_outer).sum()
    outline_deviation = 1.0 - (outer & reference_outer).sum() / max(1, union)
    hole_count, void_fraction = holes(mask)
    dist = ndimage.distance_transform_edt(mask)
    center_thickness = 2 * dist[skeletonize(mask)]
    return {
        "outline_deviation": float(outline_deviation),
        "load_path_branch_clusters": branches(mask),
        "hole_count": int(hole_count),
        "void_fraction": float(void_fraction),
        "material_fraction": float(mask.mean()),
        "skeleton_thickness_p10_px162": float(np.quantile(center_thickness, .1)),
        "connected_fraction": float(mask.sum() / max(1, object_mask(image).sum())),
    }


def main() -> None:
    for directory in (NORMAL, INPUTS): directory.mkdir(parents=True, exist_ok=True)
    ref = Image.open(REFERENCE).convert("RGB")
    target_bbox = bbox(object_mask(ref))
    reference_outer = ndimage.binary_fill_holes(object_mask(ref.resize((162, 162))))
    records = []
    for raw in sorted(RAW.glob("*.png")):
        normalized, input_path = normalize_image(raw, target_bbox)
        row = {"archetype": raw.stem, "raw": str(raw.resolve()),
               "normalized": str(normalized.resolve()), "input": str(input_path.resolve())}
        row.update(metrics(normalized, reference_outer))
        # Initial archive cell. BOP-Elites will later model the continuous values rather than this
        # coarse display bin.
        row["cell"] = [int(np.digitize(row["void_fraction"], [.30, .33, .36])),
                       int(np.digitize(row["load_path_branch_clusters"], [18, 21, 24]))]
        records.append(row)
    (OUT / "image_archive.json").write_text(json.dumps({
        "method": "gradient-free initial design for realization-aware BOP-Elites",
        "descriptors": ["void_fraction", "load_path_branch_clusters"],
        "quality_screen": ["connectedness", "minimum skeleton thickness", "BC preservation"],
        "reference": str(REFERENCE.resolve()), "records": records}, indent=2) + "\n")

    thumbs = []
    for r in records:
        im = Image.open(r["normalized"]).convert("RGB").resize((360, 360))
        thumbs.append((r, im))
    sheet = Image.new("RGB", (4 * 360, 2 * 400), "white"); draw = ImageDraw.Draw(sheet)
    for i, (r, im) in enumerate(thumbs):
        x, y = (i % 4) * 360, (i // 4) * 400
        sheet.paste(im, (x, y)); draw.text((x + 8, y + 363),
            f"{r['archetype']}  outline={r['outline_deviation']:.2f} branch={r['load_path_branch_clusters']}", fill="black")
    sheet.save(OUT / "image_qd_contact_sheet.png")
    cards = ''.join(f"<article><img src='images_normalized/{Path(r['normalized']).name}'><h2>{r['archetype']}</h2>"
                    f"<p>outline {r['outline_deviation']:.3f} · branches {r['load_path_branch_clusters']} · "
                    f"holes {r['hole_count']} · void {r['void_fraction']:.3f}</p><code>{r['input']}</code></article>" for r in records)
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Image-space QD initial archive</title><style>body{{font:15px system-ui;background:#f6f7f9;color:#17212b;margin:30px}}main{{max-width:1450px;margin:auto}}.note{{background:#eaf6f1;border-left:5px solid #168065;padding:14px}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}article{{background:white;border:1px solid #d7dde2;border-radius:10px;overflow:hidden;padding-bottom:12px}}img{{width:100%}}h2,p,code{{margin:9px 12px}}code{{display:block;font-size:10px;overflow-wrap:anywhere}}</style><main><h1>Image-space QD initial archive</h1><p class="note">8 structured prompt archetypes. GPT image is treated as a black box; descriptors are measured from generated pixels. These are the initial observations for gradient-free BOP-Elites.</p><img src="image_qd_contact_sheet.png" style="width:100%"><div class="grid">{cards}</div></main></html>'''
    (OUT / "index.html").write_text(page)
    print(OUT / "index.html")


if __name__ == "__main__": main()
