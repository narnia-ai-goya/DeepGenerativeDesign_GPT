#!/usr/bin/env python3
"""Register generated sofa-style three-view sheets to the chair BC camera frame."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from make_chair_domain import ROOT


BASE = ROOT / "experiments/chair/sofa_style_2026-09-28"
NAMES = ("open_arm", "solid_side")
VIEWS = (
    ("v00_front_lo", (128, 55, 384, 502)),
    ("v02_right_lo", (128, 55, 384, 503)),
    ("v_top", (128, 121, 384, 394)),
)


def register(name: str) -> None:
    case = BASE / name
    sheet = Image.open(case / "generated_sheet.png").convert("RGB")
    panel_width = sheet.width / 3
    for directory in ("input", "input_lr162"):
        (case / directory).mkdir(exist_ok=True)
    contact = Image.new("RGB", (512 * 3, 548), "white")
    draw = ImageDraw.Draw(contact)
    manifest = {}
    for i, (view, dst_bbox) in enumerate(VIEWS):
        x0, x1 = round(i * panel_width), round((i + 1) * panel_width)
        panel = sheet.crop((x0, 0, x1, sheet.height))
        mask = np.asarray(panel).min(axis=2) < 210
        yy, xx = np.nonzero(mask)
        if len(xx) < 100:
            raise ValueError(f"{name}/{view}: empty image mask")
        src_bbox = (int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1))
        content = panel.crop(src_bbox)
        dx0, dy0, dx1, dy1 = dst_bbox
        registered = Image.new("RGB", (512, 512), "white")
        registered.paste(content.resize((dx1 - dx0, dy1 - dy0), Image.Resampling.LANCZOS),
                         (dx0, dy0))
        registered.save(case / "input" / f"{view}.png")
        registered.resize((162, 162), Image.Resampling.LANCZOS).save(
            case / "input_lr162" / f"{view}.png")
        contact.paste(registered, (i * 512, 36))
        draw.text((i * 512 + 12, 10), view, fill="#263139")
        manifest[view] = {
            "source_panel": [x0, 0, x1, sheet.height],
            "source_bbox": src_bbox,
            "target_bbox": dst_bbox,
            "registered_512": str(case / "input" / f"{view}.png"),
            "registered_162": str(case / "input_lr162" / f"{view}.png"),
        }
    contact.save(case / "input_contact.png")
    (case / "registration.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(case / "input_contact.png")


if __name__ == "__main__":
    for name in NAMES:
        register(name)
