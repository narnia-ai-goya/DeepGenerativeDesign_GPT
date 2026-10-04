#!/usr/bin/env python3
"""Register image-generated chair orthographic views to the existing chair input frame."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from make_chair_domain import ROOT


BASE = ROOT / 'experiments/chair/image_concepts_2026-09-27/diagonal_braced'
CASE = BASE / 'multiview_registered'
REFERENCE = ROOT / 'experiments/chair/aesthetic_reference_2026-09-27/input'
GENERATED = Path('/home/goya/.codex/generated_images/01a098b3-02c6-72d0-9729-ce061f818654')
PAIRS = {
    'v00_front_lo': 'exec-458f7a63-35a3-4d07-bd3f-c4f83629470d.png',
    'v02_right_lo': 'exec-09c01e14-dd91-4e73-87b6-8fbaf51bbcf8.png',
    'v_top': 'exec-ce6434b6-420c-4fe2-aa4d-02839ac1ecd7.png',
}


def bbox(image: Image.Image) -> tuple[int, int, int, int]:
    pixels = np.asarray(image.convert('RGB'))
    mask = pixels.min(axis=2) < 215
    yy, xx = np.where(mask)
    return int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)


def main() -> None:
    inp = CASE / 'input'
    native = CASE / 'generated_orthographic'
    inp.mkdir(parents=True, exist_ok=True)
    native.mkdir(parents=True, exist_ok=True)
    records = {}
    sheet = Image.new('RGB', (512*3, 548), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (view, filename) in enumerate(PAIRS.items()):
        source = GENERATED / filename
        shutil.copy2(source, native / f'{view}.png')
        original = Image.open(source).convert('RGB')
        target = Image.open(REFERENCE / f'{view}.png').convert('RGB')
        source_box, target_box = bbox(original), bbox(target)
        cut = original.crop(source_box)
        width = target_box[2]-target_box[0]
        height = target_box[3]-target_box[1]
        cut = cut.resize((width, height), Image.Resampling.LANCZOS)
        aligned = Image.new('RGB', (512, 512), 'white')
        aligned.paste(cut, target_box[:2])
        aligned.save(inp / f'{view}.png')
        aligned.resize((162, 162), Image.Resampling.LANCZOS).save(
            CASE / f'{view}_162.png')
        sheet.paste(aligned, (i*512, 36))
        draw.text((i*512+12, 10), view, fill='#20252b')
        records[view] = {'generated_image': str((native / f'{view}.png').resolve()),
                         'source_bbox': source_box, 'target_bbox': target_box,
                         'registered_image': str((inp / f'{view}.png').resolve())}
    sheet.save(CASE / 'input_contact.png')
    (CASE / 'registration.json').write_text(json.dumps(records, indent=2) + '\n')
    print(CASE / 'input_contact.png')


if __name__ == '__main__':
    main()
