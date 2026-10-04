"""Install new transparent chair figure assets with opaque provenance copies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

ROOT = Path('/home/goya/SDL/3d_qd')
SRC = ROOT / 'experiments/chair/sofa_style_2026-09-28/text_reasoned_front_axes_2026-10-03'
OUT = ROOT / 'CVPR/fig/figure/source_images'
GENERATED = Path('/home/goya/.codex/generated_images/01a098b3-02c6-72d0-9729-ce061f818654')
ITEMS = [
    ('06_narrow_open.png', SRC / 'pair_01/narrow_open/input.png', 'exec-8b337fdb-1a8f-4bcc-814a-9f54ca1ba7e4.png'),
    ('07_wide_open.png', SRC / 'pair_01/wide_open/input.png', 'exec-3b6d72a4-24c0-48fb-9ed7-e33c7061332d.png'),
    ('08_narrow_closed.png', SRC / 'pair_02/narrow_closed/input.png', 'exec-4633fef0-424c-4d47-ae55-fc7175ba2ae3.png'),
    ('09_wide_closed.png', SRC / 'pair_02/wide_closed/input.png', 'exec-43be1a14-cc11-4935-8c2b-580dc512b8c7.png'),
    ('10_wide_closed_v2.png', SRC / 'followups/wide_closed_v2.png', 'exec-27fd5078-be15-4936-b590-1c48b4ad6f7d.png'),
    ('11_tapered_triangular_perforations.png', SRC / 'followups/tapered_triangular_perforations.png', 'exec-665e2fc5-1b0b-478d-9080-18d8b301d66c.png'),
    ('12_flared_oval_perforations.png', SRC / 'followups/flared_oval_perforations.png', 'exec-f61db0fe-bcd5-4532-a5ee-4c8ded9a3db6.png'),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    originals = OUT / 'opaque_originals'
    originals.mkdir(exist_ok=True)
    rows = []
    for name, original, generated in ITEMS:
        transparent = GENERATED / generated
        out = OUT / name
        opaque_copy = originals / name
        assert original.is_file() and transparent.is_file()
        shutil.copy2(transparent, out)
        shutil.copy2(original, opaque_copy)
        with Image.open(out) as im:
            rgba = np.asarray(im.convert('RGBA'))
        with Image.open(original) as im:
            rgb = np.asarray(im.convert('RGB'))
        assert rgba.shape[:2] == rgb.shape[:2]
        alpha = rgba[:, :, 3]
        assert np.any(alpha == 0) and np.any(alpha > 250)
        old_mask = rgb.mean(axis=2) < 190
        new_mask = alpha > 127
        iou = float(np.logical_and(old_mask, new_mask).sum() /
                    np.logical_or(old_mask, new_mask).sum())
        item = {'name': name, 'transparent': str(out), 'opaque_original': str(opaque_copy),
                'experiment_original': str(original), 'imagegen_cutout': str(transparent),
                'size': list(rgba.shape[:2][::-1]),
                'fully_transparent_fraction': float(np.mean(alpha == 0)),
                'opaque_silhouette_iou': iou,
                'sha256': sha256(out)}
        rows.append(item)
        print(name, 'alpha_zero', round(item['fully_transparent_fraction'], 3),
              'IoU', round(iou, 3))
    (OUT / 'new_chair_assets_manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    with (OUT / 'README.md').open('a') as stream:
        stream.write('\n## New text-reasoned chair image assets (2026-10-03)\n\n')
        stream.write('Files 06–12 are RGBA transparent cutouts for paper figures. Their opaque conditioning images are copied into `opaque_originals/` and remain the authoritative experimental inputs. The cutouts were created by imagegen and should not replace those inputs. Exact paths, SHA-256 values, alpha fractions, and silhouette overlap are in `new_chair_assets_manifest.json`.\n')


if __name__ == '__main__':
    main()
