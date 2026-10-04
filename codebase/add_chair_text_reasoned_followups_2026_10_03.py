"""Record measured AI feedback revision and two designed cutout images."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2

from make_chair_domain import ROOT
from build_chair_image_qd_2026_10_03 import normalized, metrics
from prepare_chair_text_reasoned_pilot_2026_10_03 import front_aperture, GATE_REF, GENERATED

OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/text_reasoned_front_axes_2026-10-03'
ROWS = [
    ('wide_closed_v2', 'exec-bec9de98-78b0-4dc2-b782-59ae88f80bb1.png',
     'wide_closed image still had arm opening 0.402; target a narrower opening while preserving backrest width and seat',
     'Edit this exact black-metal chair front-view studio image. Keep the complete seat, seat front lip, four legs, under-seat braces, four feet, camera, scale, white background, and solid wide flared backrest exactly as they are. The previous closed-arm design still left two large white holes next to the backrest. Correct only those two upper side regions: add broad, smoothly connected black-metal side gusset panels joining the backrest sides to the outer arm rails. The left and right white arm-to-back holes should become narrow vertical keyhole slots, each less than half its current width, but still visibly open. Make both sides symmetric. Preserve the central backrest load area and the white gap between backrest and seat. Photorealistic shaded black metal, single front orthographic view, no extra legs or objects, no text.'),
    ('tapered_triangular_perforations', 'exec-6c874f04-0266-4b07-b6d6-4a2cc7897d7a.png',
     'explore intentional perforation topology while preserving central back load patch',
     'Edit this exact front-view black-metal chair studio image. Keep the seat, four legs, four foot pads, under-seat braces, camera, pure white background, narrow tapered backrest silhouette, and solid central backrest load region unchanged. Add exactly two intentional large clean triangular through-holes, one in each upper outer shoulder region of the backrest panel. The holes are symmetric, have rounded fillet corners and thick continuous metal borders, and do not cross the middle central back-load area. Keep the arm openings and seat-to-back white gap. Photorealistic shaded metal, visibly engineered and manufacturable, no extra holes, no text.'),
    ('flared_oval_perforations', 'exec-9cf85bd1-633f-4d52-aa66-affde41eb62f.png',
     'explore alternate perforation topology in a flared back while preserving central back load patch',
     'Edit this exact front-view black-metal chair studio image. Preserve the broad flared backrest outline, the solid central lower backrest load-bearing patch, the seat, four legs, under-seat braces, four foot pads, camera and pure white background. Add exactly two intentional vertically elongated oval through-holes in the upper left and upper right outer portions of the backrest panel, symmetric around a broad untouched solid center spine. Give the holes smooth thick black-metal rims; keep them well away from the seat and the middle back-load patch. Keep the narrow side-arm keyhole slots. Photorealistic shaded black metal, clean engineered cutouts, no extra holes, no text.'),
]


def upper_back_holes(mask):
    white = (~mask).astype('uint8')
    n, _, stats, centers = cv2.connectedComponentsWithStats(white, 8)
    holes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cx, cy = centers[i]
        if (x > 0 and y > 0 and x + w < 512 and y + h < 512 and
                .35 < cx / 512 < .65 and .05 < cy / 512 < .22 and area > 100):
            holes.append({'center_normalized': [float(cx / 512), float(cy / 512)],
                          'area_px': int(area)})
    return holes


def main():
    dest = OUT / 'followups'
    dest.mkdir(exist_ok=True)
    _, ref = normalized(GATE_REF)
    records = []
    for name, file, reasoning, prompt in ROWS:
        path = dest / f'{name}.png'
        shutil.copy2(GENERATED / file, path)
        _, mask = normalized(path)
        measure = metrics(mask, ref)
        measure['front_arm_aperture_fraction'] = front_aperture(mask)
        holes = upper_back_holes(mask)
        row = {'id': name, 'reasoning_feedback': reasoning, 'prompt': prompt,
               'source_image': str(GENERATED / file), 'image': str(path),
               'reference': ('wide_closed' if name == 'wide_closed_v2' else
                             'narrow_open' if name.startswith('tapered') else 'wide_closed_v2'),
               'metrics': measure, 'upper_back_holes': holes,
               'image_gate': (measure['projected_interface_retention'] >= .9 and
                              measure['projected_back_load_retention'] >= .8)}
        records.append(row)
        print(name, 'aperture', round(measure['front_arm_aperture_fraction'], 3),
              'holes', len(holes), 'gate', row['image_gate'])
    (OUT / 'followups.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
