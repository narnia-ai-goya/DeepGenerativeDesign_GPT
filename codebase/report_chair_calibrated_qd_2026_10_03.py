"""Integrate two-round image-QD/3D/FEA chair experiment and visual gallery."""
from __future__ import annotations

import html
import json
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageFont

from make_chair_domain import ROOT


OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03'
INITIAL5 = {(0, 0), (0, 2), (1, 0), (1, 1), (2, 1)}
HISTORY7 = INITIAL5 | {(0, 1), (1, 2)}


def font(size: int, bold: bool = False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype(f'/usr/share/fonts/truetype/dejavu/{name}', size)


def fit(path: Path, w: int, h: int):
    with Image.open(path) as source:
        im = source.convert('RGB')
    im.thumbnail((w, h), Image.Resampling.LANCZOS)
    return im


def pool_sheet(screen: list[dict], chosen: dict[str, int]) -> Path:
    ordered = sorted(screen, key=lambda r: (r['method'] != 'QD pool', r['id']))
    page = Image.new('RGB', (2050, 2550), '#edf2f5')
    d = ImageDraw.Draw(page)
    d.text((25, 15), '30 front-view candidates | measured image gates', font=font(33, True), fill='#142b3a')
    d.text((28, 59), 'Green: image gate pass. Red: rejected. Blue tag: selected for 3D.', font=font(19), fill='#455966')
    for n, row in enumerate(ordered):
        x = (n % 5) * 410 + 13
        y = (n // 5) * 410 + 95
        color = '#d4eee1' if row['image_gate'] else '#f7dbd8'
        d.rounded_rectangle((x, y, x + 392, y + 391), radius=12, fill=color)
        img = fit(Path(row['image']), 340, 325)
        page.paste(img, (x + (392 - img.width) // 2, y + 35))
        tag = f"  R{chosen[row['id']]}" if row['id'] in chosen else ''
        d.text((x + 9, y + 7), row['id'] + tag, font=font(20, True), fill='#17374a')
        m = row['image_metrics']
        d.text((x + 9, y + 365), f"open {m['front_arm_aperture_fraction']:.2f}  width {m['upper_span_ratio']:.2f}  cell {row['image_cell']}",
               font=font(15), fill='#324b5d')
    path = OUT / 'image_pool_30_screened.png'
    page.save(path)
    return path


def result_sheet(rows: list[dict]) -> Path:
    page = Image.new('RGB', (1800, 1800), '#edf2f5')
    d = ImageDraw.Draw(page)
    d.text((25, 17), 'Selected image -> realized 3D | same +15.8 mm envelope',
           font=font(31, True), fill='#142b3a')
    for n, row in enumerate(rows):
        x = (n % 2) * 900 + 12
        y = (n // 2) * 430 + 80
        d.rounded_rectangle((x, y, x + 875, y + 415), radius=13, fill='white')
        d.text((x + 13, y + 10), f"{row['id']}  R{row['round']}  {row['method']}",
               font=font(19, True), fill='#17374a')
        src = fit(Path(row['image']), 310, 340)
        page.paste(src, (x + 18 + (310 - src.width) // 2, y + 43))
        mesh = fit(Path(row['preview']), 530, 310)
        page.paste(mesh, (x + 332 + (530 - mesh.width) // 2, y + 52))
        s = f"image {row['image_cell']} -> 3D {row['cell']} | valid {row['strict_eligible']} | {row['mass_liters']:.1f} L | C {row.get('worst_compliance_ratio', float('nan')):.2f}"
        d.text((x + 16, y + 390), s, font=font(16), fill='#405766')
    path = OUT / 'selected_3d_eight.png'
    page.save(path)
    return path


def progression(rows: list[dict], start: set[tuple[int, int]], method: str) -> list[int]:
    occupied = set(start)
    values = []
    for row in [r for r in rows if r['method'] == method]:
        if row['strict_eligible'] and row['cell'] is not None:
            occupied.add(tuple(row['cell']))
        values.append(len(occupied))
    return values


def main() -> None:
    screen = json.loads((OUT / 'screening.json').read_text())
    rounds = [json.loads((OUT / f'round_{i:02d}/evaluation.json').read_text()) for i in (1, 2)]
    rows = [row for result in rounds for row in result['rows']]
    chosen = {r['id']: r['round'] for r in rows}
    pool = pool_sheet(screen, chosen)
    result_image = result_sheet(rows)
    groups = ('QD pool', 'control pool')
    summary = {'image_pool': {m: {'count': sum(r['method'] == m for r in screen),
                                  'gate_pass': sum(r['method'] == m and r['image_gate'] for r in screen),
                                  'image_cells': sorted(set(tuple(r['image_cell']) for r in screen
                                                            if r['method'] == m and r['image_gate']))}
                              for m in groups},
               'initial_five_archive': sorted(INITIAL5),
               'historical_seven_archive': sorted(HISTORY7),
               'benchmark_five_progression': {m: progression(rows, INITIAL5, m) for m in groups},
               'history_aware_seven_progression': {m: progression(rows, HISTORY7, m) for m in groups},
               'image_to_3d_cell_hits': sum(r['image_cell'] == r['cell'] for r in rows),
               'median_selected_image_arm_aperture': median(r['image_metrics']['front_arm_aperture_fraction'] for r in rows),
               'median_realized_3d_arm_aperture': median(r['descriptor']['front_arm_aperture_fraction'] for r in rows),
               'selected': rows, 'image_pool_figure': str(pool), 'selected_3d_figure': str(result_image),
               'interpretation_limits': ['5-cell metric replays a frozen pilot archive; [0,1] and [1,2] were already observed in earlier experiments, so only the 7-cell metric counts genuinely new historical cells.',
                                         'Only four 3D candidates per arm; control is random eligible image selection, not random image generation.',
                                         'Image tool seed/model ID unavailable; results are not independent repeated trials.',
                                         'The +15.8 mm envelope was applied during post-generation composition/gating/FEA; VANILLA=1 meant no envelope guidance within Direct3D-S2.',
                                         'FEA is a repaired-voxel 35 mm proxy, two separate load cases, same expanded domain for all eight.']}
    (OUT / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Calibrated image-QD chair experiment', '',
             '30 newly generated front-view images (15 QD-directed, 15 non-targeted controls) were screened with frozen image descriptors and projected BC/envelope gates. Two sequential rounds selected two candidates per arm per round (8 3D solves total). The second QD round used observed first-round image→3D transfer; the control used a fixed-seed uniform draw from image-gate-passing images.', '',
             '| Method | Image gate | 3D strict valid | 5-cell benchmark progression | 7-cell historical progression |',
             '|---|---:|---:|---|---|']
    for m in groups:
        lines.append(f"| {m} | {summary['image_pool'][m]['gate_pass']}/15 | {sum(r['method']==m and r['strict_eligible'] for r in rows)}/4 | {summary['benchmark_five_progression'][m]} | {summary['history_aware_seven_progression'][m]} |")
    lines += ['', 'The 5-cell benchmark is a replay of the frozen pilot archive, **not a claim of newly discovered historical cells**. Earlier chair experiments had already found [0,1] and [1,2]. Judge genuine novelty from the 7-cell line.',
              f"Image-cell to realized-3D-cell hits: **{summary['image_to_3d_cell_hits']}/{len(rows)}**. Median arm aperture changed from {summary['median_selected_image_arm_aperture']:.3f} in images to {summary['median_realized_3d_arm_aperture']:.3f} in 3D. This is the main QD bottleneck: the image generator supplied only two valid QD-pool image cells, and the 3D generator compressed their apertures further.", '',
              '| ID | Method | Image cell → 3D cell | Strict | Mass L | Worst compliance / same-domain baseline |',
              '|---|---|---|---:|---:|---:|']
    for r in rows:
        val = f"{r['worst_compliance_ratio']:.3f}" if 'worst_compliance_ratio' in r else '—'
        lines.append(f"| {r['id']} | {r['method']} | {r['image_cell']} → {r['cell']} | {r['strict_eligible']} | {r['mass_liters']:.2f} | {val} |")
    lines += ['', 'All FEA ratios use the same +15.8 mm design envelope, fixed BC and 35 mm FEM domain. Lower compliance indicates greater stiffness under the respective load, but mass changes must be considered alongside it.', '',
              f'- Image pool: `{pool}`', f'- Selected image/mesh comparison: `{result_image}`',
              f'- Full data: `{OUT / "result.json"}`',
              '', 'Limitations: ' + ' '.join(summary['interpretation_limits']), '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    cards = []
    for r in rows:
        rel = f"round_{r['round']:02d}/mesh_cases/{r['id']}"
        cards.append(f'<article><h3>{html.escape(r["id"])} · {html.escape(r["method"])} · round {r["round"]}</h3>'
                     f'<div class="pair"><img src="images/{r["id"]}.png"><img src="{rel}/preview.png"></div>'
                     f'<p>image cell {r["image_cell"]} → 3D cell {r["cell"]} · strict {r["strict_eligible"]} · mass {r["mass_liters"]:.2f} L · worst C/baseline {r.get("worst_compliance_ratio", float("nan")):.3f}</p>'
                     f'<p>outside {r["outside_fraction"]:.1%} · repair {r["repair_fraction"]:.1%} · <a href="{rel}/combined_voxel.obj">OBJ</a> · <a href="{rel}/evaluation.json">data</a></p></article>')
    page = ('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Calibrated chair image-QD</title><style>body{font:16px/1.5 system-ui;max-width:1650px;margin:25px auto;padding:0 20px;background:#eef2f5;color:#152d3e}'
            'img{max-width:100%}article{background:white;border-radius:12px;padding:15px;margin:15px 0}.pair{display:flex;gap:12px}.pair img{width:49%;height:360px;object-fit:contain}a{color:#126296}</style>'
            '<h1>이미지 후보 선별 → 3D QD → FEA</h1>'
            f'<p>30 images, 8 matched 3D solves, image→3D cell hits {summary["image_to_3d_cell_hits"]}/8. '
            'Five-cell benchmark replay and seven-cell history-aware novelty are reported separately. '
            '<a href="REPORT.md">report</a> · <a href="result.json">data</a> · <a href="protocol.json">frozen protocol</a></p>'
            '<h2>30 image candidates and image gates</h2><img src="image_pool_30_screened.png">'
            '<h2>Eight selected 3D candidates</h2><img src="selected_3d_eight.png">'
            + ''.join(cards) + '</html>')
    (OUT / 'index.html').write_text(page)
    print('5-cell', summary['benchmark_five_progression'])
    print('7-cell', summary['history_aware_seven_progression'])
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
