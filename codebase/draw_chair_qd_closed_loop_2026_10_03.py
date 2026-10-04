#!/usr/bin/env python3
"""Draw a QD-LLMs-Fig.-3-inspired closed-loop chair workflow, with citation."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import cairosvg

from make_chair_domain import ROOT

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'tapered_examples_2026-10-03/qd_round_01/figure'
W, H = 2400, 1390


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    items: list[str] = []
    add = items.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    add('''<defs>
      <marker id="solidArrow" markerUnits="userSpaceOnUse" markerWidth="24" markerHeight="24" refX="20" refY="12" orient="auto">
        <path d="M2 2 L21 12 L2 22 Z" fill="#246C80"/></marker>
      <marker id="futureArrow" markerUnits="userSpaceOnUse" markerWidth="24" markerHeight="24" refX="20" refY="12" orient="auto">
        <path d="M2 2 L21 12 L2 22 Z" fill="#A76F35"/></marker>
      <clipPath id="bc"><rect x="112" y="326" width="315" height="270" rx="11"/></clipPath>
      <clipPath id="im0"><rect x="517" y="335" width="116" height="142" rx="9"/></clipPath>
      <clipPath id="im1"><rect x="648" y="335" width="116" height="142" rx="9"/></clipPath>
      <clipPath id="im2"><rect x="779" y="335" width="116" height="142" rx="9"/></clipPath>
      <clipPath id="render"><rect x="1001" y="327" width="431" height="206" rx="10"/></clipPath>
      <style>
        text{font-family:'Liberation Sans','Arial',sans-serif;fill:#1A3440}
        .title{font-size:50px;font-weight:700}.subtitle{font-size:26px;fill:#58717A}
        .head{font-size:36px;font-weight:700}.sub{font-size:29px;font-weight:700}
        .body{font-size:26px}.small{font-size:23px;fill:#5B7077}.micro{font-size:21px;fill:#687E85}
        .kicker{font-size:23px;font-weight:700;fill:#277386;letter-spacing:1px}
        .flow{stroke:#246C80;stroke-width:7;fill:none;stroke-linecap:round;stroke-linejoin:round;marker-end:url(#solidArrow)}
        .future{stroke:#A76F35;stroke-width:6;stroke-dasharray:15 11;fill:none;stroke-linecap:round;stroke-linejoin:round;marker-end:url(#futureArrow)}
      </style>
    </defs>''')

    def box(x: int, y: int, w: int, h: int, fill: str = '#FFFFFF',
            stroke: str = '#CAD9DE', radius: int = 14, width: int = 2,
            dash: str | None = None) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ''
        add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{extra}/>')

    def text(x: int | float, y: int | float, value: str, style: str = 'body',
             anchor: str | None = None, color: str | None = None) -> None:
        attrs = (f' text-anchor="{anchor}"' if anchor else '') + (f' fill="{color}"' if color else '')
        add(f'<text x="{x}" y="{y}" class="{style}"{attrs}>{html.escape(value)}</text>')

    def arrow(path: str, future: bool = False) -> None:
        add(f'<path d="{path}" class="{"future" if future else "flow"}"/>')

    def image(file: Path, x: int, y: int, w: int, h: int, clip: str | None = None) -> None:
        content = base64.b64encode(file.read_bytes()).decode('ascii')
        clip_attr = f' clip-path="url(#{clip})"' if clip else ''
        add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" '
            f'xlink:href="data:image/png;base64,{content}"{clip_attr}/>')

    def panel(x: int, w: int, num: str, title: str) -> None:
        box(x, 220, w, 620)
        box(x + 16, 238, 51, 38, '#E4F0F2', '#E4F0F2', 18, 0)
        text(x + 42, 265, num, 'kicker', 'middle')
        text(x + 78, 269, title, 'head')
        add(f'<line x1="{x+18}" y1="289" x2="{x+w-18}" y2="289" stroke="#DAE5E8" stroke-width="2"/>')

    add('<rect width="2400" height="1390" fill="#F7FAFB"/>')
    add('<rect width="2400" height="12" fill="#246C80"/>')
    text(87, 72, 'Closed-loop QD for image-conditioned structural design', 'title')
    text(89, 115, 'Keep functional constraints fixed; vary shape intent; validate the realized mesh before updating the archive.', 'subtitle')
    box(1964, 46, 347, 47, '#EEF5F3', '#D2E4DE', 23, 1)
    text(2138, 78, 'CHAIR CASE STUDY', 'kicker', 'middle')

    panel(83, 360, '1', 'Specification')
    panel(478, 455, '2', 'Image proposal')
    panel(968, 485, '3', '3D realization')
    panel(1488, 468, '4', 'Gate + quality')
    panel(1991, 329, '5', 'QD archive')

    # Distinguish fixed domain constraints from editable style language.
    image(OUT / 'source_images/01_bc_preview.png',
          112, 326, 315, 270, 'bc')
    box(103, 607, 319, 84, '#EAF4F1', '#CFE2DC', 10, 1)
    text(118, 639, 'Fixed clause', 'sub')
    text(118, 671, '4 feet · 800N -Z · 200N +Y', 'small')
    box(103, 704, 319, 82, '#F7F0E6', '#E7D5B9', 10, 1)
    text(118, 736, 'Designer-editable clause', 'sub')
    text(118, 767, 'back / armrest style', 'small')

    # Image intermediate is the distinctive part of our generation route.
    text(501, 324, 'Text + reference → image edit', 'sub')
    photos = [
        OUT / 'source_images/02_tapered_straight_input.png',
        OUT / 'source_images/03_tapered_curved_input.png',
        OUT / 'source_images/04_tapered_diagonal_input.png',
    ]
    for i, file in enumerate(photos):
        x = 517 + i * 131
        box(x, 335, 116, 142, '#FFFFFF', '#DAE4E7', 9, 1)
        image(file, x, 335, 116, 142, f'im{i}')
        text(x + 58, 509, ('straight', 'curved', 'diagonal')[i], 'small', 'middle')
    box(500, 546, 410, 92, '#F0F6F7', '#D7E5E8', 10, 1)
    text(514, 584, 'Projected image check', 'sub')
    text(514, 617, 'silhouette · interface landmarks', 'small')
    text(500, 700, 'Image diversity is a proposal;', 'body')
    text(500, 738, '3D diversity is measured later.', 'body')

    text(991, 324, 'Dense 64³ → sparse 512³', 'sub')
    box(993, 327, 431, 206, '#FFFFFF', '#DAE4E7', 10, 1)
    image(OUT / 'source_images/05_tapered_original_mesh_preview.png',
          1001, 327, 431, 206, 'render')
    box(992, 556, 433, 96, '#EFF5F6', '#D5E3E7', 10, 1)
    text(1008, 591, 'Direct3D-S2 + SDF refinement', 'sub')
    text(1008, 628, 'one realized mesh x per image', 'small')
    box(992, 688, 433, 86, '#F8FAFA', '#DFE8EA', 10, 1)
    text(1009, 723, 'Shape descriptor m(x)', 'sub')
    text(1009, 754, 'side opening · backrest taper', 'small')

    # Geometry gate and physics are explicitly sequential.
    text(1511, 326, 'Hard gate before FEM', 'sub')
    for i, (label, detail) in enumerate((
        ('BC interfaces', 'seat/back load contact'),
        ('Design domain', 'envelope + repair ≤ 5%'),
        ('Mesh integrity', 'connected + watertight'),
    )):
        y = 351 + i * 93
        box(1510, y, 424, 78, '#F1F8F5', '#D1E4DA', 9, 1)
        add(f'<circle cx="1536" cy="{y+39}" r="10" fill="#2B8D74"/>')
        text(1555, y + 32, label, 'sub')
        text(1555, y + 62, detail, 'small')
    box(1509, 647, 194, 110, '#FCEFED', '#EBCBC6', 11, 1)
    text(1606, 692, 'FAIL', 'sub', 'middle', '#A54B42')
    text(1606, 731, 'diagnostic log', 'small', 'middle')
    box(1736, 647, 198, 110, '#EAF4F1', '#C9DFD6', 11, 1)
    text(1835, 692, 'PASS', 'sub', 'middle', '#237E69')
    text(1835, 731, 'mass + FEM C', 'small', 'middle')
    add('<path d="M1598 636 V644" class="flow"/>')
    add('<path d="M1840 636 V644" class="flow"/>')

    text(2013, 326, 'm(x) → niche', 'sub')
    cells = {(1, 0): ('T', '#347D90'), (2, 1): ('FC', '#4D9376'),
             (2, 2): ('FS', '#9C7656')}
    gx, gy, cw, ch = 2030, 359, 76, 78
    for j in reversed(range(3)):
        for i in range(3):
            x = gx + i * 83
            y = gy + (2-j) * 84
            item = cells.get((i, j))
            box(x, y, cw, ch, '#EAF3F3' if item else '#F5F8F8',
                item[1] if item else '#DCE6E8', 8, 2 if item else 1)
            if item: text(x + 38, y + 51, item[0], 'sub', 'middle', item[1])
    text(2148, 642, '3 / 9 cells', 'sub', 'middle')
    text(2148, 688, 'Pareto elites per niche', 'small', 'middle')
    text(2148, 727, 'mass ↓  ·  compliance ↓', 'small', 'middle')
    text(2148, 777, 'new tapered: 0 admitted', 'micro', 'middle')

    # Upper feed-forward chain and explicit reject-to-feedback branch.
    for x0, x1 in ((444, 472), (934, 962), (1454, 1482), (1957, 1985)):
        arrow(f'M{x0} 512 H{x1}')
    arrow('M1605 843 V910 H1890 V957')
    arrow('M2149 843 V957')

    # Bottom is structured like a QD-LLMs loop, but stages are our own.
    box(83, 984, 617, 276, '#FFFCF8', '#E7D7C0', 18, 2, '11 8')
    text(107, 1030, 'A  Constraint-aware prompt emitter', 'head')
    text(108, 1080, 'Choose a feasible parent; edit shape clause.', 'body')
    text(108, 1122, 'Keep BC clause and camera fixed.', 'body')
    box(107, 1150, 567, 76, '#FAF1E7', '#EBD8BF', 10, 1)
    text(121, 1183, 'Proposed next round', 'sub')
    text(121, 1213, 'LLM emitter, failure-conditioned', 'small')

    box(771, 984, 586, 276, '#FFFCF8', '#E7D7C0', 18, 2, '11 8')
    text(795, 1030, 'B  Designer + target-cell selection', 'head')
    text(795, 1080, 'Target an empty shape niche.', 'body')
    text(795, 1122, 'Accept / revise proposed style language.', 'body')
    box(795, 1150, 536, 76, '#FAF1E7', '#EBD8BF', 10, 1)
    text(809, 1182, 'Feedback:', 'sub')
    text(809, 1213, 'valid coverage + BC failure causes', 'small')

    box(1428, 984, 892, 276, '#F2F7F8', '#D2E1E5', 18)
    text(1454, 1030, 'C  Archive state + failure memory', 'head')
    text(1454, 1080, 'Elites: prompt, image, 3D mesh, descriptor, quality.', 'body')
    text(1454, 1122, 'Rejected: seat gap, back-contact loss, repair cost.', 'body')
    box(1453, 1150, 842, 76, '#E7F0F2', '#D0E0E3', 10, 1)
    text(1467, 1184, 'Pilot observation', 'sub')
    text(1467, 1214, '5 new tapered variants failed the geometry gate.', 'small')

    # Feedback goes from right to left, then upward to image generation.
    arrow('M1420 1116 H1363', True)
    arrow('M763 1116 H706', True)
    arrow('M390 980 V907 H697 V845', True)
    text(465, 939, 'new shape clause', 'micro', 'middle', '#9E6935')
    text(83, 1314, 'Solid arrows: evaluated chair pilot. Dashed arrows: proposed adaptive next-round search.', 'small')
    text(83, 1350, 'Closed-loop layout inspired by Koh et al., QD-LLMs (GECCO Companion 2026), Fig. 3; chair data and stages are our own.', 'micro')
    add('</svg>')

    svg = OUT / 'overall_framework_closed_loop.svg'
    svg.write_text('\n'.join(items))
    width_px = 7.1 * 96
    cairosvg.svg2pdf(bytestring=svg.read_bytes(), write_to=str(OUT / 'overall_framework_closed_loop.pdf'),
                     output_width=width_px, output_height=width_px * H / W)
    cairosvg.svg2png(bytestring=svg.read_bytes(), write_to=str(OUT / 'overall_framework_closed_loop.png'),
                     output_width=3600, output_height=2085)
    (OUT / 'closed_loop_caption.txt').write_text(
        'Image-conditioned structural quality-diversity workflow. A designer fixes the chair domain and functional '
        'interfaces (four fixed feet, 800 N seat force along -Z, 200 N backrest force along +Y) while varying shape '
        'language. Text and a reference rendering generate candidate images, which '
        'condition dense-to-sparse 3D realization. Mesh-derived morphology descriptors assign candidates to niches; '
        'BC, domain, and connectivity gates precede mass and FEM compliance evaluation. Feasible designs update a '
        'Pareto archive. Failure diagnostics and archive coverage inform the proposed next-round prompt emitter '
        'under designer control. The current pilot occupies 3/9 cells; five new tapered candidates were rejected. '
        'Closed-loop arrangement inspired by Koh et al., QD-LLMs, Fig. 3 (GECCO Companion 2026); all chair '
        'images, structural gates, and experiments are original to this work.\n')
    print(svg)


if __name__ == '__main__':
    main()
