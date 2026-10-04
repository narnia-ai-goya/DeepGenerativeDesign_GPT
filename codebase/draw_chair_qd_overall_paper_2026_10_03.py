#!/usr/bin/env python3
"""Publication-ready, evidence-aligned overall framework figure (SVG/PDF/PNG)."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import cairosvg

from make_chair_domain import ROOT

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'tapered_examples_2026-10-03/qd_round_01/figure'
SOURCE_IMAGES = OUT / 'source_images'
W, H = 2400, 1280


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    elements: list[str] = []
    add = elements.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
        'aria-label="Designer-steered image-conditioned quality diversity framework for structural chair design">')
    add('''<defs>
      <marker id="arrow" markerUnits="userSpaceOnUse" markerWidth="22" markerHeight="22" refX="19" refY="11" orient="auto">
        <path d="M1 1 L20 11 L1 21 Z" fill="#256478"/></marker>
      <marker id="looparrow" markerUnits="userSpaceOnUse" markerWidth="22" markerHeight="22" refX="19" refY="11" orient="auto">
        <path d="M1 1 L20 11 L1 21 Z" fill="#A76B35"/></marker>
      <clipPath id="img1"><rect x="588" y="368" width="135" height="135" rx="10"/></clipPath>
      <clipPath id="img2"><rect x="730" y="368" width="135" height="135" rx="10"/></clipPath>
      <clipPath id="img3"><rect x="872" y="368" width="135" height="135" rx="10"/></clipPath>
      <clipPath id="bc"><rect x="135" y="354" width="335" height="278" rx="12"/></clipPath>
      <clipPath id="mesh"><rect x="1105" y="373" width="520" height="235" rx="11"/></clipPath>
      <style>
        text{font-family:'Liberation Sans','Arial',sans-serif;fill:#19313C}
        .title{font-size:52px;font-weight:700}.subtitle{font-size:26px;fill:#516A73}
        .num{font-size:25px;font-weight:700;fill:#247385;letter-spacing:1.5px}
        .head{font-size:40px;font-weight:700}.sub{font-size:32px;font-weight:700}
        .body{font-size:29px}.small{font-size:25px;fill:#526873}.tiny{font-size:22px;fill:#667A82}
        .white{fill:white}.blue{fill:#256478}.orange{fill:#A76B35}
        .flow{fill:none;stroke:#256478;stroke-width:7;stroke-linecap:round;marker-end:url(#arrow)}
        .loop{fill:none;stroke:#A76B35;stroke-width:7;stroke-linecap:round;marker-end:url(#looparrow)}
      </style>
    </defs>''')

    def rect(x: float, y: float, w: float, h: float, fill: str = '#FFF',
             stroke: str = '#CBD8DC', r: float = 14, sw: float = 2,
             dash: str | None = None) -> None:
        tail = f' stroke-dasharray="{dash}"' if dash else ''
        add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{tail}/>')

    def txt(x: float, y: float, value: str, cls: str = 'body',
            anchor: str | None = None, color: str | None = None) -> None:
        attrs = f' text-anchor="{anchor}"' if anchor else ''
        if color: attrs += f' fill="{color}"'
        add(f'<text x="{x}" y="{y}" class="{cls}"{attrs}>{html.escape(value)}</text>')

    def line(x1: float, y1: float, x2: float, y2: float, stroke: str = '#D8E2E5',
             sw: float = 2) -> None:
        add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>')

    def path(d: str, cls: str = 'flow') -> None:
        add(f'<path d="{d}" class="{cls}"/>')

    def embed(file: Path, x: float, y: float, w: float, h: float,
              clip: str | None = None) -> None:
        image = base64.b64encode(file.read_bytes()).decode('ascii')
        clipping = f' clip-path="url(#{clip})"' if clip else ''
        add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
            f'preserveAspectRatio="xMidYMid meet" '
            f'xlink:href="data:image/png;base64,{image}"{clipping}/>')

    def card(x: int, w: int, number: str, title: str) -> None:
        rect(x, 195, w, 572, '#FFF', '#D1DDE0', 18)
        rect(x + 20, 215, 69, 42, '#E4F0F2', '#E4F0F2', 21, 0)
        txt(x + 54, 246, number, 'num', 'middle')
        txt(x + 104, 249, title, 'head')
        line(x + 26, 278, x + w - 26, 278)

    add('<rect width="2400" height="1280" fill="#F6F9FA"/>')
    add('<rect width="2400" height="13" fill="#246B7D"/>')
    txt(90, 75, 'Designer-steered structural quality diversity', 'title')
    txt(92, 119, 'Search image concepts; admit only physically valid 3D designs; use the archive to guide the next proposal.', 'subtitle')
    rect(2038, 45, 263, 43, '#E9F2EF', '#C8DBD5', 21, 1)
    txt(2170, 74, 'CHAIR PILOT', 'num', 'middle')

    card(90, 423, '01', 'Specification')
    card(540, 495, '02', 'Image search')
    card(1062, 570, '03', '3D realization')
    card(1659, 651, '04', 'Physical evaluation')

    # (1) Domain and functional interfaces, using the measured chair BC.
    rect(122, 312, 359, 340, '#F9FBFB', '#DDE6E8', 12, 1)
    embed(SOURCE_IMAGES / '01_bc_preview.png',
          135, 354, 335, 278, 'bc')
    txt(125, 327, 'Ω + boundary conditions', 'sub')
    txt(122, 692, '4 fixed feet · 800N -Z · 200N +Y', 'small')
    txt(122, 731, 'Designer: style + preferred niches', 'small')

    # (2) Text and image diversity, as proposals rather than validated shape.
    txt(568, 325, 'Text-latent proposals', 'sub')
    txt(568, 352, 'Novel shape phrases → distinct image edits', 'small')
    image_base = BASE / 'tapered_examples_2026-10-03'
    photos = [
        (SOURCE_IMAGES / '02_tapered_straight_input.png', 'straight'),
        (SOURCE_IMAGES / '03_tapered_curved_input.png', 'curved'),
        (SOURCE_IMAGES / '04_tapered_diagonal_input.png', 'diagonal'),
    ]
    for i, (file, label) in enumerate(photos):
        x = 588 + i * 142
        rect(x, 368, 135, 135, '#FFF', '#D9E3E5', 10, 1)
        embed(file, x, 368, 135, 135, f'img{i+1}')
        txt(x + 67, 536, label, 'small', 'middle')
    rect(565, 578, 444, 118, '#F1F6F7', '#D9E6E8', 12, 1)
    txt(587, 616, 'Proposal descriptors', 'sub')
    txt(587, 657, 'semantic novelty + designer preference', 'small')

    # (3) Actual sparse mesh renders, avoiding a diagram-only surrogate.
    txt(1087, 325, 'Direct3D-S2: dense → sparse', 'sub')
    txt(1087, 355, '64³ topology; 512³ SDF surface refinement', 'small')
    rect(1087, 373, 520, 235, '#FFF', '#DDE6E8', 11, 1)
    embed(SOURCE_IMAGES / '05_tapered_original_mesh_preview.png', 1105, 373, 520, 235, 'mesh')
    rect(1087, 632, 520, 63, '#EBF3F5', '#D2E0E4', 10, 1)
    txt(1347, 674, 'sparse SDF surface regularization', 'small', 'middle')
    txt(1087, 731, 'One image + text → one candidate mesh', 'small')

    # (4) Hard feasibility gate and quality are explicitly separate.
    txt(1686, 325, 'Gate first; measure quality second', 'sub')
    checks = [
        ('Interface', 'seat/back BC contact'),
        ('Domain', 'envelope + low repair cost'),
        ('Geometry', 'connected + watertight'),
    ]
    for i, (name, detail) in enumerate(checks):
        y = 350 + i * 75
        rect(1687, y, 585, 65, '#F3F8F6', '#CFDFD8', 10, 1)
        add(f'<circle cx="1718" cy="{y+33}" r="13" fill="#2A8A70"/>')
        txt(1746, y + 42, name, 'sub')
        txt(1925, y + 42, detail, 'small')
    path('M1720 587 V620 H1818', 'flow')
    path('M2090 587 V620 H2190', 'flow')
    rect(1712, 632, 230, 85, '#FCEFED', '#ECC8C2', 11, 1)
    txt(1827, 667, 'REJECT', 'sub', 'middle', '#A8473A')
    txt(1827, 696, 'record failure', 'tiny', 'middle')
    rect(2022, 632, 248, 85, '#E8F3F1', '#BEDBD4', 11, 1)
    txt(2146, 667, 'ADMIT', 'sub', 'middle', '#247A69')
    txt(2146, 696, 'mass + FEM C', 'tiny', 'middle')

    # Top-row and down-flow connectors.
    path('M515 475 H533')
    path('M1037 475 H1055')
    path('M1634 475 H1652')
    path('M2145 769 V812 H1770 V842')

    # Lower feedback / archive band: unlike a linear pipeline, QD is a loop.
    rect(90, 842, 450, 355, '#FFF', '#DBD0BD', 18)
    rect(112, 865, 406, 47, '#F7EFE4', '#F7EFE4', 18, 0)
    txt(315, 899, 'DESIGNER FEEDBACK', 'num', 'middle', '#9B632C')
    txt(119, 963, 'Choose a target niche', 'sub')
    txt(119, 1011, 'or revise style / BC.', 'body')
    txt(119, 1065, 'Rejected designs inform', 'small')
    txt(119, 1102, 'feasibility of future proposals.', 'small')
    rect(117, 1130, 371, 40, '#F7EFE4', '#E6D3B9', 18, 1)
    txt(302, 1158, 'human-in-the-loop acquisition', 'tiny', 'middle')

    rect(568, 842, 1742, 355, '#FFF', '#CBDDE0', 18)
    txt(597, 894, '05', 'num')
    txt(658, 899, 'Realized-shape QD archive', 'head')
    txt(1510, 895, 'Within each cell: Pareto quality  (mass ↓, compliance ↓)', 'small')
    line(597, 922, 2282, 922)
    # Mesh-derived 3x3 descriptor map. The three cells are actual pilot elites.
    gx, gy, cw, ch = 670, 951, 125, 72
    occupied = {(1, 0): ('T', '#2D7B8E'),
                (2, 1): ('FC', '#4C9275'), (2, 2): ('FS', '#9C7554')}
    for j in reversed(range(3)):
        for i in range(3):
            x, y = gx + i * (cw + 5), gy + (2-j) * (ch + 5)
            token = occupied.get((i, j))
            rect(x, y, cw, ch, '#EBF3F4' if token else '#F6F9F9',
                 token[1] if token else '#DCE6E8', 8, 3 if token else 1)
            if token:
                txt(x + cw/2, y + 47, token[0], 'sub', 'middle', token[1])
    txt(863, 1190, 'side opening →', 'tiny', 'middle')
    txt(605, 949, 'back taper ↑', 'tiny')
    line(1090, 946, 1090, 1164)
    # A measured performance/mass scatter, not a fictional Pareto cloud.
    txt(1125, 971, 'Measured pilot elites', 'sub')
    px0, py0, px1, py1 = 1190, 1138, 1520, 1005
    line(px0, py0, px1, py0, '#9FB3BA', 3)
    line(px0, py0, px0, py1, '#9FB3BA', 3)
    vals = [(.0, .0)]
    points = [('T', 21.11, .972, '#2D7B8E'),
              ('FC', 24.03, 1.109, '#4C9275'),
              ('FS', 23.06, .930, '#9C7554')]
    for label, mass, compliance, color in points:
        x = px0 + 22 + (mass - 20.5) / 4.1 * 285
        y = py0 - 18 - (compliance - .89) / .25 * 103
        add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="{color}"/>')
        add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="16" fill="none" '
            f'stroke="{color}" stroke-width="2"/>')
        txt(x + 16, y - 8, label, 'tiny')
    txt(1360, 1170, 'mass [L] →', 'tiny', 'middle')
    txt(1132, 1115, 'C / ref.', 'tiny')
    line(1590, 946, 1590, 1164)
    txt(1623, 978, 'Archive update', 'sub')
    txt(1623, 1028, 'Accept only validated designs.', 'body')
    txt(1623, 1075, 'Keep shape niches even when a', 'small')
    txt(1623, 1110, 'design is globally dominated.', 'small')
    txt(1623, 1160, 'Observed pilot: 3 / 9 cells occupied.', 'tiny')

    # Gold return path makes the feedback loop visually distinct from feedforward.
    path('M567 1025 H548', 'loop')
    path('M315 839 V800 H785 V772', 'loop')
    txt(450, 818, 'new prompt / target', 'tiny', 'middle', '#9B632C')
    txt(90, 1240, 'Pilot status: new tapered image variants failed BC or repair gates; they remain observations, not archive elites.', 'tiny')
    add('</svg>')

    svg = OUT / 'overall_framework.svg'
    svg.write_text('\n'.join(elements))
    # CairoSVG interprets output_width in CSS pixels (96 px/in), then converts
    # the vector PDF page to 72 pt/in. Target a 7.1-inch two-column figure.
    paper_width_px = 7.1 * 96
    cairosvg.svg2pdf(bytestring=svg.read_bytes(), write_to=str(OUT / 'overall_framework.pdf'),
                     output_width=paper_width_px,
                     output_height=paper_width_px * H / W)
    cairosvg.svg2png(bytestring=svg.read_bytes(), write_to=str(OUT / 'overall_framework.png'),
                     output_width=3600, output_height=1920)
    caption = (
        'Overall framework of designer-steered, image-conditioned structural quality diversity. '
        'The designer supplies the chair design domain, four fixed feet, an 800 N seat load along -Z, '
        'a 200 N backrest load along +Y, and style preferences. '
        'Frozen text embeddings guide diverse image edits; each image conditions dense-to-sparse 3D generation. '
        'The resulting mesh is checked against functional interfaces, the design envelope, repair cost, and connectivity. '
        'Admitted candidates receive structural evaluation and enter a morphology-indexed QD archive, '
        'where mass and compliance are treated as separate quality objectives. '
        'Archive coverage and failures guide the next designer-approved prompt. '
        'The pictured pilot archive contains three feasible cells from earlier chair runs; newly generated tapered variants '
        'are excluded because they fail structural-interface or repair gates.'
    )
    (OUT / 'caption.txt').write_text(caption + '\n')
    (OUT / 'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>Chair QD overall framework</title><style>
body{font-family:system-ui,sans-serif;background:#edf2f4;color:#17313b;max-width:1500px;margin:auto;padding:22px}
img{display:block;width:100%;background:white;border:1px solid #d5e0e3}h2{margin-top:40px}
a{color:#19677b;margin-right:16px}p{line-height:1.5}</style>
<h1>Designer-steered structural quality diversity</h1>
<p><a href="source_images/index.html">Source images and BC force preview</a></p>
<h2>Closed-loop layout</h2>
<p><a href="overall_framework_closed_loop.pdf">PDF · paper width</a>
<a href="overall_framework_closed_loop.svg">SVG · editable vector</a>
<a href="overall_framework_closed_loop.png">PNG · 3600 × 2085</a>
<a href="closed_loop_caption.txt">Caption</a></p>
<img src="overall_framework_closed_loop.svg" alt="Closed-loop chair QD framework with generation, validation, archive, and feedback">
<p>Solid arrows show evaluated stages; dashed arrows show the proposed adaptive next-round search.</p>
<h2>Earlier linear layout</h2>
<p><a href="overall_framework.pdf">PDF · paper width</a>
<a href="overall_framework.svg">SVG · editable vector</a>
<a href="overall_framework.png">PNG · 3600 × 1920</a>
<a href="caption.txt">Caption</a></p>
<img src="overall_framework.svg" alt="Overall framework: specification, image search, 3D realization, validation, QD archive and feedback">
<p>Figure reflects the current chair pilot. New tapered candidates appear as observations and are excluded from the archive when BC or repair gates fail.</p></html>''')
    print(svg)


if __name__ == '__main__':
    main()
