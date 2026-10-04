#!/usr/bin/env python3
"""Draw a designer-steered QD framework with actual chair pilot imagery."""
from __future__ import annotations

import base64
import html

from make_chair_domain import ROOT

BASE=ROOT/'experiments/chair/sofa_style_2026-09-28/backrest_bc_multi_examples_2026-09-29'
ASSETS=BASE/'figure_assets'


def main():
    s=['<svg xmlns="http://www.w3.org/2000/svg" width="2800" height="1540" '
       'viewBox="0 0 2800 1540" role="img" aria-label="Designer-steered quality diversity '
       'framework for image-conditioned structural chair design">']
    add=s.append
    add('''<defs>
      <linearGradient id="canvas" x2="1" y2="1"><stop stop-color="#f8faf8"/><stop offset="1" stop-color="#edf4f2"/></linearGradient>
      <linearGradient id="deep" x2="1"><stop stop-color="#174f5d"/><stop offset="1" stop-color="#227686"/></linearGradient>
      <filter id="shadow" x="-20%" y="-30%" width="150%" height="160%"><feDropShadow dx="0" dy="8" stdDeviation="12" flood-color="#24515a" flood-opacity=".12"/></filter>
      <marker id="head" markerWidth="14" markerHeight="14" refX="11" refY="7" orient="auto"><path d="M1 1 L12 7 L1 13 Z" fill="#4c8790"/></marker>
      <marker id="futurehead" markerWidth="14" markerHeight="14" refX="11" refY="7" orient="auto"><path d="M1 1 L12 7 L1 13 Z" fill="#bf8d59"/></marker>
      <style>
        text{font-family:'Noto Sans CJK KR','Noto Sans KR',sans-serif;fill:#1d353b}
        .title{font-size:56px;font-weight:850;letter-spacing:-1.4px}.subtitle{font-size:23px;fill:#5c7378}
        .kicker{font-size:19px;font-weight:850;letter-spacing:2px;fill:#397581}
        .section{font-size:35px;font-weight:800}.head{font-size:25px;font-weight:800}
        .body{font-size:21px}.small{font-size:18px;fill:#587078}.tiny{font-size:16px;fill:#687c80}
        .white{fill:#fff}.line{stroke:#4c8790;stroke-width:8;fill:none;marker-end:url(#head)}
        .future{stroke:#bf8d59;stroke-width:7;stroke-dasharray:16 13;fill:none;marker-end:url(#futurehead)}
      </style>
    </defs>''')
    add('<rect width="2800" height="1540" fill="url(#canvas)"/>')

    def rect(x,y,w,h,fill='#fff',stroke='#d4e0df',sw=2,r=18,shadow=False,dash=None):
        a=f'x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
        if shadow:a+=' filter="url(#shadow)"'
        if dash:a+=f' stroke-dasharray="{dash}"'
        add(f'<rect {a}/>')
    def txt(x,y,v,cls='body',anchor=None,fill=None,transform=None):
        a=f'x="{x}" y="{y}" class="{cls}"'
        if anchor:a+=f' text-anchor="{anchor}"'
        if fill:a+=f' fill="{fill}"'
        if transform:a+=f' transform="{transform}"'
        add(f'<text {a}>{html.escape(v)}</text>')
    def image(name,x,y,w,h):
        b=base64.b64encode((ASSETS/f'{name}.png').read_bytes()).decode('ascii')
        add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" '
            f'preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{b}"/>')
    def panel(x,y,w,h,kicker,title):
        rect(x,y,w,h,'#fff','#d5e1e0',2,26,True)
        txt(x+34,y+48,kicker,'kicker')
        txt(x+34,y+101,title,'section')
    def pill(x,y,w,text_,fill='#e8f2f1',textcolor='#246d74'):
        rect(x,y,w,36,fill,fill,0,18)
        txt(x+w/2,y+25,text_,'small','middle',textcolor)
    def connector(path,future=False):
        add(f'<path d="{path}" class="{"future" if future else "line"}"/>')

    # A distinctive cycle, with the archive as the destination and feedback source.
    txt(72,77,'Designer-steered Quality Diversity','title')
    txt(73,117,'Image concepts become structurally feasible 3D alternatives, then populate a morphology-indexed design space.','subtitle')
    add('<circle cx="2080" cy="67" r="9" fill="#258775"/>')
    txt(2100,74,'implemented geometry pilot','small')
    add('<line x1="2428" y1="68" x2="2508" y2="68" stroke="#bf8d59" stroke-width="6" stroke-dasharray="13 10"/>')
    txt(2522,74,'proposed research loop','small')

    panel(68,186,430,1035,'DESIGNER','Intent + constraints')
    panel(575,186,860,456,'IMAGE SPACE','Multiview concept pool')
    panel(575,708,860,514,'3D REALIZATION','Dense → sparse')
    panel(1510,708,414,514,'EVALUATION','Feasibility → quality')
    panel(1994,186,735,1036,'QD ARCHIVE  ·  PROPOSED','Shape niches + Pareto quality')

    # Designer input: transparent BC preview and explicit steering controls.
    rect(99,316,368,354,'#f8fbfb','#d8e3e3',1,18)
    image('bc_preview',106,318,355,292)
    txt(281,634,'4 feet · seat load · backrest load','small','middle')
    txt(105,714,'Designer specifies','head')
    pill(104,737,168,'Envelope')
    pill(281,737,166,'Load interfaces')
    pill(104,785,344,'Style / prompt family')
    rect(105,855,345,242,'#eff6f4','#b6d0c9',2,17)
    txt(127,894,'Preference controls','head')
    txt(127,937,'• choose prompt directions','body')
    txt(127,974,'• select promising niches','body')
    txt(127,1011,'• request new variants','body')
    txt(127,1070,'Human decisions stay in the loop.','tiny')

    # Candidate image pool. Text-distance exploration proposes prompts; actual
    # morphology descriptors are assigned only after mesh realization.
    txt(614,323,'Image + text proposals','head')
    txt(614,357,'Prompt distance helps propose varied concepts; the designer chooses which to realize.','small')
    for i,(asset,name) in enumerate((('open_arm_v00_front_lo','OPEN ARM'),
                                     ('solid_side_v00_front_lo','SOLID SIDE'),
                                     ('diagonal_front','DIAGONAL BRACE'))):
        x=605+i*274
        rect(x,384,246,213,'#fbfcfc','#d4e2e1',1,16)
        image(asset,x+13,391,220,163)
        txt(x+123,578,name,'tiny','middle')
    pill(618,608,373,'front · right · top views')
    txt(1025,633,'camera-aligned silhouette targets','tiny')

    # Actual 3D pilot outcomes, shown as a family rather than one linear output.
    pill(612,833,206,'Dense 64³')
    add('<path d="M830 851 H1052" class="line"/>')
    pill(1080,833,278,'Sparse 512³ + BC clamp')
    for i,(asset,name) in enumerate((('sparse_open','OPEN ARM'),
                                     ('sparse_solid','SOLID SIDE'),
                                     ('sparse_diagonal','DIAGONAL BRACE'))):
        x=604+i*274
        rect(x,892,246,255,'#f9fbfb','#d4e2e1',1,16)
        image(asset,x+11,900,224,203)
        txt(x+123,1129,name,'tiny','middle')
    txt(612,1194,'Current pilot: 3 distinct shape families; all pass the geometric BC gate.','small')

    # Evaluation is split visibly into completed geometry checks and future
    # structural/quality tests. Do not imply FEA was run.
    rect(1540,835,352,142,'#e9f5ef','#a4cfb5',2,16)
    txt(1562,872,'DONE  ·  Geometry gate','head')
    txt(1562,913,'6 BC patches covered ≥95%','small')
    txt(1562,946,'one BC-connected body','small')
    rect(1540,997,352,172,'#fff8ed','#d5b583',2,16,dash='10 8')
    txt(1562,1035,'NEXT  ·  Independent FEA','head')
    txt(1562,1075,'seat load −Z; backrest +Y','small')
    txt(1562,1109,'compliance / stress / mass','small')
    txt(1562,1144,'image fidelity retained as gate','tiny')

    # QD archive: two interpretable, phenotype-level morphology axes, without
    # pretending that physics quality has already been measured.
    txt(2031,324,'Descriptor space  d(x)','head')
    txt(2031,357,'Mesh-derived shape diversity; not a text-only distance.','small')
    gx,gy,cw,ch=2110,405,105,110
    for row in range(4):
        for col in range(5):
            color='#f2f6f5' if (row+col)%2==0 else '#eaf1ef'
            rect(gx+col*cw,gy+row*ch,cw-8,ch-8,color,'#c5d8d5',1,10)
    cells=((1,1,'OA','#4a84a4'),(3,3,'SS','#9b8565'),(0,4,'DB','#6c9276'))
    for row,col,label,color in cells:
        x,y=gx+col*cw,gy+row*ch
        rect(x,y,cw-8,ch-8,'#f7fcfa',color,4,10)
        add(f'<circle cx="{x+(cw-8)/2}" cy="{y+42}" r="22" fill="{color}"/>')
        txt(x+(cw-8)/2,y+50,label,'tiny','middle','#fff')
        txt(x+(cw-8)/2,y+82,'pilot','tiny','middle')
    txt(2352,886,'Backrest openness  →','small','middle')
    txt(2050,655,'Side enclosure','small',transform='rotate(-90 2050 655)')
    rect(2030,918,663,181,'#f7f1e8','#d7c1a4',2,17,dash='11 9')
    txt(2055,959,'Within each niche  ·  future quality search','head')
    txt(2055,999,'Pareto choice: compliance ↓  and  mass ↓','body')
    txt(2055,1034,'FEA + image fidelity + BC feasibility determine quality.','small')
    txt(2055,1072,'Archive locations above illustrate shape diversity only.','tiny')
    txt(2032,1179,'The designer selects a niche and requests a new image concept.','small')

    # Solid implemented path and a dashed planned archive/feedback loop.
    connector('M499 420 H567')
    connector('M1005 647 V699')
    connector('M1436 946 H1501')
    connector('M1926 946 H1984',True)
    connector('M2350 1225 V1361 H281 V1233',True)
    rect(768,1328,1310,76,'#f9f1e7','#d5b38c',2,38,dash='12 10')
    txt(1423,1377,'QD feedback: preferred niches → new prompt proposals → new 3D candidates','head','middle')
    txt(72,1490,'Status: multiview + dense/sparse + geometric BC audit demonstrated on three chairs; separate-load FEA and QD archive remain proposed.','small')
    add('</svg>')
    out=BASE/'qd_framework.svg'
    out.write_text('\n'.join(s))
    print(out)


if __name__=='__main__':main()
