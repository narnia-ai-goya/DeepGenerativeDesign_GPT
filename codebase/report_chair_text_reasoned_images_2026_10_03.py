"""Make an image-first comparison for AI-authored chair shape prompts."""
from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from make_chair_domain import ROOT

OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/text_reasoned_front_axes_2026-10-03'


def main() -> None:
    data = json.loads((OUT / 'protocol.json').read_text())
    rows = data['candidates']
    fig, axes = plt.subplots(2, 2, figsize=(11, 12), layout='constrained')
    for ax, row in zip(axes.flat, rows):
        with Image.open(row['image']) as im:
            ax.imshow(im)
        m = row['metrics']
        ax.set_title(f"{row['id']}\nback width ratio {m['upper_span_ratio']:.2f} · arm opening {m['front_arm_aperture_fraction']:.2f}", fontsize=12)
        ax.axis('off')
    fig.savefig(OUT / 'image_contact_sheet.png', dpi=170, facecolor='white')
    plt.close(fig)
    cards = []
    for row in rows:
        m = row['metrics']
        rel = Path(row['image']).relative_to(OUT)
        cards.append(f'''<article><img src="{html.escape(str(rel))}" alt="{html.escape(row['id'])}">
<h2>{html.escape(row['id'])}</h2><p>등받이 상단 폭 비율 <b>{m['upper_span_ratio']:.3f}</b> · 정면 팔걸이 열린 공간 <b>{m['front_arm_aperture_fraction']:.3f}</b></p>
<p>좌면·발 2D 보존 {m['projected_interface_retention']:.1%} · 등받이 하중 부위 {m['projected_back_load_retention']:.1%}</p>
<details><summary>AI가 작성한 형상 지시문</summary><p>{html.escape(row['shape_text'])}</p></details></article>''')
    followups = json.loads((OUT / 'followups.json').read_text()) if (OUT / 'followups.json').exists() else []
    followup_cards = []
    if followups:
        fig, axes = plt.subplots(1, len(followups), figsize=(15, 6), layout='constrained')
        for ax, row in zip(axes, followups):
            with Image.open(row['image']) as im:
                ax.imshow(im)
            ax.set_title(f"{row['id']}\narm opening {row['metrics']['front_arm_aperture_fraction']:.2f} · back holes {len(row['upper_back_holes'])}", fontsize=11)
            ax.axis('off')
        fig.savefig(OUT / 'reasoned_followups.png', dpi=170, facecolor='white')
        plt.close(fig)
    for row in followups:
        m = row['metrics']
        rel = Path(row['image']).relative_to(OUT)
        followup_cards.append(f'''<article><img src="{html.escape(str(rel))}" alt="{html.escape(row['id'])}">
<h2>{html.escape(row['id'])}</h2><p>팔걸이 열린 공간 <b>{m['front_arm_aperture_fraction']:.3f}</b> · 등받이 상단 관통공 <b>{len(row['upper_back_holes'])}개</b></p>
<p>등받이 하중 부위 2D 보존 {m['projected_back_load_retention']:.1%}</p>
<details><summary>실패 분석과 수정 문장</summary><p>{html.escape(row['reasoning_feedback'])}</p><p>{html.escape(row['prompt'])}</p></details></article>''')
    page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI text → chair image diversity</title>
<style>body{{font:16px/1.5 system-ui;max-width:1500px;margin:auto;padding:24px;background:#f3f6f8;color:#172431}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:20px}}article{{background:white;border:1px solid #d4dfe6;border-radius:13px;padding:16px}}img{{width:100%;display:block}}a{{color:#155b91}}p{{margin:.5em 0}}</style></head><body>
<h1>AI가 작성한 텍스트로 정면 형상 조절</h1><p>같은 참조 의자에서 등받이 폭(좁음/넓음) × 팔걸이 안쪽 공간(열림/닫힘)을 조합한 네 이미지입니다. 좌면, 다리, 발, 중앙 등받이는 보존하도록 지시했습니다.</p>
<p><a href="image_contact_sheet.png">네 이미지 한 장으로 보기</a> · <a href="protocol.json">정확한 프롬프트와 수치</a></p>
<div class="grid">{''.join(cards)}</div>
<h2>측정 피드백과 관통공 변형</h2><p>wide_closed의 열린 공간 0.402를 보고 AI가 양옆 슬롯을 더 좁히도록 문장을 수정했습니다. 후속 이미지에서 0.278로 줄었습니다. 두 관통공 예시에서는 중앙 등받이 하중 부위를 그대로 남기도록 지시했습니다.</p>
<p><a href="reasoned_followups.png">후속 이미지 한 장으로 보기</a> · <a href="followups.json">수정 이유와 정확한 프롬프트</a></p>
<div class="grid">{''.join(followup_cards)}</div><p>표시한 값은 정면 이미지의 픽셀 측정치입니다. 3D의 BC 접촉이나 기계적 성능은 별도 검증이 필요합니다.</p></body></html>'''
    (OUT / 'index.html').write_text(page)
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
