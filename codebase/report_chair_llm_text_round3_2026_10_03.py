"""Report the LLM-feedback text round through new images, Dense, Sparse and FEA."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

from make_chair_domain import ROOT

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402


OUT = ROOT / 'experiments/chair/sofa_style_2026-09-28/text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03/llm_text_proposal_round_03_2026-10-03'
BASE = OUT.parent


def relative(path: str | Path) -> str:
    return os.path.relpath(path, OUT)


def qd_history(selected: dict) -> tuple[str, str, dict]:
    historical = json.loads((BASE / 'result.json').read_text())
    previous = json.loads((BASE / 'sparse_fea_loop_2026-10-03/result.json').read_text())
    old = {tuple(cell) for cell in historical['historical_seven_archive']}
    assert len(old) == 7
    assert previous['history_aware_archive_progression'] == {
        'QD pool': [7, 7, 7, 7], 'control pool': [7, 7, 7, 7]}
    assert tuple(selected['cell']) not in old and selected['frozen_cell'] is None
    # x: front arm aperture; y: upper / middle span. The fourth x-bin is
    # exploratory and must not be reported as an improvement of the frozen 3x3.
    cells = []
    for y in reversed(range(3)):
        row = []
        for x in range(4):
            key = (x, y)
            state = 'new' if key == tuple(selected['cell']) else ('old' if key in old else 'empty')
            row.append({'cell': [x, y], 'state': state})
            xlabels = ('0–.15', '.15–.30', '.30–.45', '.45–.60*')
            row[-1]['html'] = (f'<div class="cell {state}"><b>[{x},{y}]</b><small>{xlabels[x]}</small>'
                               f'<span>{"이번 후보" if state == "new" else ("기존 점유" if state == "old" else "빈 셀")}</span></div>')
        cells.append(row)
    grid = ('<div class="archive"><div class="ylabel">상단/중간 폭 비율<br>상: 0.967–1.20<br>중: 0.733–0.967<br>하: 0.50–0.733</div>'
            '<div class="cells">' + ''.join(c['html'] for row in cells for c in row) + '</div></div>'
            '<p class="axis">정면 팔걸이 개방률 → · * 네 번째 열은 이번 탐색에서만 확장</p>')
    cards = []
    rows = []
    for item in previous['selected']:
        cell = item['cell']
        rows.append({'id': item['id'], 'method': item['method'], 'round': item['round'],
                     'cell': cell, 'mass_liters': item['mass_liters'],
                     'worst_compliance_ratio': item['worst_compliance_ratio'],
                     'strict_eligible': item['strict_eligible'], 'image': item['image'],
                     'preview': str(Path(item['aligned_mesh']).with_name('sparse_preview.png'))})
        image_path = Path(item['image'])
        preview_path = Path(item['aligned_mesh']).with_name('sparse_preview.png')
        assert image_path.exists() and preview_path.exists(), item['id']
        cards.append(f'<article><h3>{html.escape(item["id"])} · {html.escape(item["method"])} · R{item["round"]}</h3>'
                     f'<p>3D 셀 {cell} · 질량 {item["mass_liters"]:.2f} L · 최악 C/기준 {item["worst_compliance_ratio"]:.3f}</p>'
                     f'<div class="pair"><img loading="lazy" src="{relative(image_path)}" alt="입력 이미지">'
                     f'<img loading="lazy" src="{relative(preview_path)}" alt="Sparse 메쉬"></div></article>')
    data = {'frozen_3x3_historical_occupied': sorted(map(list, old)),
            'frozen_3x3_coverage_before': '7/9', 'frozen_3x3_coverage_after': '7/9',
            'exploratory_4x3_coverage_before': '7/12',
            'exploratory_4x3_coverage_after': '8/12',
            'new_candidate_cell': selected['cell'],
            'new_candidate_not_frozen_3x3_elite': True,
            'previous_sparse_fea_candidates': rows,
            'comparison_note': 'The previous eight candidates did not increase history-aware frozen 3x3 coverage. The fourth aperture bin was introduced only for the new LLM candidate; 8/12 is not comparable to 7/9.'}
    (OUT / 'qd_history.json').write_text(json.dumps(data, indent=2) + '\n')
    return grid, ''.join(cards), data


def dense_preview(name: str) -> Path:
    case = OUT / name
    path = case / 'dense_preview.png'
    if path.exists():
        return path
    mesh = trimesh.load(case / 'dense_aligned.obj', force='mesh', process=False)
    page = Image.new('RGB', (870, 460), '#f6f8fa')
    draw = ImageDraw.Draw(page)
    center = np.asarray([0., .01, .46])
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
    for column, (label, azim) in enumerate((('FRONT', 0), ('RIGHT', 90))):
        eye, up = camera_from_elev_azim(center, 2., 15, azim)
        rendered = render_lit(mesh, eye, center, up, size=420, fit_extent=.56,
                              color=(.64, .69, .73))
        page.paste(Image.fromarray(rendered).convert('RGB'), (column * 435 + 7, 27))
        draw.text((column * 435 + 18, 7), label, font=font, fill='#263c4a')
    page.save(path)
    return path


def fit(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert('RGB')
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    return image


def main() -> None:
    screen = json.loads((OUT / 'image_screening.json').read_text())
    dense = json.loads((OUT / 'dense_screening.json').read_text())
    selected = json.loads((OUT / 'round_01/evaluation.json').read_text())['rows'][0]
    chosen = next(row for row in screen if row['id'] == selected['id'])
    for row in dense:
        if row['dense_generated']:
            dense_preview(row['id'])
    trace = (OUT / selected['id'] / 'sparse_generation.log').read_text()
    assert trace.count('[sp FEA step ') == 4 and '[sparse SDF smooth]' in trace
    assert selected['strict_eligible'] and selected['cell'] == [3, 1]
    archive_grid, past_cards, history = qd_history(selected)
    canvas = Image.new('RGB', (1800, 590), '#edf2f5')
    draw = ImageDraw.Draw(canvas)
    bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 25)
    labels = ('New LLM image', 'Dense transfer', 'Sparse + FEA + smooth')
    paths = (Path(chosen['image']), OUT / selected['id'] / 'dense_preview.png',
             Path(selected['aligned_mesh']).with_name('sparse_preview.png'))
    widths = (440, 650, 650)
    lefts = (10, 460, 1130)
    for name, path, width, left in zip(labels, paths, widths, lefts):
        draw.rounded_rectangle((left, 9, left + width, 575), radius=12, fill='white')
        draw.text((left + 15, 15), name, font=bold, fill='#18384b')
        picture = fit(path, width - 20, 505)
        canvas.paste(picture, (left + (width - picture.width) // 2, 54 + (505 - picture.height) // 2))
    figure = OUT / 'llm_image_dense_sparse_fea.png'
    canvas.save(figure)
    summary = {'stage':'one prospective LLM text-feedback round',
               'new_text_proposals':4, 'image_generations':5,
               'image_functional_pass':sum(row['functional_image_gate'] for row in screen),
               'dense_generated':sum(row['dense_generated'] for row in dense),
               'sparse_fea_generated':1, 'selected_id':selected['id'],
               'image_aperture':chosen['image_metrics']['front_arm_aperture_fraction'],
               'dense_aperture':next(row['dense_descriptor']['front_arm_aperture_fraction']
                                     for row in dense if row['id']==selected['id']),
               'sparse_raw_aperture':0.5045551463461911,
               'realized_aperture':selected['descriptor']['front_arm_aperture_fraction'],
               'frozen_3x3_cell':selected['frozen_cell'],
               'exploratory_4x3_cell':selected['cell'],
               'mass_liters':selected['mass_liters'],
               'worst_compliance_ratio':selected['worst_compliance_ratio'],
               'strict_geometry_fea_valid':selected['strict_eligible'],
               'qd_history':history,
               'figure':str(figure),
               'limitations':['The new [3,1] cell is in an extended 4x3 archive; old 7/9 coverage is not numerically comparable.',
                              'Only one new prompt candidate reached Sparse+FEA.',
                              'The new geometry is stiff enough for a valid solve, but worse than the same-domain compliance baseline.',
                              'Dense follow-up aperture threshold 0.24 was chosen exploratorily after viewing the image batch.']}
    (OUT / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# LLM feedback-to-text chair QD round 3', '',
             'Unlike the prior fixed-pool selection, an in-session LLM wrote four new text prompts from the missing 3D cells and measured image→Sparse aperture collapse. Five new images were generated, including one LLM repair of a wide-back failure. Functional projected BC/envelope gates passed 3/5. The old image descriptor upper bound 0.45 was not used to reject a physically plausible image.', '',
             '| Candidate | Image aperture | Dense aperture | Functional image | Dense to Sparse |',
             '|---|---:|---:|---:|---|']
    for row in screen:
        dr = next(x for x in dense if x['id'] == row['id'])
        aperture = (f"{dr['dense_descriptor']['front_arm_aperture_fraction']:.3f}"
                    if dr['dense_generated'] else '—')
        lines.append(f"| {row['id']} | {row['image_metrics']['front_arm_aperture_fraction']:.3f} | {aperture} | {row['functional_image_gate']} | {'yes' if dr.get('follow_to_sparse') else 'no'} |")
    lines += ['',
              'The strongest transfer, `r3_narrow_outerloop`, continued through 30 Sparse steps with nonzero seat-FEA gradients at 15/20/25/30 and σ=3 volume-matched SDF smoothing. Independent 800 N seat and 200 N back FEA used the same +15.8 mm domain as the previous loop.', '',
              f"Aperture: **{summary['image_aperture']:.3f} image → {summary['dense_aperture']:.3f} Dense → {summary['sparse_raw_aperture']:.3f} raw Sparse → {summary['realized_aperture']:.3f} BC/envelope composition**. Mass **{summary['mass_liters']:.2f} L**; worst compliance / same-domain baseline **{summary['worst_compliance_ratio']:.3f}**. Seat and back FEA are valid, but the stiffness is worse than baseline. The frozen 3×3 aperture range ends at 0.45, so this geometry is out of range there. Under an explicitly extended fourth aperture bin (0.45,0.60], it occupies exploratory cell **[3,1]**. This is phenotype expansion, not proof of superior quality or a comparable 3×3 archive gain.", '',
              f'- Visual: `{figure}`', f'- Original Sparse OBJ: `{selected["generated_mesh"]}`',
              f'- FEA evaluation: `{OUT / "round_01/evaluation.json"}`',
              f'- New text prompts: `{OUT / "proposals.json"}`',
              f'- Historical QD comparison: `{OUT / "qd_history.json"}`',
              '', '## Cumulative QD context', '',
              'The historical frozen 3×3 archive covered **7/9 cells**. Eight earlier Sparse+FEA candidates (four QD pool and four control pool) did not add a new cell: **7/9 → 7/9**. The new LLM-text candidate has frozen cell `None`, so the same 3×3 archive remains **7/9**. In a separately extended 4×3 archive, the historical cells plus the new strict-valid candidate occupy **8/12** cells. These denominators must not be compared as a QD gain. Its worst compliance ratio 1.590 is worse than all eight listed earlier candidates; no quality advantage is established.', '',
              '', 'Limitations: ' + ' '.join(summary['limitations']), '']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines))
    cards = []
    for row in screen:
        dr = next(x for x in dense if x['id'] == row['id'])
        view = (f'<img src="{row["id"]}/dense_preview.png">' if dr['dense_generated'] else
                '<p>Image functional gate failed; Dense not run.</p>')
        reason = ', '.join(row['reasons']) or 'functional image gate passed'
        cards.append(f'<article><h3>{html.escape(row["id"])} · {"PASS" if row["functional_image_gate"] else "FAIL"}</h3>'
                     f'<p>image aperture {row["image_metrics"]["front_arm_aperture_fraction"]:.3f} · {html.escape(reason)}</p>'
                     f'<div class="pair"><img src="{row["id"]}.png">{view}</div></article>')
    page = ('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>LLM text QD round 3</title><style>body{font:16px/1.5 system-ui;max-width:1480px;margin:24px auto;padding:0 20px;background:#edf2f5;color:#173548}'
            'article{background:white;border-radius:12px;margin:18px 0;padding:16px}.pair{display:flex;gap:12px;align-items:center}.pair img{width:49%;height:300px;object-fit:contain}img.hero{width:100%}a{color:#126396}'
            '.archive{display:flex;gap:12px;align-items:center}.ylabel{min-width:175px;font-size:13px}.cells{display:grid;grid-template-columns:repeat(4,minmax(100px,1fr));gap:8px;flex:1}'
            '.cell{min-height:95px;padding:9px;border-radius:9px;border:2px solid #c8d2d9;display:flex;flex-direction:column}.cell small{color:#385669}.cell span{margin-top:auto}.cell.old{background:#c9e5ed;border-color:#6da7b7}.cell.new{background:#f9d9a4;border-color:#bc7315}.cell.empty{background:#f7f9fa}.axis{text-align:center}'
            '.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.stats div{background:white;border-radius:10px;padding:14px}.stats b{display:block;font-size:25px}'
            '.past{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.past article{margin:0}.past .pair img{height:230px}'
            '@media(max-width:800px){.past{grid-template-columns:1fr}.archive{display:block}.ylabel{margin-bottom:10px}.cells{grid-template-columns:repeat(4,minmax(0,1fr))}.cell{font-size:11px;min-height:76px}.cell small{font-size:10px}}</style>'
            '<h1>새 LLM 텍스트 → 이미지 → Dense → Sparse + FEA</h1>'
            f'<p>4 new texts, 5 images, 3 projected functional passes, 3 Dense, 1 Sparse+FEA. '
            f'Exploratory 4×3 cell {selected["cell"]}; worst C/ref {selected["worst_compliance_ratio"]:.3f}. '
            '<a href="RESULTS.md">report</a> · <a href="result.json">data</a> · <a href="proposals.json">new prompts</a> · <a href="qd_history.json">QD history</a></p>'
            '<h2>QD 결과: 이전 실험과 이번 후보</h2>'
            '<div class="stats"><div>기존 3×3 archive<b>7/9 → 7/9</b>이번 후보는 기존 범위 밖</div>'
            '<div>탐색적 4×3 archive<b>7/12 → 8/12</b>네 번째 개방률 구간을 추가</div>'
            f'<div>새 후보 최악 C/기준<b>{selected["worst_compliance_ratio"]:.3f}</b>1보다 높아 강성 저하</div></div>'
            '<article><h3>3D 형상 archive</h3>' + archive_grid +
            '<p>파랑은 과거 점유 셀 7개, 주황은 이번 LLM 텍스트 후보입니다. 기존 8개 Sparse+FEA 추가 실험은 기존 3×3 coverage를 늘리지 못했습니다. 이번 후보는 독립 FEA와 형상 검사를 통과했지만, 기존 기준보다 약합니다. 7/9와 8/12는 분모가 달라 직접 비교할 수 없습니다.</p></article>'
            '<h2>이번 후보: 입력 이미지 → Dense → Sparse</h2>'
            '<img class="hero" src="llm_image_dense_sparse_fea.png">'
            f'<h2>Original Sparse mesh</h2><img src="round_01/mesh_cases/{selected["id"]}/sparse_preview.png">'
            '<h2>이전 8개 Sparse+FEA 후보</h2><p>과거 후보의 입력 이미지와 원본 Sparse 메쉬입니다. 이 후보들은 이번 텍스트 생성 라운드의 결과가 아닙니다.</p>'
            '<div class="past">' + past_cards + '</div>'
            '<h2>새 이미지 후보 5개와 Dense 전달</h2>' + ''.join(cards) + '</html>')
    (OUT / 'index.html').write_text(page)
    print(OUT / 'index.html')


if __name__ == '__main__':
    main()
