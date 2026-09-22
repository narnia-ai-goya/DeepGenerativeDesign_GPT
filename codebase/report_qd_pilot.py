"""Build a portable HTML archive, candidate gallery, CSV and Markdown report."""
import argparse
import csv
import html
import json
from pathlib import Path


def build(out):
    state = json.loads((out / 'summary.json').read_text())
    protocol = json.loads((out / 'protocol.json').read_text())
    results = [json.loads(p.read_text()) for p in sorted((out / 'cases').glob('*/result.json'))]
    rows = []
    for r in results:
        rows.append({'id': r['id'], 'method': r['method'], 'round': r['round'],
                     'parent': r['parent'], 'seed': r['seed'], 'valid': r['valid'],
                     **r['genome'], 'compliance_mJ': r.get('compliance_J', 0)*1000 if 'compliance_J' in r else None,
                     'volume_mm3': r.get('volume_mm3'),
                     'design_volume_fraction': r.get('descriptors', [None, None])[0],
                     'upper_z_material_share': r.get('descriptors', [None, None])[1],
                     'max_stress_MPa': r.get('stress_max_MPa'),
                     'thin_ridge_fraction': r.get('thin_ridge_fraction'),
                     'seconds': r['seconds'], 'mesh_mm': r.get('mesh_mm'),
                     'invalid_reasons': '; '.join(r['invalid_reasons'])})
    fields = list(rows[0]) if rows else ['id', 'method', 'round', 'parent', 'seed', 'valid',
              'style', 'cfg', 'sp_cfg', 'sp_guide_w_peak', 'compliance_mJ', 'volume_mm3',
              'design_volume_fraction', 'upper_z_material_share', 'max_stress_MPa',
              'thin_ridge_fraction', 'seconds', 'mesh_mm', 'invalid_reasons']
    with (out / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    def link(path, label=None):
        p = Path(path)
        try: href = str(p.relative_to(out))
        except ValueError: href = str(p)
        return f'<a href="{html.escape(href, quote=True)}">{html.escape(label or str(p))}</a>'
    def number(x, digits=3): return f'{x:.{digits}f}' if x is not None else '—'
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             '<title>Bracket QD pilot</title>',
             '''<style>body{font:16px system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 24px;color:#203043;background:#f7f9fc}h1,h2{color:#152839}a{color:#165ab3;overflow-wrap:anywhere}table{border-collapse:collapse;background:white;margin:16px 0}th,td{padding:10px;border:1px solid #d8e0ea;text-align:left}section{background:white;padding:20px;margin:18px 0;border-radius:12px}.archives{display:flex;gap:24px;flex-wrap:wrap}.grid td{width:140px;height:70px;text-align:center}.empty{color:#9aa5b3;background:#f0f3f7}.filled{background:#d4ebe1}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}.card{background:white;padding:18px;border-radius:12px}.card img{width:100%;height:260px;object-fit:contain}.card p{margin:8px 0}.path{font:12px monospace}.small{font-size:13px;color:#536477}select{padding:8px}.scroll{overflow-x:auto}</style>''',
             '<h1>Bracket: MAP-Elites vs random search</h1>',
             f'<p>상태: <b>{html.escape(state["phase"])}</b> · 실제 평가 {len(results)}/{protocol["unique_jobs"]}회</p>',
             '<p>고정된 이미지 3세트와 생성 변수 3개를 탐색합니다. 각 방법은 공통 초기 후보 3개 + 두 라운드 × 3개로 9회씩 평가합니다. 공통 후보를 중복 실행하지 않아 실제 작업은 15개입니다.</p>',
             '<p>가로축: 설계 영역 재료 점유율. 세로축: 재료 중 CAD 높이 중간선 위에 놓인 비율. 두 축 모두 사전에 [0,1]을 4등분했습니다. 각 칸에서는 최종 FEA compliance가 작은 후보를 보관합니다.</p>',
             '<p>설계 영역은 원본 CAD에서 고정·하중 고체를 뺀 부분이며, 동일한 1 mm 격자에서 측정합니다. 최종 FEA는 1000 N, −Z, E=110 GPa, ν=0.3으로 고정했습니다.</p>',
             '<p class="small">한 번의 소규모 파일럿입니다. 동일 seed에서도 CUDA 결과가 달라질 수 있으며, 통계적 우월성이나 제조 적합성을 증명하지 않습니다. 이미지 세트는 기존 builtin Kagome와 legacy 스타일을 함께 재사용했습니다. 추가 이미지 API 호출은 없습니다.</p>']
    if (out / 'comparison.png').exists():
        parts.append('<a href="comparison.svg"><img style="width:100%" src="comparison.png" alt="Behavior space, coverage, and compliance plots"></a><p class="small">좌측 회색 영역은 CAD 상·하부의 재료 수용량만으로도 도달할 수 없는 조합입니다. 그래프의 전체 격자 범위와 coverage 분모 16은 유지합니다.</p>')
    parts.append('<details><summary>고정된 입력 이미지 3세트 보기</summary><div class="cards">')
    for style in protocol['bank']:
        src = f'image_bank/{style}/v00_front_lo.png'
        parts.append(f'<article class="card"><h3>{html.escape(style)}</h3><img src="{src}" alt="{style} 입력 정면"><p class="path">{out / "image_bank" / style}</p></article>')
    parts.append('</div><p class="small">대표 정면만 표시했습니다. 생성에는 각 폴더의 6개 뷰를 모두 사용합니다.</p></details>')
    parts.append('<div class="scroll"><table><tr><th>방법</th><th>평가</th><th>유효</th><th>채운 칸</th><th>Coverage</th><th>최소 C (mJ)</th><th>QD score</th></tr>')
    md = ['# Bracket QD pilot', '', f'상태: {state["phase"]}', '',
          '| 방법 | 평가 | 유효 | 채운 칸 | Coverage | 최소 C (mJ) | QD score |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for method, a in state['methods'].items():
        valid = sum(h['valid'] for h in a['history'])
        best = a['best_compliance_J']*1000 if a['best_compliance_J'] is not None else None
        cells = [method, str(a['evaluations']), str(valid), f'{a["occupied_cells"]}/16',
                 f'{a["coverage"]:.1%}', number(best), number(a['qd_score'])]
        parts.append('<tr>' + ''.join(f'<td>{x}</td>' for x in cells) + '</tr>')
        md.append('| ' + ' | '.join(cells) + ' |')
    parts.append('</table></div><p class="small">QD score = Σ 1/(1+C/0.01 J), occupied cells only. Coverage = occupied/16. 표와 archive는 완료된 라운드 기준입니다.</p><div class="archives">')
    for method, a in state['methods'].items():
        elites = {tuple(e['cell']): e for e in a['elites']}
        parts.append(f'<section><h2>{method}</h2><table class="grid">')
        for y in reversed(range(4)):
            parts.append(f'<tr><th>{y/4:.2f}–{(y+1)/4:.2f}</th>')
            for x in range(4):
                e = elites.get((x, y))
                if e:
                    parts.append(f'<td class="filled"><a href="#{e["id"]}">{e["id"]}</a><br>{e["compliance_J"]*1000:.3f} mJ</td>')
                else: parts.append('<td class="empty">빈 칸</td>')
            parts.append('</tr>')
        parts.append('<tr><th>상부 재료 비율 ↑<br>점유율 →</th>' + ''.join(f'<th>{i/4:.2f}–{(i+1)/4:.2f}</th>' for i in range(4)) + '</tr></table></section>')
    parts.append('</div><h2>후보 형상</h2><label>표시: <select id="filter"><option value="all">전체</option><option value="elite">최종 elite</option><option value="map_elites">MAP-Elites</option><option value="random">Random</option><option value="shared">공통 초기 후보</option></select></label><div class="cards">')
    elite_ids = {e['id'] for a in state['methods'].values() for e in a['elites']}
    for r in results:
        parts.append(f'<article class="card" id="{r["id"]}" data-method="{r["method"]}" data-elite="{int(r["id"] in elite_ids)}"><h3>{r["id"]}</h3>')
        if 'preview' in r:
            src = str(Path(r['preview']).relative_to(out))
            parts.append(f'<a href="{src}"><img loading="lazy" src="{src}" alt="{r["id"]} 최종 형상"></a>')
        parts.append(f'<p>{html.escape(r["genome"]["style"])} · {"유효" if r["valid"] else "실패"}</p>')
        if 'descriptors' in r:
            parts.append(f'<p>C = {r["compliance_J"]*1000:.3f} mJ · V = {r["volume_mm3"]:,.0f} mm³</p><p>점유율 {r["descriptors"][0]:.3f} · 상부 비율 {r["descriptors"][1]:.3f}</p>')
        parts.append(f'<p class="small">Parent: {r["parent"] or "없음"} · seed {r["seed"]}<br>' + ' · '.join(f'{k}={r["genome"][k]:.2f}' for k in protocol['parameters']) + '</p>')
        if r.get('mesh_mm'): parts.append('<p>' + link(r['mesh_mm'], 'STL 다운로드 (mm)') + '</p><p class="path">' + html.escape(r['mesh_mm']) + '</p>')
        if r['invalid_reasons']: parts.append('<p>' + html.escape('; '.join(r['invalid_reasons'])) + '</p>')
        parts.append('</article>')
    record_names = ['protocol.json', 'summary.json', 'results.csv', 'report.md']
    record_names.extend(name for name in ['findings.html', 'findings.md', 'findings.json', 'verification.json'] if (out / name).exists())
    parts.append('</div><h2>기록</h2><p>' + ' · '.join(link(out / name) for name in record_names) + '</p>')
    parts.append('''<script>document.getElementById('filter').addEventListener('change',e=>{document.querySelectorAll('.card[data-method]').forEach(c=>{const f=e.target.value;c.hidden=!(f==='all'||(f==='elite'?c.dataset.elite==='1':c.dataset.method===f));});});</script></html>''')
    (out / 'report.html').write_text('\n'.join(parts))
    md.extend(['', '## Protocol and limitations', '',
               '- Fixed 4×4 archive, both descriptors in [0,1], no post-hoc bin tuning.',
               '- Design volume fraction and upper-Z material share use the same fixed 1 mm CAD samples.',
               '- Final mesh FEA compliance is minimized; load/material/post-processing are fixed.',
               '- Watertight, one component, ≥99% domain containment with half-voxel tolerance, finite 0<C<1 J.',
               '- Stress, thickness and style transfer are diagnostics; no manufacturing or style-quality guarantee.',
               '- Equal nine-call logical budgets; three shared initial results; fifteen unique physical evaluations.',
               '- Failed candidates count; comparisons use a single stochastic run.',
               '- Positive bounded QD score is Σ 1/(1+C/0.01 J), not a cross-task metric.', '',
               '## Files', '', *[f'- {out / n}' for n in ['report.html', 'results.csv', 'summary.json', 'protocol.json']], ''])
    (out / 'report.md').write_text('\n'.join(md))
    print(out / 'report.html')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out', type=Path)
    build(ap.parse_args().out.resolve())
