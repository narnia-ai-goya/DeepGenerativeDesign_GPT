"""Summarize the pre-registered three-round chair image-QD pilot."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path('/home/goya/SDL/3d_qd/experiments/chair/sofa_style_2026-09-28/archive_feedback_loop_2026-10-03')


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    protocol = load(ROOT / 'protocol.json')
    rounds = [load(ROOT / f'round_{number:02d}' / 'result.json') for number in protocol['rounds']]
    summary = {
        'protocol': str(ROOT / 'protocol.json'),
        'rounds': protocol['rounds'],
        'initial_occupied_cells': rounds[0]['occupied_before'],
        'final_occupied_cells': rounds[-1]['occupied_after'],
        'coverage_trajectory': [rounds[0]['occupied_before']] + [r['occupied_after'] for r in rounds],
        'methods': {},
        'new_elites': [name for r in rounds for name in r['new_elites']],
        'fea_scope': rounds[-1]['fea_scope'],
    }
    for method in ('archive_qd', 'uniform_random'):
        rows = [row for r in rounds for row in r['rows'] if row['method'] == method]
        summary['methods'][method] = {
            'generated': len(rows),
            'geometry_pass': sum(row['geometry_gate'] for row in rows),
            'fea_valid': sum(row['fea_valid'] for row in rows),
            'archive_eligible': sum(row['archive_eligible'] for row in rows),
            'rows': rows,
        }
    (ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    cards = []
    for r in rounds:
        for row in r['rows']:
            rel = f"round_{r['round']:02d}/{row['id']}"
            image = f"{rel}/input.png"
            status = 'archive eligible' if row['archive_eligible'] else 'gate failed'
            details = (
                f"3D cell {row['cell']} · seat BC {row['seat_bc']:.2f} · "
                f"back BC {row['back_bc']:.2f} · repair {100*row['repair_fraction']:.1f}%"
            )
            if row['fea_valid']:
                details += f" · mass {row['mass_liters']:.2f} L · worst compliance ratio {row['worst_compliance_ratio']:.3f}"
            else:
                details += ' · ' + ', '.join(row['geometry_reasons'])
            cards.append(
                f'<article><img src="{html.escape(image)}" alt="{html.escape(row["id"])}">'
                f'<div><b>Round {r["round"]}: {html.escape(row["method"])} / {html.escape(row["id"])}</b>'
                f'<p class="{("pass" if row["archive_eligible"] else "fail")}">{status}</p>'
                f'<p>{html.escape(details)}</p><a href="round_{r["round"]:02d}/index.html">3D result</a></div></article>'
            )

    qd = summary['methods']['archive_qd']
    random = summary['methods']['uniform_random']
    traj = ' → '.join(map(str, summary['coverage_trajectory']))
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Chair image-QD feedback pilot</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;background:#111921;color:#e9f1f7;max-width:1200px;margin:auto;padding:30px}}a{{color:#88c6ff}}.metrics{{display:flex;gap:14px;flex-wrap:wrap}}.metric{{background:#233343;border-radius:12px;padding:15px;min-width:185px}}.metric strong{{display:block;font-size:26px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}}article{{background:#1c2a37;border:1px solid #3a5063;border-radius:12px;overflow:hidden}}article img{{width:100%;display:block}}article div{{padding:14px}}.pass{{color:#6fe3ad}}.fail{{color:#ffa99c}}p{{margin:.5em 0}}small{{color:#b7cad7}}</style></head><body>
<h1>Chair image-QD archive feedback pilot</h1><p>Frozen rounds 3–5: one archive-guided prompt and one uniform-random prompt per round, each generated from the same reference and evaluated with the same 3D settings.</p>
<div class="metrics"><div class="metric">Archive coverage<strong>{traj} / 9</strong></div><div class="metric">QD passes<strong>{qd['archive_eligible']} / {qd['generated']}</strong></div><div class="metric">Random passes<strong>{random['archive_eligible']} / {random['generated']}</strong></div><div class="metric">New Pareto elites<strong>{len(summary['new_elites'])}</strong></div></div>
<h2>Candidates</h2><div class="grid">{''.join(cards)}</div>
<h2>Interpretation</h2><p>The guided arm produced {qd['archive_eligible']} archive-eligible candidates and the random arm {random['archive_eligible']}. Coverage changed from {rounds[0]['occupied_before']}/9 to {rounds[-1]['occupied_after']}/9. New elites in already occupied cells improve the Pareto archive but do not increase coverage.</p>
<p><small>This six-image pilot is too small to establish superiority. The 35 mm FEM calculation is a repaired-voxel proxy, not independent FEA on the raw OBJ. See <a href="protocol.json">frozen protocol</a>, <a href="summary.json">data</a>, and <a href="../tapered_original_physics_contour_2026-10-03/index.html">combined-load contour for the original chair mesh</a>.</small></p></body></html>'''
    (ROOT / 'index.html').write_text(page)
    report = [
        '# Chair image-QD feedback pilot', '',
        f"Protocol: {ROOT / 'protocol.json'}", '',
        'Three prospective rounds (3–5), one guided and one random image per round.',
        f"Archive occupancy: {traj} / 9 cells.",
        f"QD: {qd['archive_eligible']}/{qd['generated']} archive-eligible; random: {random['archive_eligible']}/{random['generated']}.",
        f"New Pareto elites: {', '.join(summary['new_elites']) or 'none'}.", '',
        '| Round | Method | Candidate | Geometry | FEA | Archive | 3D cell |',
        '|---|---|---|---|---|---|---|',
    ]
    for r in rounds:
        for row in r['rows']:
            report.append(f"| {r['round']} | {row['method']} | {row['id']} | {row['geometry_gate']} | {row['fea_valid']} | {row['archive_eligible']} | {row['cell']} |")
    report += ['', 'The six-image pilot cannot establish method superiority. FEA uses a repaired 35 mm voxel proxy, not raw-OBJ direct FEA.', '', f"Viewer: {ROOT / 'index.html'}"]
    (ROOT / 'REPORT.md').write_text('\n'.join(report) + '\n')
    print(ROOT / 'index.html')


if __name__ == '__main__':
    main()
