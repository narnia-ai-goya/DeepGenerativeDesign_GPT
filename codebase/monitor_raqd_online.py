"""Publish per-case progress and run the final checkpoint analysis automatically."""
from __future__ import annotations

import html
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/bracket/raqd_online_2026-09-13'


def atomic(path, text):
    temporary = path.with_suffix(path.suffix+'.tmp'); temporary.write_text(text); temporary.replace(path)


while True:
    completed = []
    for path in sorted((OUT/'cases').glob('*/result.json')):
        try: completed.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError): pass
    running = sorted(path.parent.name for path in (OUT/'cases').glob('*/running.json')
                     if not (path.parent/'result.json').exists())
    summary_path = OUT/'summary.json'
    phase = 'starting'
    if summary_path.exists():
        try: phase = json.loads(summary_path.read_text()).get('phase', phase)
        except (json.JSONDecodeError, OSError): pass
    payload = {'updated_at': time.time(), 'phase': phase, 'completed': len(completed),
               'total': 52, 'running': running,
               'valid': sum(row.get('valid', False) for row in completed),
               'volume_feasible': sum(row.get('constraints_satisfied', False) for row in completed),
               'completed_ids': [row['id'] for row in completed]}
    atomic(OUT/'live_status.json', json.dumps(payload, indent=2)+'\n')
    recent = ''.join(f'<li>{html.escape(row["id"])} · valid={row.get("valid")} · '
                     f'volume feasible={row.get("constraints_satisfied")}</li>' for row in completed[-12:])
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RA-QD 실행 상태</title><style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:38px auto;padding:0 22px;color:#17212b;line-height:1.6}}.state{{background:#edf7f4;border-left:4px solid #177a65;padding:14px}}code{{overflow-wrap:anywhere}}</style><h1>RA-QD 실행 상태</h1><p class="state">단계: <b>{html.escape(phase)}</b><br>완료: <b>{len(completed)}/52</b><br>유효: {payload['valid']} · 체적 제약 통과: {payload['volume_feasible']}<br>현재 실행: {html.escape(', '.join(running))}</p><h2>최근 완료</h2><ul>{recent}</ul><p>상태 절대경로: <code>{OUT/'live_status.json'}</code></p></html>'''
    atomic(OUT/'live_status.html', page)
    if phase == 'complete':
        subprocess.run([sys.executable, str(ROOT/'codebase/analyze_raqd_online.py'),
                        '--experiment', str(OUT)], check=False)
        break
    time.sleep(30)
