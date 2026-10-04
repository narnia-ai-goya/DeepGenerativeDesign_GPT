#!/usr/bin/env python3
"""Repeat local negative-space guidance with the verified negative-inside SDF sign."""
from __future__ import annotations

import concurrent.futures
import json

from run_chair_joint_focus_projection import OUT, run_one

CASES = (('corrected_focus_w5', 5.0, 0, True),
         ('corrected_focus_w20', 20.0, 1, True))


def main():
    target = OUT / 'joint_focus_targets.npz'
    if not target.exists():
        raise FileNotFoundError(target)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(lambda row: run_one(row, target), CASES))
    (OUT / 'runs_corrected.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps([{'name': r['name'], 'exit_code': r['exit_code'],
                       'bc_pass': r.get('audit', {}).get('bc_geometry_pass'),
                       'components': r.get('audit', {}).get('mesh_components')}
                      for r in rows], indent=2))
    return max(r['exit_code'] for r in rows)


if __name__ == '__main__':
    raise SystemExit(main())
