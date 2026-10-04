#!/usr/bin/env python3
"""Probe whether stronger late sparse guidance can retain local empty space."""
from __future__ import annotations

import json

from run_chair_direct_negative_space import OUT, run_one


def main():
    mask = OUT / 'negative_space_mask.npz'
    if not mask.exists():
        raise FileNotFoundError(mask)
    result = run_one(('direct_late_w10', 10.0, 0, 20, .01), mask)
    (OUT / 'run_late.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'name': result['name'], 'exit_code': result['exit_code'],
                      'bc_pass': result.get('audit', {}).get('bc_geometry_pass'),
                      'components': result.get('audit', {}).get('mesh_components')}, indent=2))
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
