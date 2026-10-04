#!/usr/bin/env python3
"""Audit connected components and BC coverage of the chair sparse trials."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from diagnose_bc_preservation import field, interior_points, load_mesh
from make_chair_domain import ROOT


BASE = ROOT / 'experiments/chair/aesthetic_reference_2026-09-27'


def inspect(path: Path, points: dict[str, np.ndarray]) -> dict:
    mesh = load_mesh(path)
    components = sorted(mesh.split(only_watertight=False),
                        key=lambda item: abs(item.volume), reverse=True)
    sdf = field(mesh)
    return {
        'path': str(path),
        'components': len(components),
        'component_volumes_litres': [round(abs(float(part.volume)) * 1000, 4)
                                     for part in components[:5]],
        'bc_containment': {kind: round(float((sdf(sample) > -.004).mean()), 4)
                           for kind, sample in points.items()},
    }


def main() -> None:
    config = json.loads((BASE / 'archpath_cfg9/config_sparse.json').read_text())
    rng = np.random.default_rng(42)
    points = {kind: interior_points(load_mesh(Path(config['stages']['post'][kind])),
                                     10000, rng)
              for kind in ('fix', 'load')}
    cases = {}
    for name in ('archpath_cfg9', 'sparse_buffer5', 'sparse_buffer20'):
        cases[name] = inspect(BASE / name / 'sparse/mesh.obj', points)
    for name in ('post', 'post_dilate3', 'post_dilate6'):
        cases[name] = inspect(BASE / 'archpath_cfg9' / name / 'mesh_bc_preserved.obj', points)
    result = {'method': '10000 fixed-seed interior BC samples; signed-distance tolerance 4 mm',
              'cases': cases}
    destination = BASE / 'sparse_checks.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(destination)


if __name__ == '__main__':
    main()
