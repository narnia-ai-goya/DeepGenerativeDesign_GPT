#!/usr/bin/env python3
"""Post-process one chair sparse mesh and independently verify seat-load FEA."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from diagnose_bc_preservation import field, interior_points, load_mesh
from run_connectivity_qd_sampling import PYTHON, ROOT, render_mesh


def call(command: list[str], log: Path, env: dict[str, str] | None = None) -> None:
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'exit {result.returncode}; see {log}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('case', type=Path)
    args = parser.parse_args()
    case = args.case.resolve()
    cfg_path = case / 'config_sparse.json'
    cfg = json.loads(cfg_path.read_text())
    post = case / 'post'
    post.mkdir(exist_ok=True)
    raw = case / 'sparse/mesh.obj'
    hybrid = post / 'mesh_bc_preserved.obj'
    final = post / 'final.obj'
    if not hybrid.exists():
        call([str(PYTHON), str(ROOT / 'codebase/code/post_hybrid_union_clip.py'),
              '--config', str(cfg_path), '--in', str(raw), '--out', str(hybrid)],
             post / 'boolean.log')
    if not final.exists():
        call([str(PYTHON), str(ROOT / 'codebase/code/surface_remesh_pre.py'),
              '--config', str(cfg_path), '--in', str(hybrid), '--out', str(final)],
             post / 'remesh.log')
    mesh = load_mesh(final)
    sdf = field(mesh)
    rng = np.random.default_rng(42)
    bc = {}
    for kind in ('fix', 'load'):
        reference = load_mesh(Path(cfg['stages']['post'][kind]))
        points = interior_points(reference, 10000, rng)
        bc[kind] = float((sdf(points) > -0.004).mean())
    clearance = load_mesh(ROOT / 'data_real/chair/occupant_keepout.stl')
    clearance_points = interior_points(clearance, 10000, rng)
    clearance_occupied = float((sdf(clearance_points) > 0.0).mean())
    summary = {'final_mesh': str(final), 'volume_litres': abs(float(mesh.volume))*1000,
               'watertight': bool(mesh.is_watertight),
               'components': len(mesh.split(only_watertight=False)),
               'bc_containment': bc,
               'occupant_clearance_occupied_fraction': clearance_occupied}
    summary['geometry_valid'] = bool(summary['watertight'] and summary['components'] == 1
                                     and min(bc.values()) >= .99 and clearance_occupied <= .01)
    if summary['geometry_valid']:
        env = dict(os.environ, FENICS_PY='/home/goya/miniconda3/envs/fenics/bin/python')
        fea_dir = post / 'fea_independent'
        fea_dir.mkdir(exist_ok=True)
        fea_summary = fea_dir / 'fea_tet_summary.json'
        if not fea_summary.exists():
            call([str(PYTHON), str(ROOT / 'codebase/code/fea_prep_and_run.py'),
                  '--config', str(cfg_path), '--in', str(final), '--out-dir', str(fea_dir)],
                 post / 'fea.log', env)
        summary['fea'] = json.loads(fea_summary.read_text())
        summary['fea_summary'] = str(fea_summary)
    render_mesh(final, post / 'final_preview.png', case.name)
    (case / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
