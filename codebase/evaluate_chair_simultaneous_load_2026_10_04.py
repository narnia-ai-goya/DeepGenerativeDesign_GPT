"""Independent chair evaluation with seat and back forces in one FEM solve."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import trimesh

import evaluate_chair_calibrated_qd_2026_10_03 as existing
from evaluate_chair_protected_parts_qd_2026_10_03 import preview
from make_chair_domain import ROOT
from run_chair_dual_load_fea_2026_10_04 import PARENT, SPEC


OUT = PARENT / 'simultaneous_load_dense_sparse_2026-10-04'
FENICS = '/home/goya/miniconda3/envs/fenics/bin/python'
SOLVER = ROOT / 'codebase/code/fenics_fea_bracket.py'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('dense', 'sparse'), required=True)
    args = ap.parse_args()
    names = (['dense_control', 'dense_dual'] if args.stage == 'dense' else
             ['sparse_d0_s0', 'sparse_d0_s1', 'sparse_d1_s0', 'sparse_d1_s1'])
    eval_root = OUT / f'{args.stage}_evaluation'
    folder = eval_root / 'round_01'
    folder.mkdir(parents=True, exist_ok=True)
    source = json.loads((PARENT / 'round_01/selection.json').read_text())['selected'][0]
    selected = []
    for name in names:
        dst = folder / name / 'generation'
        dst.mkdir(parents=True, exist_ok=True)
        link = dst / 'mesh.obj'
        src = OUT / name / 'generation/mesh.obj'
        if not src.exists():
            raise FileNotFoundError(src)
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src)
        selected.append(dict(source, id=name, method=name))
    (folder / 'selection.json').write_text(json.dumps({'round': 1, 'selected': selected}, indent=2) + '\n')
    existing.OUT = eval_root
    sys.argv = ['evaluation', '--round', '1', '--overflow-aperture-max', '.6']
    existing.main()
    rows = json.loads((folder / 'evaluation.json').read_text())['rows']
    results = []
    for row in rows:
        name = row['id']
        case = folder / 'mesh_cases' / name
        raw = trimesh.load(case / 'aligned_main.obj', force='mesh', process=False)
        raw_preview = OUT / f'{name}_raw_preview.png'
        preview(raw, raw_preview)
        fea = case / 'fea'
        work = fea / 'simultaneous'
        work.mkdir(parents=True, exist_ok=True)
        command = [FENICS, str(SOLVER),
                   '--domain-dir', str(SPEC / 'fea_domain'),
                   '--density', str(fea / 'rho.npy'),
                   '--nodes', str(fea / 'nodes.npy'),
                   '--output', str(work / 'dc.npy'),
                   '--mesh-cache', str(SPEC / 'fea_domain/chair_035.msh'),
                   '--mesh-size', '.035', '--penal', '2', '--E0', '1.0',
                   '--load-magnitude', '800',
                   '--second-load-stl', str(SPEC / 'fea_domain_back/load.stl'),
                   '--second-load-magnitude', '200',
                   '--second-load-mode', 'y']
        env = {**os.environ, 'LOAD_MODE': '-z', 'BC_SURFACE_DIST': '1',
               'BC_DIST': '0.025', 'FEA_MAX_NODE_MAP_DISTANCE': '0.04',
               'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
               'MKL_NUM_THREADS': '1'}
        with (work / 'fea.log').open('w') as stream:
            process = subprocess.run(command, cwd=ROOT, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT)
        log = (work / 'fea.log').read_text(errors='replace')
        seat_count = re.search(r'Load:\s*(\d+)\s+nodes', log)
        back_count = re.search(r'Load2:\s*(\d+)\s+nodes', log)
        if process.returncode or not seat_count or not back_count or \
                'Load combination: simultaneous' not in log:
            raise RuntimeError(f'simultaneous FEA failed for {name}: {log[-1000:]}')
        combined = json.loads((work / 'dc_info.json').read_text())['compliance']
        result = {'id': name, 'stage': args.stage, 'mass_liters': row['mass_liters'],
                  'combined_compliance': combined,
                  'seat_only_compliance': row['fea_035']['seat']['compliance_proxy'],
                  'back_only_compliance': row['fea_035']['back']['compliance_proxy'],
                  'seat_nodes': int(seat_count.group(1)),
                  'back_nodes': int(back_count.group(1)),
                  'raw_preview': str(raw_preview),
                  'evaluated_preview': row['preview'],
                  'raw_obj': row['generated_mesh'],
                  'fea_log': str(work / 'fea.log'),
                  'strict_eligible': row['strict_eligible']}
        results.append(result)
        print(name, 'combined C', f'{combined:.6e}', 'mass L',
              f'{row["mass_liters"]:.3f}', flush=True)
    (OUT / f'{args.stage}_combined_evaluation.json').write_text(
        json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
