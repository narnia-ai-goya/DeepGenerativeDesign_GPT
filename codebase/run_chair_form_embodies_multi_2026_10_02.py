#!/usr/bin/env python3
"""Matched seat-load FEM pilot on three chair images in the registered model frame."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from make_chair_domain import ROOT
from run_connectivity_qd_sampling import GENERATOR, PYTHON, generation_env

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
REGISTERED = BASE / 'registered_spec_2026-10-02'
OUT = BASE / 'form_embodies_multi_2026-10-02'
FENICS = Path('/home/goya/miniconda3/envs/fenics/bin/python')
FEA_SCRIPT = ROOT / 'codebase/code/fenics_fea_bracket.py'
CASES = {
    'open_arm': BASE / 'open_arm/input_lr162',
    'solid_side': BASE / 'solid_side/input_lr162',
    'diagonal_braced': BASE / 'backrest_bc_multi_examples_2026-09-29/diagonal_braced/input_lr162',
}


def transform_stl(source: Path, target: Path, rotation: np.ndarray, source_center: np.ndarray,
                  physical_center: np.ndarray, scale: float) -> None:
    mesh = trimesh.load(source, force='mesh')
    mesh.vertices = (mesh.vertices - physical_center) @ rotation / scale + source_center
    mesh.export(target)


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cal = json.loads((REGISTERED / 'calibration.json').read_text())['native_to_physical']
    source_center = np.asarray(cal['source_center_m'])
    physical_center = np.asarray(cal['physical_center_m'])
    scale = float(cal['uniform_scale'])
    rotation = Rotation.from_euler('x', cal['rotation_x_degrees'], degrees=True).as_matrix()
    domain = OUT / 'native_fea_domain'
    domain.mkdir(exist_ok=True)
    sources = {
        'original_DesignSpace.stl': ROOT / 'data_real/chair/original_DesignSpace.stl',
        'fixed.stl': ROOT / 'data_real/chair/fixed_remesh.stl',
        # Seat only: backrest is retained as geometric BC but needs its own force direction.
        'load.stl': ROOT / 'data_real/chair/load_remesh.stl',
    }
    for name, path in sources.items():
        transform_stl(path, domain / name, rotation, source_center, physical_center, scale)
    meta = {'physical_force': '-z', 'native_force': '-y',
            'guidance_force_N': 42300, 'verification_force_N': 800,
            'backrest_force': 'not applied',
            'calibration': str(REGISTERED / 'calibration.json'),
            'native_domain': str(domain), 'sources': {k: str(v) for k, v in sources.items()}}
    (OUT / 'preflight.json').write_text(json.dumps(meta, indent=2) + '\n')
    cache = OUT / 'native_fea_shared.msh'
    cmd = [str(FENICS), str(FEA_SCRIPT), '--build-mesh-only', '--domain-dir', str(domain),
           '--mesh-size', '0.035', '--mesh-cache', str(cache)]
    with (OUT / 'native_fea_build.log').open('w') as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f'FEM tet build failed: {OUT / "native_fea_build.log"}')
    print(json.dumps(meta, indent=2), flush=True)
    print(cache, flush=True)


def dense(case: str, fea: bool, gpu: int, aligned: bool = False) -> None:
    name = f'{case}_{"fea_on_aligned" if fea and aligned else "fea_on" if fea else "fea_off"}'
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((REGISTERED / 'strong/config.json').read_text())
    cfg['name'] = 'chair_form_embodies_' + name
    cfg['views'] = 'v00_front_lo'
    m = cfg['stages']['mesh']
    m.update({'load_dense_cache': None, 'save_dense_cache': str(out / 'dense_cache.npz'),
              'skip_sparse': True, 'fea_w': 5e-11 if fea else 0.0,
              'fea_every_n': 5, 'fea_warmup': 0.3,
              'fea_domain_dir': str(OUT / 'native_fea_domain'),
              'fea_mesh_cache': str(OUT / 'native_fea_shared.msh'),
              'fea_bracket_stl': str(OUT / 'native_fea_domain/original_DesignSpace.stl'),
              'load_mode': '-y'})
    config = out / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    cmd = [str(PYTHON), str(GENERATOR), '--config', str(config),
           '--target-dir', str(CASES[case]), '--out', str(out / 'generation')]
    env = generation_env(gpu)
    if fea and aligned:
        # The 15.8 mm voxel centers are as far as 25.8 mm from a few domain
        # boundary tet nodes; still under two voxel pitches after registration.
        env['FEA_MAX_NODE_MAP_DISTANCE'] = '0.03'
        env['FEA_LOAD_MAGNITUDE'] = '42300'
        env['FEA_DIAG'] = '1'
    with (out / 'generation.log').open('w') as log:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    log_text = (out / 'generation.log').read_text()
    fea_successes = log_text.count('FEA compliance @ step')
    fea_failures = log_text.count('[WARN] FEA call failed')
    (out / 'run.json').write_text(json.dumps({'command': cmd, 'exit_code': proc.returncode,
        'fea_w': m['fea_w'], 'image': str(CASES[case] / 'v00_front_lo.png'),
        'fea_success_log_count': fea_successes, 'fea_failures': fea_failures,
        'mapping_threshold_m': 0.03 if aligned and fea else None}, indent=2) + '\n')
    print(name, proc.returncode, out / 'generation/mesh_dense_raw.obj', flush=True)
    if proc.returncode:
        raise RuntimeError(f'{name} failed: {out / "generation.log"}')
    if fea and (not fea_successes or fea_failures):
        raise RuntimeError(f'{name} FEM calls did not all succeed: {out / "generation.log"}')


def sparse(case: str, fea: bool, gpu: int, dense_fea: bool | None = None) -> None:
    if dense_fea is None:
        dense_fea = fea
    name = (f'{case}_{"fea_on" if fea else "fea_off"}_sparse' if dense_fea == fea else
            f'{case}_dense_{"on" if dense_fea else "off"}_sparse_{"on" if fea else "off"}')
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    dense_name = f'{case}_{"fea_on_aligned" if dense_fea else "fea_off"}'
    cfg = json.loads((REGISTERED / 'strong_sparse_soft_halo1_fix13/config.json').read_text())
    cfg['name'] = 'chair_form_embodies_' + name
    cfg['views'] = 'v00_front_lo'
    m = cfg['stages']['mesh']
    m.update({'load_dense_cache': str(OUT / dense_name / 'dense_cache.npz'),
              'save_dense_cache': None, 'skip_sparse': False,
              'fea_w': 0.0, 'sp_fea_w': 3e-12 if fea else 0.0,
              'sp_fea_mode': 'manual', 'sp_fea_every': 5,
              'sp_fea_warmup': 0.5, 'sp_fea_step_size': 0.01,
              'fea_domain_dir': str(OUT / 'native_fea_domain'),
              'fea_mesh_cache': str(OUT / 'native_fea_shared.msh'),
              'fea_bracket_stl': str(OUT / 'native_fea_domain/original_DesignSpace.stl'),
              'load_mode': '-y'})
    config = out / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    cmd = [str(PYTHON), str(GENERATOR), '--config', str(config),
           '--target-dir', str(CASES[case]), '--out', str(out / 'generation')]
    env = generation_env(gpu)
    if fea:
        env.update({'FEA_MAX_NODE_MAP_DISTANCE': '0.03',
                    'FEA_LOAD_MAGNITUDE': '42300', 'FEA_DIAG': '1'})
    with (out / 'generation.log').open('w') as log:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    log_text = (out / 'generation.log').read_text()
    fea_successes = log_text.count('[sp FEA step') - log_text.count('] SKIP')
    fea_failures = log_text.count('[sp FEA] FAIL')
    (out / 'run.json').write_text(json.dumps({'command': cmd, 'exit_code': proc.returncode,
        'sp_fea_w': m['sp_fea_w'], 'image': str(CASES[case] / 'v00_front_lo.png'),
        'dense_fea': dense_fea, 'sparse_fea': fea,
        'fea_step_log_count': fea_successes, 'fea_failures': fea_failures}, indent=2) + '\n')
    print(name, proc.returncode, out / 'generation/mesh.obj', flush=True)
    if proc.returncode or (fea and (not fea_successes or fea_failures)):
        raise RuntimeError(f'{name} failed or FEM did not run: {out / "generation.log"}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'dense', 'sparse', 'cross'))
    parser.add_argument('--case', choices=CASES, default='open_arm')
    parser.add_argument('--fea', action='store_true')
    parser.add_argument('--aligned', action='store_true')
    parser.add_argument('--cross', choices=('dense_on_sparse_off', 'dense_off_sparse_on'))
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'dense':
        dense(args.case, args.fea, args.gpu, args.aligned)
    elif args.stage == 'sparse':
        sparse(args.case, args.fea, args.gpu)
    else:
        if not args.cross:
            parser.error('cross stage requires --cross')
        sparse(args.case, args.cross == 'dense_off_sparse_on', args.gpu,
               dense_fea=args.cross == 'dense_on_sparse_off')


if __name__ == '__main__':
    main()
