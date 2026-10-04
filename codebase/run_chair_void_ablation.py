#!/usr/bin/env python3
"""Isolate the causes of the filled under-seat wall in the chair pilot."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from run_connectivity_qd_sampling import GENERATOR, PYTHON, ROOT, generation_env


BASE = ROOT / 'experiments/chair/pilot_2026-09-25'
OUT = ROOT / 'experiments/chair/void_ablation_2026-09-27'
CASES = (
    'center_void', 'four_rails_loose', 'four_rails_tight',
    'four_rails_no_corridor', 'four_rails_out100',
    'volume_025', 'front_right_only',
)
BACKPATH_CASES = ('rails_out100_backpath', 'rails_out100_backpath_4view',
                  'rails_out100_backpath_pw2')


def make_mask(kind: str, source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    data = {k: np.array(v) for k, v in source.items()}
    if kind not in ('center_void', 'four_rails_loose', 'four_rails_tight'):
        return data
    origin, pitch = data['origin'], data['pitch_xyz']
    axes = [origin[i] + (np.arange(64) + .5) * pitch[i] for i in range(3)]
    x, y, z = np.meshgrid(*axes, indexing='ij')
    underseat = (z >= .080) & (z <= .420)
    if kind == 'center_void':
        remove = underseat & (np.abs(x) < .115) & (np.abs(y) < .115)
    else:
        radius = .130 if kind == 'four_rails_loose' else .095
        near_legs = np.zeros((64, 64, 64), bool)
        for cx in (-.195, .195):
            for cy in (-.180, .180):
                near_legs |= (x-cx)**2 + (y-cy)**2 <= radius**2
        remove = underseat & ~near_legs
    remove &= ~data['bc']
    data['bracket'] &= ~remove
    data['design'] = data['bracket'] & ~data['bc']
    assert np.all(data['bc'] <= data['bracket'])
    return data


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = np.load(ROOT / 'data_real/chair/voxel.npz')
    baseline = json.loads((BASE / 'support_pw3.5/config_dense.json').read_text())
    manifest = {}
    for name in CASES:
        case = OUT / name
        case.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(json.dumps(baseline))
        cfg['name'] = f'chair_void_ablation_{name}'
        cfg['stages']['mesh']['save_dense_cache'] = str(case / 'dense_cache.npz')
        if name == 'center_void':
            kind = 'center_void'
        elif name.startswith('four_rails'):
            kind = 'four_rails_tight' if name == 'four_rails_tight' else 'four_rails_loose'
        else:
            kind = 'original'
        if kind != 'original':
            data = make_mask(kind, source)
            path = case / 'voxel.npz'
            np.savez(path, **data)
            cfg['stages']['mesh']['bc_proper'] = str(path)
            cfg['stages']['mesh']['bracket_occ'] = str(path)
            removed = int((source['bracket'] & ~data['bracket']).sum())
        else:
            removed = 0
        if name == 'four_rails_no_corridor':
            cfg['stages']['mesh']['pw'] = 0.0
            cfg['stages']['mesh']['load_path_mask'] = None
        if name == 'four_rails_out100':
            cfg['stages']['mesh']['out_w'] = 100.0
        if name == 'volume_025':
            cfg['stages']['mesh']['vw'] = 10.0
            cfg['stages']['mesh']['vol_target'] = .25
        if name == 'front_right_only':
            cfg['views'] = 'v00_front_lo,v02_right_lo'
            cfg['stages']['mesh']['views'] = cfg['views']
            cfg['stages']['mesh']['n_views'] = 2
        cfg_path = case / 'config_dense.json'
        cfg_path.write_text(json.dumps(cfg, indent=2) + '\n')
        manifest[name] = {'case': str(case), 'config': str(cfg_path),
                          'mask_kind': kind, 'removed_envelope_voxels': removed,
                          'pw': cfg['stages']['mesh'].get('pw', 0),
                          'out_w': cfg['stages']['mesh']['out_w'],
                          'vw': cfg['stages']['mesh']['vw'],
                          'views': cfg['views']}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(OUT / 'manifest.json')


def run(name: str, gpu: int) -> None:
    case = OUT / name
    cfg_path = case / 'config_dense.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f'run prepare first: {cfg_path}')
    env = generation_env(gpu)
    command = [str(PYTHON), str(GENERATOR), '--config', str(cfg_path),
               '--target-dir', str(BASE / 'input'), '--out', str(case / 'dense')]
    with (case / 'dense.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    (case / 'run.json').write_text(json.dumps({
        'case': name, 'gpu': gpu, 'exit_code': result.returncode,
        'mesh': str(case / 'dense/mesh_dense.obj'),
        'cache': str(case / 'dense_cache.npz')}, indent=2) + '\n')
    if result.returncode:
        raise SystemExit(f'{name}: exit {result.returncode}; see {case / "dense.log"}')
    print(case / 'dense/mesh_dense.obj', flush=True)


def prepare_backpath() -> None:
    original = np.load(ROOT / 'data_real/chair/voxel.npz')
    origin, pitch = original['origin'], original['pitch_xyz']
    axes = [origin[i] + (np.arange(64) + .5) * pitch[i] for i in range(3)]
    x, y, z = np.meshgrid(*axes, indexing='ij')
    vertical = ((np.abs(np.abs(x) - .205) <= .045) &
                (np.abs(y - .210) <= .045) & (z >= .445) & (z <= .865))
    crossbar = ((np.abs(x) <= .240) & (np.abs(y - .210) <= .045)
                & (z >= .805) & (z <= .870))
    base_corridor = np.load(BASE / 'support_pw3.5/load_corridors.npz')['mask']
    manifest = {}
    for name in BACKPATH_CASES:
        case = OUT / name
        case.mkdir(parents=True, exist_ok=True)
        source_case = OUT / 'four_rails_out100'
        config = json.loads((source_case / 'config_dense.json').read_text())
        data = np.load(source_case / 'voxel.npz')
        voxel = case / 'voxel.npz'
        np.savez(voxel, **{k: np.array(v) for k, v in data.items()})
        backpath = (vertical | crossbar) & data['bracket'] & ~data['bc']
        corridor = base_corridor | backpath
        np.savez(case / 'load_corridors.npz', mask=corridor)
        config['name'] = f'chair_void_ablation_{name}'
        config['stages']['mesh'].update(
            bc_proper=str(voxel), bracket_occ=str(voxel),
            load_path_mask=str(case / 'load_corridors.npz'),
            save_dense_cache=str(case / 'dense_cache.npz'))
        if name.endswith('_4view'):
            config['views'] = 'v00_front_lo,v02_right_lo,v04_back_lo,v_top'
            config['stages']['mesh']['views'] = config['views']
            config['stages']['mesh']['n_views'] = 4
        if name.endswith('_pw2'):
            config['stages']['mesh']['pw'] = 2.0
        config_path = case / 'config_dense.json'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        manifest[name] = {'case': str(case), 'config': str(config_path),
                          'new_backpath_voxels': int(backpath.sum()),
                          'total_corridor_voxels': int(corridor.sum()),
                          'views': config['views'],
                          'pw': config['stages']['mesh']['pw']}
    (OUT / 'backpath_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(OUT / 'backpath_manifest.json')


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    sub.add_parser('prepare-backpath')
    run_parser = sub.add_parser('run')
    run_parser.add_argument('name', choices=CASES + BACKPATH_CASES)
    run_parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'prepare-backpath':
        prepare_backpath()
    else:
        run(args.name, args.gpu)


if __name__ == '__main__':
    main()
