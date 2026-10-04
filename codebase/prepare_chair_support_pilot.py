#!/usr/bin/env python3
"""Prepare an explicit leg-to-seat load-corridor ablation for chair dense generation."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/chair/pilot_2026-09-25'
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=float, default=5.0)
    args = parser.parse_args()
    weight = args.weight
    OUT = BASE / f'support_pw{weight:g}'
    OUT.mkdir(parents=True, exist_ok=True)
    grid = np.load(ROOT / 'data_real/chair/voxel.npz')
    o, p = grid['origin'], grid['pitch_xyz']
    xyz = [o[i] + (np.arange(64) + .5) * p[i] for i in range(3)]
    xx, yy, zz = np.meshgrid(*xyz, indexing='ij')
    mask = np.zeros((64, 64, 64), dtype=bool)
    for y in (-.18, .18):
        for x in (-.195, .195):
            mask |= ((xx-x)**2 + (yy-y)**2 <= .052**2) & (zz >= .045) & (zz <= .485)
    mask &= grid['bracket'] & ~grid['bc']
    if mask.sum() < 1000:
        raise RuntimeError('support corridor unexpectedly small')
    np.savez(OUT / 'load_corridors.npz', mask=mask)
    front = mask.any(axis=1).astype(np.uint8) * 255
    Image.fromarray(np.flipud(front.T)).resize((512, 512), Image.Resampling.NEAREST).save(
        OUT / 'load_corridors_front.png')
    cfg = json.loads((BASE / 'config_dense.json').read_text())
    cfg['name'] = f'chair_neutral_support_pw{weight:g}'
    cfg['stages']['mesh'].update(pw=weight,
                                 load_path_mask=str(OUT / 'load_corridors.npz'),
                                 save_dense_cache=str(OUT / 'dense_cache.npz'))
    (OUT / 'config_dense.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + '\n')
    full = json.loads(json.dumps(cfg))
    full['name'] = f'chair_neutral_support_pw{weight:g}_sparse'
    full['stages']['mesh'].update(skip_sparse=False,
                                 load_dense_cache=str(OUT / 'dense_cache.npz'),
                                 save_dense_cache=None)
    (OUT / 'config_sparse.json').write_text(json.dumps(full, indent=2, ensure_ascii=False) + '\n')
    (OUT / 'support_mask.json').write_text(json.dumps({
        'purpose': 'pilot functional support prior; not a fixed boundary condition',
        'radius_m': .052, 'z_range_m': [.045, .485],
        'corridor_voxels_excluding_BC': int(mask.sum()),
        'guidance_weight': weight}, indent=2) + '\n')
    print(OUT / 'config_dense.json')


if __name__ == '__main__':
    main()
