#!/usr/bin/env python3
"""Prepare chair generation with a curved back path aligned to the new reference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from make_chair_domain import ROOT


OUT = ROOT / 'experiments/chair/aesthetic_reference_2026-09-27'
SOURCE = ROOT / 'experiments/chair/void_ablation_2026-09-27/rails_out100_backpath/config_dense.json'


def make_path() -> np.ndarray:
    grid = np.load(ROOT / 'data_real/chair/voxel.npz')
    origin, pitch = grid['origin'], grid['pitch_xyz']
    axes = [origin[i] + (np.arange(64)+.5)*pitch[i] for i in range(3)]
    x, y, z = np.meshgrid(*axes, indexing='ij')
    leg = np.load(ROOT / 'experiments/chair/pilot_2026-09-25/support_pw3.5/load_corridors.npz')['mask']
    t = np.clip((z-.465)/(.860-.465), 0, 1)
    x_upright = .205-.016*t
    y_upright = .211+.012*t-.006*np.sin(np.pi*t)
    uprights = ((np.abs(np.abs(x)-x_upright) < .047)
                & (np.abs(y-y_upright) < .045)
                & (z >= .450) & (z <= .875))
    unit_x = np.clip(x/.207, -1, 1)
    top_z = .852+.030*(1-unit_x**2)
    top = ((np.abs(x) < .220) & (np.abs(y-.217) < .040)
           & (np.abs(z-top_z) < .035))
    middle_z = .660+.012*(1-unit_x**2)
    middle = ((np.abs(x) < .220) & (np.abs(y-.217) < .034)
              & (np.abs(z-middle_z) < .027))
    back = (uprights | top | middle) & grid['bracket'] & ~grid['bc']
    mask = leg | back
    if back.sum() < 1000 or not np.all(mask <= grid['bracket']):
        raise RuntimeError('unexpected aesthetic back path')
    return mask


def main() -> None:
    mask = make_path()
    variants = {'archpath_cfg7': (7., 'v00_front_lo,v02_right_lo,v_top'),
                'archpath_cfg9': (9., 'v00_front_lo,v02_right_lo,v_top'),
                'archpath_4view': (7., 'v00_front_lo,v02_right_lo,v04_back_lo,v_top')}
    for name, (cfg_value, views) in variants.items():
        case = OUT / name
        case.mkdir(exist_ok=True)
        np.savez(case / 'arch_path.npz', mask=mask)
        cfg = json.loads(SOURCE.read_text())
        cfg['name'] = f'chair_aesthetic_{name}'
        cfg['views'] = views
        cfg['stages']['mesh'].update(
            views=views, n_views=len(views.split(',')), cfg=cfg_value,
            load_path_mask=str(case / 'arch_path.npz'),
            save_dense_cache=str(case / 'dense_cache.npz'))
        (case / 'config_dense.json').write_text(json.dumps(cfg, indent=2) + '\n')
    print(json.dumps({'path_voxels': int(mask.sum()),
                      'variants': list(variants)}, indent=2))


if __name__ == '__main__':
    main()
