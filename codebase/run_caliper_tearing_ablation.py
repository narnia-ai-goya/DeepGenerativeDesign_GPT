#!/usr/bin/env python3
"""Ablate where thin tears enter the six-view disk-brake pipeline.

The earlier parameter sweep changed only sparse regularizer strength.  This
experiment changes the *mechanism* that can make a disconnected/thin result:

  - `active_gate`: do not let sparse latents grow outside the real envelope,
    eliminating a later boolean cut through a member.
  - `thin_gate`: retain that hard envelope but give thin members and BC
    attachments one 64-grid-cell of temporary growth room before the true
    high-resolution envelope clip.
  - `bc_anchor`: keep the raw STL BC solid while sparse guidance runs.  The
    final mesh still uses the functional piston cavities from post-processing.
  - `interior_freeze` / `latent_freeze`: preserve dense-stage cores through
    sparse refinement, respectively at decoded SDF and latent levels.
  - `combined_stable`: combines the least invasive guards. Wedge trimming is
    evaluated separately because an exact mesh Minkowski operation is costly.

All candidates are geometry-only (FEA weights zero), use exactly the same six
conditioning images, seed, dense/sparse schedule, and post/boolean pipeline.
This makes a visible difference attributable to the tear-control mechanism.
"""
import argparse
import copy
import json
from pathlib import Path

from run_from_image import ROOT, DATA_ROOT, EXP, stage_gen, stage_post


CANDIDATES = {
    # This is deliberately a generation repeat, not a copied historical result:
    # all cases below have the same code revision and are reproducible together.
    'control_repeat': {},
    # Primary hypothesis: members are born outside a curved MC envelope then cut.
    'active_gate': {'restrict_active_to_envelope': True},
    # One cell of local attachment/thin-feature room, but no global envelope bloat.
    'thin_gate': {
        'restrict_active_to_envelope': True,
        'restrict_dilate_thin': 1.0,
        'thin_target_vox512': 10.0,
        'restrict_dilate_peg': 2.0,
    },
    # Raw (0 mm) BC is enough to make interfaces stable without merging piston bores.
    'bc_anchor': {
        'restrict_active_to_envelope': True,
        'force_bc_solid': True,
        'force_bc_dilate_mm': 0.0,
        'bc_inloop_only': True,
    },
    # Dense cores are held at sparse decoded-field level.
    'interior_freeze': {
        'restrict_active_to_envelope': True,
        'sp_interior_freeze': True,
        'sp_interior_erode': 1,
    },
    # Stronger version: restore the sparse latent at core positions every step.
    'latent_freeze': {
        'restrict_active_to_envelope': True,
        'sp_interior_latent_freeze': True,
        'sp_interior_erode': 1,
    },
    # Intended delivered recipe: local gate + BC support + dense-core preservation.
    # Wedge trimming is measured separately because its exact Minkowski operation is costly.
    'combined_stable': {
        'restrict_active_to_envelope': True,
        'restrict_dilate_thin': 1.0,
        'thin_target_vox512': 10.0,
        'restrict_dilate_peg': 2.0,
        'force_bc_solid': True,
        'force_bc_dilate_mm': 0.0,
        'bc_inloop_only': True,
        'sp_interior_freeze': True,
        'sp_interior_erode': 1,
    },
}


def config_for(base, name):
    cfg = copy.deepcopy(base)
    cfg['name'] = f'caliper_tearing_{name}'
    cfg['stages']['mesh'].update(CANDIDATES[name])
    # A geometry ablation must not hide the cause behind a different FEA gradient.
    cfg['stages']['mesh'].update({'fea_w': 0.0, 'sp_fea_w': 0.0})
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--style', default='builtin_kagome_trial')
    p.add_argument('--candidate', default='all', help='all or comma-separated names')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    chosen = list(CANDIDATES) if args.candidate == 'all' else args.candidate.split(',')
    bad = sorted(set(chosen) - set(CANDIDATES))
    if bad:
        p.error(f'unknown candidates: {bad}')
    source = DATA_ROOT / 'data/caliper/conditioning' / args.style
    views = ('v00_front_lo.png', 'v02_right_lo.png', 'v04_back_lo.png',
             'v06_left_lo.png', 'v_top.png', 'v_bottom.png')
    missing = [str(source / v) for v in views if not (source / v).exists()]
    if missing:
        p.error('missing calibrated six-view inputs:\n' + '\n'.join(missing))

    root = EXP / 'caliper_tearing_ablation_2026-09-17' / 'caliper'
    base = json.loads((ROOT / 'configs/caliper.json').read_text())
    for name in chosen:
        case = root / name
        case.mkdir(parents=True, exist_ok=True)
        cfg_path = case / 'config.json'
        cfg_path.write_text(json.dumps(config_for(base, name), indent=2) + '\n')
        c = {'config': str(cfg_path)}
        wd = case / 'gen'
        print(f'\n########## caliper tearing ablation / {name} ##########', flush=True)
        if stage_gen('caliper', c, args.style, args.force, wd=wd, tag=f' [{name}]'):
            stage_post('caliper', c, args.style, args.force, wd=wd, tag=f' [{name}]')


if __name__ == '__main__':
    main()
