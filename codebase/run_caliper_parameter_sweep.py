#!/usr/bin/env python3
"""Controlled geometry-first parameter search for the disk-brake caliper.

The first pass turns FEA guidance off so shape regularity can be compared without
the expensive, currently weakly-scaled dense FEA term. Re-run selected candidates
with ``--fea on`` for mechanical verification.
"""
import argparse
import json
from pathlib import Path

from run_from_image import ROOT, DATA_ROOT, EXP, stage_gen, stage_post, stage_fea

CANDIDATES = {
    'control': {},
    'sign_only': {'sp_topology_sdf_fix': True},
    'guarded_light': {'sp_topology_sdf_fix': True, 'sp_rmin_w': 1.5, 'sp_thick_w': 1.0,
                      'sp_thick_target': 0.065, 'sp_lap_w': 60.0},
    'guarded_mid': {'sp_topology_sdf_fix': True, 'sp_rmin_w': 2.0, 'sp_thick_w': 2.0,
                    'sp_thick_target': 0.070, 'sp_lap_w': 75.0},
    'guarded_strong': {'sp_topology_sdf_fix': True, 'sp_rmin_w': 3.0, 'sp_thick_w': 3.0,
                       'sp_thick_target': 0.080, 'sp_lap_w': 100.0},
    'fidelity_mid': {'sp_topology_sdf_fix': True, 'sp_rmin_w': 2.0, 'sp_thick_w': 2.0,
                     'sp_thick_target': 0.070, 'sp_lap_w': 25.0},
    'regularized_legacy_sign': {'sp_topology_sdf_fix': False, 'sp_rmin_w': 2.0,
                                'sp_thick_w': 2.0, 'sp_thick_target': 0.070, 'sp_lap_w': 75.0},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--style', default='builtin_kagome_trial')
    parser.add_argument('--candidate', default='all', help='all or comma-separated candidate names')
    parser.add_argument('--fea', choices=('off', 'on'), default='off')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    chosen = list(CANDIDATES) if args.candidate == 'all' else args.candidate.split(',')
    unknown = set(chosen) - set(CANDIDATES)
    if unknown:
        parser.error(f'unknown candidate(s): {sorted(unknown)}')
    source = DATA_ROOT / 'data/caliper/conditioning' / args.style
    expected = ['v00_front_lo.png', 'v02_right_lo.png', 'v04_back_lo.png', 'v06_left_lo.png', 'v_top.png', 'v_bottom.png']
    missing = [str(source / x) for x in expected if not (source / x).exists()]
    if missing:
        parser.error('six calibrated views are required:\n' + '\n'.join(missing))
    root = EXP / 'caliper_parameter_sweep'
    root.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / 'configs/caliper.json').read_text())
    for name in chosen:
        config = json.loads(json.dumps(base))
        config['name'] = f'caliper_{name}'
        config['stages']['mesh'].update(CANDIDATES[name])
        if args.fea == 'off':
            config['stages']['mesh'].update({'fea_w': 0.0, 'sp_fea_w': 0.0})
        case = root / name
        config_path = case / 'config.json'
        case.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        c = {'config': str(config_path)}
        wd = case / 'gen'
        print(f'\n########## caliper parameter sweep / {name}/{args.fea} ##########', flush=True)
        if stage_gen('caliper', c, args.style, args.force, wd=wd, tag=f' [{name}/{args.fea}]'):
            if stage_post('caliper', c, args.style, args.force, wd=wd, tag=f' [{name}]'):
                if args.fea == 'on':
                    stage_fea('caliper', c, args.style, args.force, wd=wd, tag=f' [{name}]')


if __name__ == '__main__':
    main()
