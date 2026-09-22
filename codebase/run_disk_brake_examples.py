#!/usr/bin/env python3
"""Generate diverse disk-brake caliper examples from calibrated six-view inputs.

This uses the caliper_ffff image bank, whose conditioning folders already follow the
current six-camera naming convention. FEA is disabled because this is an image/geometry
diversity pass; selected examples can be independently verified afterwards.
"""
import argparse
import json

from run_from_image import ROOT, DATA_ROOT, EXP, stage_gen, stage_post

DEFAULT = ','.join((
    'bionic_bone', 'de_kagome', 'de_diagrid_thick', 'de_rib',
    'de_slotrib', 'de_trilattice', 'de_vierendeel', 'de_warren',
))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--style', default=DEFAULT, help='comma-separated six-view styles')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    root = EXP / 'caliper_visual_examples'
    root.mkdir(parents=True, exist_ok=True)
    base = json.loads((ROOT / 'configs/caliper_ffff.json').read_text())
    base['stages']['mesh'].update({'fea_w': 0.0, 'sp_fea_w': 0.0})
    config_path = root / 'config_geometry_only.json'
    config_path.write_text(json.dumps(base, indent=2) + '\n')
    c = {'config': str(config_path)}
    for style in args.style.split(','):
        source = DATA_ROOT / 'data/caliper_ffff/conditioning' / style
        required = ('v00_front_lo.png', 'v02_right_lo.png', 'v04_back_lo.png', 'v06_left_lo.png', 'v_top.png', 'v_bottom.png')
        missing = [str(source / x) for x in required if not (source / x).exists()]
        if missing:
            print(f'[skip] {style}: missing calibrated views', flush=True)
            continue
        wd = root / style / 'gen'
        print(f'\n########## disk brake example / {style} ##########', flush=True)
        if stage_gen('caliper_ffff', c, style, args.force, wd=wd, tag=' [geometry-only]'):
            stage_post('caliper_ffff', c, style, args.force, wd=wd, tag=' [geometry-only]')


if __name__ == '__main__':
    main()
