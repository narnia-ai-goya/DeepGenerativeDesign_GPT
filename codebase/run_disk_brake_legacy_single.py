#!/usr/bin/env python3
"""Run legacy disk-brake conditioning styles that only supply v06_left_lo.

The legacy images are deliberately treated as one-view inputs. They are not relabelled
as front/top/back views, which would claim a camera correspondence they do not have.
"""
import argparse
import json
from pathlib import Path

from run_from_image import ROOT, DATA_ROOT, EXP, stage_gen, stage_post, stage_fea

DEFAULT = 'de_Ktruss,de_Xbrace,de_prattframe,de_trilattice,de_warren'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--style', default=DEFAULT, help='comma-separated legacy style names')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    styles = args.style.split(',')
    run_root = EXP / 'caliper_legacy_single'
    run_root.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / 'configs/caliper.json').read_text())
    config['name'] = 'disk_brake_legacy_single_view'
    config['views'] = 'v06_left_lo'
    config['stages']['mesh']['n_views'] = 1
    config_path = run_root / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    c = {'config': str(config_path)}
    for style in styles:
        source = DATA_ROOT / 'data/caliper/conditioning' / style / 'v06_left_lo.png'
        if not source.exists():
            print(f'[skip] {style}: missing {source}')
            continue
        wd = run_root / style / 'gen'
        print(f'\n########## caliper legacy single / {style} ##########', flush=True)
        if stage_gen('caliper', c, style, args.force, wd=wd, tag=' [legacy-single]'):
            if stage_post('caliper', c, style, args.force, wd=wd, tag=' [legacy-single]'):
                stage_fea('caliper', c, style, args.force, wd=wd, tag=' [legacy-single]')


if __name__ == '__main__':
    main()
