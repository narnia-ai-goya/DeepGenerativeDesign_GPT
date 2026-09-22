#!/usr/bin/env python3
"""Generate the disk-brake caliper from only the calibrated top conditioning view."""
import copy
import json
from pathlib import Path

from run_from_image import ROOT, DATA_ROOT, EXP, stage_gen, stage_post


def main():
    style = 'builtin_kagome_trial'
    source = DATA_ROOT / f'data/caliper/conditioning/{style}/v_top.png'
    if not source.exists():
        raise FileNotFoundError(source)
    base = json.loads((ROOT / 'configs/caliper.json').read_text())
    cfg = copy.deepcopy(base)
    cfg['name'] = 'caliper_top_only_control'
    cfg['views'] = 'v_top'
    cfg['stages']['mesh'].update({
        'n_views': 1,
        # Geometry-only comparison with the six-view tear ablation.
        'fea_w': 0.0,
        'sp_fea_w': 0.0,
    })
    case = EXP / 'caliper_tearing_ablation_2026-09-17/caliper/top_only_control'
    case.mkdir(parents=True, exist_ok=True)
    config = case / 'config.json'
    config.write_text(json.dumps(cfg, indent=2) + '\n')
    c = {'config': str(config)}
    wd = case / 'gen'
    if stage_gen('caliper', c, style, force=False, wd=wd, tag=' [top-only control]'):
        stage_post('caliper', c, style, force=False, wd=wd, tag=' [top-only control]')


if __name__ == '__main__':
    main()
