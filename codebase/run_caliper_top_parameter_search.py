#!/usr/bin/env python3
"""Broad top-view-only connectivity search for the disk-brake caliper.

Each candidate is geometry-only and has the identical v_top image, seed,
envelope, BC/cavity post-processing, and sampling schedule.  The search spans
the dense reachability criterion and sparse-stage drift controls, the two
mechanisms implicated by stage-wise component measurements.
"""
import argparse
import copy
import json

from run_from_image import ROOT, DATA_ROOT, EXP, stage_gen, stage_post


CANDIDATES = {
    # Dense reachability: criterion threshold / force trade-off.
    'r20_t020': {'cw': 20.0, 'reach_threshold': .20, 'reach_iters': 140, 'cw_warmup': .15},
    'r30_t040': {'cw': 30.0, 'reach_threshold': .40, 'reach_iters': 160, 'cw_warmup': .15},
    'r50_t030': {'cw': 50.0, 'reach_threshold': .30, 'reach_iters': 160, 'cw_warmup': .15},
    'r80_t050': {'cw': 80.0, 'reach_threshold': .50, 'reach_iters': 180, 'cw_warmup': .15},
    'r80_t070': {'cw': 80.0, 'reach_threshold': .70, 'reach_iters': 180, 'cw_warmup': .15},
    # Surface extraction: retain an established bridge without forcing all material thicker.
    'r50_t050_mc025': {'cw': 50.0, 'reach_threshold': .50, 'reach_iters': 160,
                        'cw_warmup': .15, 'mc_threshold': .25},
    'r50_t050_mc035': {'cw': 50.0, 'reach_threshold': .50, 'reach_iters': 160,
                        'cw_warmup': .15, 'mc_threshold': .35},
    # Sparse drift: reduce the late image-guidance peak that can tear a dense bridge.
    'r50_t050_spflat': {'cw': 50.0, 'reach_threshold': .50, 'reach_iters': 160,
                         'cw_warmup': .15, 'sp_guide_w': 4.0,
                         'sp_guide_w_peak': 20.0, 'sp_n_inner_late': 1},
    'r50_t050_spslow': {'cw': 50.0, 'reach_threshold': .50, 'reach_iters': 160,
                         'cw_warmup': .15, 'sp_guide_w': 5.0,
                         'sp_guide_w_peak': 30.0, 'sp_guide_lr': .002,
                         'sp_n_inner_late': 1},
    # Best available preservation mechanism, with two dense connectivity strengths.
    'r30_t040_latent': {'cw': 30.0, 'reach_threshold': .40, 'reach_iters': 160,
                         'cw_warmup': .15, 'sp_topology_sdf_fix': True,
                         'sp_interior_latent_freeze': True, 'sp_interior_erode': 1},
    'r80_t050_latent': {'cw': 80.0, 'reach_threshold': .50, 'reach_iters': 180,
                         'cw_warmup': .15, 'sp_topology_sdf_fix': True,
                         'sp_interior_latent_freeze': True, 'sp_interior_erode': 1},
    # Combined sparse protection without an envelope active-set gate (which fragmented earlier).
    'r50_t050_latent_spflat': {'cw': 50.0, 'reach_threshold': .50, 'reach_iters': 160,
                                'cw_warmup': .15, 'sp_topology_sdf_fix': True,
                                'sp_interior_latent_freeze': True, 'sp_interior_erode': 1,
                                'sp_guide_w': 4.0, 'sp_guide_w_peak': 20.0,
                                'sp_n_inner_late': 1},
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', default='all', help='all or comma-separated candidate keys')
    args = p.parse_args()
    chosen = list(CANDIDATES) if args.candidate == 'all' else args.candidate.split(',')
    invalid = sorted(set(chosen) - set(CANDIDATES))
    if invalid:
        p.error(f'unknown candidate(s): {invalid}')
    style = 'builtin_kagome_trial'
    if not (DATA_ROOT / f'data/caliper/conditioning/{style}/v_top.png').exists():
        raise FileNotFoundError('v_top.png')
    base = json.loads((ROOT / 'configs/caliper.json').read_text())
    root = EXP / 'caliper_top_parameter_search_2026-09-17' / 'caliper'
    for name in chosen:
        cfg = copy.deepcopy(base)
        cfg['name'] = f'caliper_top_search_{name}'
        cfg['views'] = 'v_top'
        cfg['stages']['mesh'].update({'n_views': 1, 'fea_w': 0.0, 'sp_fea_w': 0.0})
        cfg['stages']['mesh'].update(CANDIDATES[name])
        case = root / name
        case.mkdir(parents=True, exist_ok=True)
        cp = case / 'config.json'
        cp.write_text(json.dumps(cfg, indent=2) + '\n')
        c = {'config': str(cp)}
        wd = case / 'gen'
        print(f'\n########## top parameter search / {name} ##########', flush=True)
        if stage_gen('caliper', c, style, force=False, wd=wd, tag=f' [{name}]'):
            stage_post('caliper', c, style, force=False, wd=wd, tag=f' [{name}]')


if __name__ == '__main__':
    main()
