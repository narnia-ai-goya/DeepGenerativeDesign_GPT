#!/usr/bin/env python3
"""Top-view-only caliper: make dense BC connectivity survive thresholding.

The ordinary reachability loss declared success at occupancy 0.1.  That lets a
faint diffusion-path satisfy dense connectivity while disappearing at the MC
threshold.  These cases require a materially occupied path before sparse512.
"""
import copy
import json

from run_from_image import ROOT, DATA_ROOT, EXP, stage_gen, stage_post


CANDIDATES = {
    # Isolate dense reach threshold/weight: no sparse special preservation.
    'top_reach30_t030': {'cw': 30.0, 'reach_threshold': 0.30,
                         'reach_iters': 140, 'cw_warmup': 0.15},
    'top_reach50_t050': {'cw': 50.0, 'reach_threshold': 0.50,
                         'reach_iters': 160, 'cw_warmup': 0.15},
    # Prevent the now-real dense bridges from being eroded in sparse refinement.
    'top_reach50_latent_fix': {'cw': 50.0, 'reach_threshold': 0.50,
                                'reach_iters': 160, 'cw_warmup': 0.15,
                                'sp_topology_sdf_fix': True,
                                'sp_interior_latent_freeze': True,
                                'sp_interior_erode': 1},
}


def main():
    style = 'builtin_kagome_trial'
    source = DATA_ROOT / f'data/caliper/conditioning/{style}/v_top.png'
    if not source.exists():
        raise FileNotFoundError(source)
    base = json.loads((ROOT / 'configs/caliper.json').read_text())
    root = EXP / 'caliper_tearing_ablation_2026-09-17/caliper'
    for name, changes in CANDIDATES.items():
        cfg = copy.deepcopy(base)
        cfg['name'] = f'caliper_{name}'
        cfg['views'] = 'v_top'
        cfg['stages']['mesh'].update({'n_views': 1, 'fea_w': 0.0, 'sp_fea_w': 0.0})
        cfg['stages']['mesh'].update(changes)
        case = root / name
        case.mkdir(parents=True, exist_ok=True)
        cp = case / 'config.json'
        cp.write_text(json.dumps(cfg, indent=2) + '\n')
        c = {'config': str(cp)}
        wd = case / 'gen'
        print(f'\n########## top-only connectivity / {name} ##########', flush=True)
        if stage_gen('caliper', c, style, force=False, wd=wd, tag=f' [{name}]'):
            stage_post('caliper', c, style, force=False, wd=wd, tag=f' [{name}]')


if __name__ == '__main__':
    main()
