"""Select round two using measured round-one FEA, mass and 3D phenotype."""
from __future__ import annotations

import json

import numpy as np

from select_chair_calibrated_image_qd_2026_10_03 import (
    OUT as PARENT, feature, prior_pairs, estimate,
)


OUT = PARENT / 'sparse_fea_loop_2026-10-03'
HISTORY = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)}
BASELINE_MASS = 27.10


def fea_prediction(candidate: dict, observed: list[dict]) -> dict:
    x = feature(candidate['image_metrics'])
    distance = np.array([np.linalg.norm(feature(r['image_metrics']) - x) for r in observed])
    weights = np.exp(-.5 * (distance / .28) ** 2)
    weights /= weights.sum()
    comp = float(sum(w * r['worst_compliance_ratio'] for w, r in zip(weights, observed)))
    mass = float(sum(w * r['mass_liters'] for w, r in zip(weights, observed)))
    return {'worst_compliance_ratio': comp, 'mass_liters': mass,
            'neighbors': [{'id': r['id'], 'weight': float(w),
                           'worst_compliance_ratio': r['worst_compliance_ratio'],
                           'mass_liters': r['mass_liters']}
                          for w, r in zip(weights, observed)]}


def main() -> None:
    first = json.loads((OUT / 'round_01/evaluation.json').read_text())['rows']
    if not all(r['strict_eligible'] and 'worst_compliance_ratio' in r for r in first):
        raise RuntimeError('round-one candidates must have valid two-load FEA before selection')
    screen = json.loads((PARENT / 'screening.json').read_text())
    used = {r['id'] for r in first}
    prior = prior_pairs(1) + [
        {'id': r['id'], 'image_metrics': r['image_metrics'], 'descriptor': r['descriptor'],
         'cell': r['cell'], 'valid': r['strict_eligible']} for r in first]
    qd = [r.copy() for r in screen if r['method'] == 'QD pool' and r['image_gate'] and r['id'] not in used]
    controls = [r.copy() for r in screen if r['method'] == 'control pool' and r['image_gate'] and r['id'] not in used]
    chosen = []
    for _ in range(2):
        for row in qd:
            transfer = estimate(row, prior, HISTORY)
            fea = fea_prediction(row, first)
            x = feature(row['image_metrics'])
            spread = min((np.linalg.norm(x - feature(c['image_metrics'])) for c in chosen), default=.5)
            different_image_cell = float(row['image_cell'] not in [c['image_cell'] for c in chosen])
            # Lower worst two-load compliance is better. Mass above the baseline
            # is penalized, but low mass cannot compensate for very weak stiffness.
            quality = np.clip((1.6 - fea['worst_compliance_ratio']) / .6, 0., 1.)
            quality *= min(1., BASELINE_MASS / fea['mass_liters'])
            novelty = transfer['neighbor_novelty']
            row['estimated_transfer'] = transfer
            row['estimated_fea'] = fea
            row['acquisition_components'] = {'quality': float(quality),
                                             'novelty': novelty,
                                             'image_spread': float(min(spread, 1.)),
                                             'distinct_image_cell': different_image_cell}
            row['acquisition'] = float(.35 * quality + .3 * transfer['p_valid_proxy'] * novelty +
                                       .15 * min(spread, 1.) + .2 * different_image_cell)
        best = max(qd, key=lambda r: (r['acquisition'], r['id']))
        chosen.append(best)
        qd.remove(best)
    rng = np.random.default_rng(20261005)
    control = [controls[int(i)] for i in rng.choice(len(controls), size=2, replace=False)]
    selected = chosen + control
    result = {'round': 2, 'status': 'frozen after round-one two-load FEA, before round-two 3D generation',
              'selected': selected, 'history_occupied_cells': sorted([list(x) for x in HISTORY]),
              'round_one_fea_source': str(OUT / 'round_01/evaluation.json'),
              'qd_score': '0.35 two-load compliance-and-mass quality + 0.30 valid novel-cell proxy + 0.15 image spread + 0.20 distinct image cell',
              'control': 'seeded uniform draw from remaining image-gate-passing control pool',
              'control_seed': 20261005,
              'limitations': ['four round-one FEA samples; empirical predictor is exploratory',
                              'no training or gradient of image-model latent',
                              'expected new-cell term is near zero because only two historically empty cells remain']}
    folder = OUT / 'round_02'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'selection.json').write_text(json.dumps(result, indent=2) + '\n')
    print([(r['id'], r['image_cell'], r.get('acquisition')) for r in selected])


if __name__ == '__main__':
    main()
