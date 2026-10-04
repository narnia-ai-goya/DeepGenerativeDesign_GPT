"""Frozen two-round image screen and non-parametric chair QD selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_chair_image_qd_2026_10_03 import normalized, metrics
from prepare_chair_text_reasoned_pilot_2026_10_03 import front_aperture
from make_chair_domain import ROOT


BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03'
OLD = BASE / 'text_reasoned_front_axes_2026-10-03/long_qd_fea_2026-10-03'
REFERENCE = BASE / 'single_view_image_qd_2026-10-03/images/baseline.png'
INITIAL = ((0, 0), (0, 2), (1, 0), (1, 1), (2, 1))
RANGES = ((0., .45), (.5, 1.2))


def cell(values: tuple[float, float] | list[float]) -> list[int] | None:
    if not all(lo <= x <= hi for x, (lo, hi) in zip(values, RANGES)):
        return None
    return [min(2, int((x - lo) / (hi - lo) * 3)) for x, (lo, hi) in zip(values, RANGES)]


def feature(image_metrics: dict) -> np.ndarray:
    return np.array([image_metrics['front_arm_aperture_fraction'] / .45,
                     (image_metrics['upper_span_ratio'] - .5) / .7], float)


def screen() -> None:
    manifest = json.loads((OUT / 'image_manifest.json').read_text())
    _, reference = normalized(REFERENCE)
    rows = []
    for item in manifest['rows']:
        _, mask = normalized(Path(item['image']))
        values = metrics(mask, reference)
        values['front_arm_aperture_fraction'] = front_aperture(mask)
        image_cell = cell((values['front_arm_aperture_fraction'], values['upper_span_ratio']))
        # Conservative front-view proxy for the expanded (+15.8 mm) design box.
        # 3D side-view clipping is tested only after generation.
        outside = mask.copy()
        outside[32:511, 110:402] = False
        projected_outside = float(outside.sum() / max(1, mask.sum()))
        reasons = []
        if image_cell is None: reasons.append('image descriptor outside frozen range')
        if values['projected_interface_retention'] < .9: reasons.append('projected seat/feet interface')
        if values['projected_back_load_retention'] < .8: reasons.append('projected back-load patch')
        if projected_outside > .01: reasons.append('projected envelope')
        rows.append({'id': item['id'], 'method': item['method'], 'image': item['image'],
                     'image_metrics': values, 'image_cell': image_cell,
                     'projected_outside_fraction': projected_outside,
                     'image_gate': not reasons, 'image_reasons': reasons})
    (OUT / 'screening.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('image gate', {method: (sum(r['image_gate'] for r in rows if r['method'] == method),
                                  sum(r['method'] == method for r in rows))
                         for method in ('QD pool', 'control pool')})


def prior_pairs(round_number: int) -> list[dict]:
    result = json.loads((OLD / 'result.json').read_text())
    feedback = json.loads((OLD / 'feedback_round_02/result.json').read_text())
    prior = result['rows'] + feedback['rows']
    protected = json.loads((BASE / 'text_reasoned_front_axes_2026-10-03/protected_parts_qd/result.json').read_text())
    _, reference = normalized(REFERENCE)
    pairs = []
    for row in prior + protected['rows']:
        im = row.get('image_metrics')
        if im is None:
            _, mask = normalized(Path(row['source_image']))
            im = metrics(mask, reference)
            im['front_arm_aperture_fraction'] = front_aperture(mask)
        pairs.append({'id': row['id'], 'image_metrics': im, 'descriptor': row['descriptor'],
                      'cell': row['cell'], 'valid': bool(row.get('strict_eligible',
                                                               row.get('geometry_gate', False)))})
    if round_number == 2:
        first = json.loads((OUT / 'round_01/evaluation.json').read_text())
        pairs.extend({'id': row['id'], 'image_metrics': row['image_metrics'],
                      'descriptor': row['descriptor'], 'cell': row['cell'],
                      'valid': row['strict_eligible']} for row in first['rows'])
    return pairs


def estimate(row: dict, prior: list[dict], occupied: set[tuple[int, int]]) -> dict:
    x = feature(row['image_metrics'])
    vectors = np.array([feature(p['image_metrics']) for p in prior])
    distance = np.linalg.norm(vectors - x, axis=1)
    neighbors = np.argsort(distance)[:5]
    weights = np.exp(-.5 * (distance[neighbors] / .32) ** 2)
    weights = weights / weights.sum()
    chosen = [prior[int(i)] for i in neighbors]
    p_valid = float((.5 + sum(w * p['valid'] for w, p in zip(weights, chosen))) / 1.5)
    novelty = float(sum(w for w, p in zip(weights, chosen)
                        if p['valid'] and p['cell'] is not None and tuple(p['cell']) not in occupied))
    y = np.array([[p['descriptor']['front_arm_aperture_fraction'],
                   p['descriptor']['front_upper_span_ratio']] for p in chosen])
    predicted = np.sum(y * weights[:, None], axis=0)
    return {'predicted_descriptor': predicted.tolist(), 'predicted_cell': cell(predicted),
            'p_valid_proxy': p_valid, 'neighbor_novelty': novelty,
            'nearest_prior': [{'id': p['id'], 'distance': round(float(distance[int(i)]), 4),
                               'cell': p['cell'], 'valid': p['valid']}
                              for i, p in zip(neighbors, chosen)]}


def select(round_number: int) -> None:
    rows = json.loads((OUT / 'screening.json').read_text())
    prior = prior_pairs(round_number)
    occupied = set(INITIAL)
    previous_ids = set()
    if round_number == 2:
        first = json.loads((OUT / 'round_01/evaluation.json').read_text())
        occupied |= {tuple(row['cell']) for row in first['rows']
                     if row['strict_eligible'] and row['cell'] is not None}
        previous_ids = {row['id'] for row in first['rows']}
    qd = [r.copy() for r in rows if r['method'] == 'QD pool' and r['image_gate'] and r['id'] not in previous_ids]
    control = [r.copy() for r in rows if r['method'] == 'control pool' and r['image_gate'] and r['id'] not in previous_ids]
    selected_qd = []
    for _ in range(2):
        for candidate in qd:
            candidate['estimate'] = estimate(candidate, prior, occupied)
            x = feature(candidate['image_metrics'])
            spread = min((np.linalg.norm(x - feature(r['image_metrics'])) for r in selected_qd), default=.5)
            distinct_cell = float(candidate['image_cell'] not in [r['image_cell'] for r in selected_qd])
            candidate['acquisition'] = (candidate['estimate']['p_valid_proxy'] *
                                        (.15 + candidate['estimate']['neighbor_novelty']) +
                                        .25 * min(spread, 1.) +
                                        .4 * distinct_cell +
                                        .05 * candidate['image_metrics']['projected_interface_retention'])
        best = max(qd, key=lambda r: (r['acquisition'], r['id']))
        selected_qd.append(best)
        qd.remove(best)
    # Independent, pre-seeded uniform draw among the image-gate-passing control pool.
    rng = np.random.default_rng(20261003 + round_number)
    picks = rng.choice(len(control), size=2, replace=False)
    selected_control = [control[int(i)] for i in picks]
    selected = selected_qd + selected_control
    result = {'round': round_number, 'status': 'selection frozen before new 3D generation',
              'initial_occupied': sorted([list(c) for c in occupied]),
              'selected': selected, 'remaining_qd': len(qd),
              'remaining_control': len(control) - len(selected_control),
              'method': 'top-five distance-weighted empirical transfer; QD expected novelty + image spread + distinct image-cell preference; uniform random eligible control',
              'calibration_ids': [p['id'] for p in prior], 'control_seed': 20261003 + round_number}
    dest = OUT / f'round_{round_number:02d}'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'selection.json').write_text(json.dumps(result, indent=2) + '\n')
    print('round', round_number, [(r['id'], r['method'], r['image_cell']) for r in selected])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('screen', 'select'))
    parser.add_argument('--round', type=int, choices=(1, 2))
    args = parser.parse_args()
    if args.action == 'screen': screen()
    else:
        if args.round is None: parser.error('select requires --round')
        select(args.round)


if __name__ == '__main__':
    main()
