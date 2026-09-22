"""Propose target-directed RA-QD probes from the completed bracket development set."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from qd_archive import PARAMETERS
from raqd_core import cell_index

STYLES = ['builtin_kagome_trial', 'bionic_bone', 'de_trilattice']


def encode(style, genome):
    one_hot = [float(style == candidate) for candidate in STYLES]
    numeric = [(genome[name] - lo) / (hi - lo) for name, (lo, hi) in PARAMETERS.items()]
    return one_hot + numeric


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot', type=Path, required=True)
    ap.add_argument('--descriptor-study', type=Path, required=True)
    ap.add_argument('--volume-summary', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--count', type=int, default=3)
    args = ap.parse_args()
    pilot, study = args.pilot.resolve(), args.descriptor_study.resolve()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((pilot / 'summary.json').read_text())
    analysis = json.loads((study / 'analysis.json').read_text())
    feature_rows = json.loads((study / 'features.json').read_text())
    feature_by_id = {row['id']: row for row in feature_rows}
    selected = analysis['selected_pair']; names = selected['descriptors']; ranges = selected['ranges']

    training = [r for r in summary['results'] if r['valid']]
    x = np.asarray([encode(r['genome']['style'], r['genome']) for r in training])
    y = np.asarray([[feature_by_id[r['id']][name] for name in names] for r in training])
    model = ExtraTreesRegressor(n_estimators=256, min_samples_leaf=1,
                                max_features=1.0, random_state=20260913).fit(x, y)

    volume = json.loads(args.volume_summary.resolve().read_text())
    controlled = min((r for r in volume['results'] if r['id'] != 'baseline_disabled' and r['valid']),
                     key=lambda r: r['relative_volume_error'])
    baseline_features = feature_by_id['initial_00']
    correction = np.asarray([controlled['features'][name] - baseline_features[name] for name in names])

    rng = np.random.default_rng(20260913)
    n = 60000
    pool = []
    encoded = []
    for i in range(n):
        genome = {'style': STYLES[i % len(STYLES)]}
        for name, (lo, hi) in PARAMETERS.items():
            genome[name] = float(rng.uniform(lo, hi))
        pool.append(genome); encoded.append(encode(genome['style'], genome))
    encoded = np.asarray(encoded)
    per_tree = np.stack([tree.predict(encoded) for tree in model.estimators_])
    predictions = per_tree.mean(axis=0) + correction
    uncertainty = per_tree.std(axis=0)
    widths = np.asarray([hi-lo for lo, hi in ranges])
    normalized_uncertainty = np.linalg.norm(uncertainty / widths, axis=1)

    occupied = {cell_index([row[name] for name in names], (4, 4), ranges) for row in feature_rows}
    empty = sorted(set(itertools.product(range(4), repeat=2)) - occupied)
    candidates = []
    used = set()
    for target in empty:
        center = np.asarray([ranges[j][0] + (target[j]+.5)/4*(ranges[j][1]-ranges[j][0])
                             for j in range(2)])
        distance = np.linalg.norm((predictions-center)/widths, axis=1)
        acquisition = distance + .15 * normalized_uncertainty
        for index in np.argsort(acquisition):
            signature = tuple(round(pool[index][name], 3) if name != 'style' else pool[index][name]
                              for name in ['style', *PARAMETERS])
            if signature not in used:
                used.add(signature); break
        candidates.append({'target_cell': list(target), 'genome': pool[index],
                           'predicted_descriptors': predictions[index].tolist(),
                           'predicted_target_distance': float(distance[index]),
                           'prediction_uncertainty': uncertainty[index].tolist(),
                           'acquisition': float(acquisition[index])})
    candidates.sort(key=lambda row: row['acquisition'])
    chosen = candidates[:args.count]
    overrides = dict(controlled['overrides'])
    jobs = []
    for i, proposal in enumerate(chosen):
        jobs.append({'id': f'raqd_probe_{i:02d}', 'method': 'raqd_v0', 'round': 1,
                     'seed': 44000+i, 'parent': None, **proposal,
                     'volume_control': overrides})
    payload = {'status': 'prepared', 'method': 'RA-QD v0 target-directed probe',
               'descriptor_names': names, 'descriptor_ranges': ranges, 'dims': [4, 4],
               'development_samples': len(training), 'occupied_development_cells': sorted(map(list, occupied)),
               'empty_development_cells': list(map(list, empty)),
               'volume_control_source': str(args.volume_summary.resolve()),
               'volume_control_case': controlled['id'], 'volume_control': overrides,
               'global_control_descriptor_correction': correction.tolist(),
               'notes': ['The development meshes select descriptors and train the first proposer.',
                         'Extra-Trees predictions schedule full evaluations but never enter the verified archive.',
                         'Only final meshes determine realized cells and target hits.'],
               'jobs': jobs, 'all_empty_cell_candidates': candidates}
    save(out / 'proposal.json', payload)
    save(out / 'jobs.json', jobs)
    print(out / 'jobs.json')


if __name__ == '__main__':
    main()
