#!/usr/bin/env python3
"""Sequential 3D-archive-informed selection for fixed three-round chair pilot."""
from __future__ import annotations

import argparse
import json

import numpy as np

from make_chair_domain import ROOT

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
TEXT = BASE / 'text_latent_qd_2026-10-03'
ROOT_OUT = BASE / 'archive_feedback_loop_2026-10-03'
OLD = BASE / 'tapered_examples_2026-10-03/qd_round_01/archive.json'
PREVIOUS = BASE / 'archive_feedback_round_02_2026-10-03/result.json'


def select(round_index: int) -> None:
    assert 3 <= round_index <= 5
    protocol = json.loads((ROOT_OUT / 'protocol.json').read_text())
    out = ROOT_OUT / f'round_{round_index:02d}'
    out.mkdir(parents=True, exist_ok=True)
    archive_path = OLD if round_index == 3 else ROOT_OUT / f'round_{round_index-1:02d}/archive_state.json'
    archive = json.loads(archive_path.read_text())
    bank = json.loads((TEXT / 'bank.json').read_text())
    entries = bank['candidates']
    by_id = {r['id']: r for r in entries}
    vectors = np.load(TEXT / 'text_embeddings.npy')
    original = json.loads(OLD.read_text())['candidates']
    previous = json.loads(PREVIOUS.read_text())['rows']
    observed = [r for r in original if r['round'] == 0] + previous
    for i in range(3, round_index):
        observed += json.loads((ROOT_OUT / f'round_{i:02d}/result.json').read_text())['rows']
    used = {r['id'] for r in observed if r['id'] in by_id}
    used.update({'tapered__curved__standard', 'tapered__diagonal__standard'})
    # Each new observation, including a failed 3D candidate, updates the
    # descriptor transport estimate.  Feasibility remains a conservative
    # smoothed heuristic because the pilot is far too small to fit a model.
    valid = [r for r in observed if r['id'] in by_id]
    ov = np.array([vectors[by_id[r['id']]['embedding_index']] for r in valid])
    occupied = {tuple(map(int, key.split(','))) for key in archive['archive']}
    d0 = np.linspace(*archive['descriptor_ranges']['side_open_fraction'], 4)
    d1 = np.linspace(*archive['descriptor_ranges']['backrest_taper_ratio'], 4)
    residuals = np.random.default_rng(20261003).standard_normal((30000, 2))
    rows = []
    for entry in entries:
        if entry['id'] in used:
            continue
        vector = vectors[entry['embedding_index']]
        similarity = ov @ vector
        weights = np.exp((similarity - similarity.max()) / .035)
        weights *= np.array([1.30 if entry['back'] == by_id[r['id']]['back'] else 1.0
                             for r in valid])
        weights /= weights.sum()
        desc = np.array([[r['descriptor']['side_open_fraction'],
                          r['descriptor']['backrest_taper_ratio']] for r in valid])
        mu = weights @ desc
        sampled = mu + residuals * np.array([.055, .085])
        ix = np.searchsorted(d0, sampled[:, 0], side='right') - 1
        iy = np.searchsorted(d1, sampled[:, 1], side='right') - 1
        in_range = (ix >= 0) & (ix < 3) & (iy >= 0) & (iy < 3)
        empty = in_range & np.array([(int(a), int(b)) not in occupied for a, b in zip(ix, iy)])
        p_empty = float(empty.mean())
        p_range = float(in_range.mean())
        feasibility = []
        for r in valid:
            if r.get('archive_eligible'):
                feasibility.append(1.)
            elif r.get('id') == 'round__diagonal__standard':
                # Historical round-2 near pass; used only as selection evidence,
                # never admitted retroactively to the prospective archive.
                feasibility.append(.75)
            elif r.get('geometry_gate'):
                feasibility.append(.8)
            elif r.get('seat_bc', r.get('source_seat_bc_coverage', 0)) < .5:
                feasibility.append(.1)
            else:
                feasibility.append(.35)
        p_feasible = .25 + .5 * float(weights @ np.asarray(feasibility))
        if entry['back'] == 'tapered':
            p_feasible *= .7  # five previous tapered variants failed raw BC/repair gates
        novelty = float(np.min(1. - similarity))
        score = p_empty * p_feasible + .15 * novelty + .05 * p_range
        rows.append({'id': entry['id'], 'score': score, 'p_empty': p_empty,
                     'p_in_range': p_range, 'p_feasible_heuristic': p_feasible,
                     'novelty': novelty, 'predicted_descriptor': mu.tolist()})
    rows.sort(key=lambda r: (-r['score'], r['id']))
    qd_id = rows[0]['id']
    pool = [r['id'] for r in entries if r['id'] not in used | {qd_id}]
    random_id = str(np.random.default_rng(20261004 + round_index).choice(pool))
    selected = [{'method': 'archive_qd', **by_id[qd_id]},
                {'method': 'uniform_random', **by_id[random_id]}]
    payload = {'round': round_index, 'archive_before': str(archive_path),
               'occupied_before': sorted(map(list, occupied)),
               'observed_3d_ids': [r['id'] for r in observed],
               'excluded_ids': sorted(used),
               'selected': selected, 'scores': rows,
               'random_seed': 20261004 + round_index,
               'protocol': str(ROOT_OUT / 'protocol.json'),
               'note': 'Uncalibrated kernel/MC heuristic; selection frozen before generating this round images.'}
    (out / 'selection.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(round_index, qd_id, random_id, out / 'selection.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('round_index', type=int)
    args = parser.parse_args()
    select(args.round_index)
