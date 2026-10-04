#!/usr/bin/env python3
"""Freeze one archive-aware chair prompt and one budget-matched random control.

This is a small-data acquisition heuristic, not a trained 3D surrogate.  The
fixed prompt bank, existing 3D measurements, geometry failures and archive
occupancy are the only inputs.  No new result is read while selecting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from make_chair_domain import ROOT

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
TEXT = BASE / 'text_latent_qd_2026-10-03'
OLD = BASE / 'tapered_examples_2026-10-03/qd_round_01'
OUT = BASE / 'archive_feedback_round_02_2026-10-03'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bank = json.loads((TEXT / 'bank.json').read_text())
    archive = json.loads((OLD / 'archive.json').read_text())
    entries = bank['candidates']
    vectors = np.load(TEXT / 'text_embeddings.npy')
    by_id = {row['id']: row for row in entries}
    used = {row['id'] for row in archive['candidates'] if row['id'] in by_id}
    used.update({'tapered__curved__standard', 'tapered__diagonal__standard'})
    observed = [row for row in archive['candidates'] if row['round'] == 0]
    obs_vectors = np.array([vectors[by_id[row['id']]['embedding_index']] for row in observed])
    occupied = {tuple(map(int, cell.split(','))) for cell in archive['archive']}
    ranges = archive['descriptor_ranges']
    d0 = np.linspace(*ranges['side_open_fraction'], 4)
    d1 = np.linspace(*ranges['backrest_taper_ratio'], 4)
    # MC integrates only descriptor uncertainty. The bandwidths are deliberately
    # broad because four 3D observations cannot support calibrated posteriors.
    rng_mc = np.random.default_rng(20261003)
    residuals = rng_mc.standard_normal((30000, 2))
    scored = []
    for row in entries:
        if row['id'] in used:
            continue
        v = vectors[row['embedding_index']]
        similarity = obs_vectors @ v
        weights = np.exp((similarity - similarity.max()) / .035)
        weights *= np.array([
            1.30 if row['back'] == by_id[o['id']]['back'] else 1.0 for o in observed
        ])
        weights /= weights.sum()
        descriptor = np.array([[o['descriptor']['side_open_fraction'],
                                o['descriptor']['backrest_taper_ratio']] for o in observed])
        mu = weights @ descriptor
        samples = mu + residuals * np.array([.055, .085])
        i = np.searchsorted(d0, samples[:, 0], side='right') - 1
        j = np.searchsorted(d1, samples[:, 1], side='right') - 1
        in_range = (i >= 0) & (i < 3) & (j >= 0) & (j < 3)
        empty = in_range & np.array([(int(a), int(b)) not in occupied for a, b in zip(i, j)])
        p_empty = float(empty.mean())
        p_in_range = float(in_range.mean())
        # Failed tapered variants are direct evidence about realization risk,
        # but they do not prove every tapered prompt will fail.
        geometry_prior = .40 if row['back'] == 'tapered' else .80
        if row['back'] == 'flared':
            geometry_prior *= .75  # observed flared case exceeded descriptor range
        if row['gap'] == 'raised':
            geometry_prior *= .92
        novelty = float(np.min(1.0 - similarity))
        score = p_empty * geometry_prior + .15 * novelty + .05 * p_in_range
        scored.append({'id': row['id'], 'score': score, 'p_empty_cell': p_empty,
                       'p_descriptor_in_range': p_in_range,
                       'geometry_prior': geometry_prior,
                       'novelty_cosine_distance': novelty,
                       'predicted_descriptor': mu.tolist(),
                       'text_cell': row['text_cell']})
    scored.sort(key=lambda r: (-r['score'], r['id']))
    qd = scored[0]['id']
    control_pool = [row['id'] for row in entries if row['id'] not in used | {qd}]
    random_id = str(np.random.default_rng(20261004).choice(control_pool))
    selected = [{'method': 'archive_qd', **by_id[qd]},
                {'method': 'uniform_random', **by_id[random_id]}]
    payload = {
        'round': 2, 'input_archive': str(OLD / 'archive.json'),
        'input_text_bank': str(TEXT / 'bank.json'),
        'selection_rule': 'MC empty-3D-cell probability x geometry prior + CLIP novelty + in-range probability',
        'disclaimer': 'Heuristic descriptor uncertainty from four observed 3D cases; not calibrated Bayesian posterior.',
        'occupied_before': sorted(map(list, occupied)),
        'excluded_prompt_ids': sorted(used),
        'scores': scored, 'selected': selected,
        'random_seed': 20261004,
    }
    (OUT / 'selection.json').write_text(json.dumps(payload, indent=2) + '\n')
    print('archive_qd', qd, 'score', round(scored[0]['score'], 3))
    print('uniform_random', random_id)
    print(OUT / 'selection.json')


if __name__ == '__main__':
    main()
