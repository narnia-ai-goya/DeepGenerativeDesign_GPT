"""Posterior sampling utilities for the controlled online RA-QD experiment.

The sampler uses a small random-Fourier-feature GP approximation with Matheron's
rule.  Styles use an equicorrelated categorical kernel and continuous controls
use an RBF kernel.  Evaluation seeds are treated as replicate noise, not inputs.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import qmc

from qd_archive import PARAMETERS
from raqd_core import cell_index

STYLES = ('builtin_kagome_trial', 'bionic_bone', 'de_trilattice',
          'gpt_bridge_loadpath')


def unit_to_genome(row):
    style = STYLES[min(len(STYLES) - 1, int(float(row[0]) * len(STYLES)))]
    genome = {'style': style}
    for value, (name, (lo, hi)) in zip(row[1:], PARAMETERS.items()):
        genome[name] = float(lo + value * (hi - lo))
    return genome


def genome_to_unit(genome):
    values = []
    for name, (lo, hi) in PARAMETERS.items():
        values.append((float(genome[name]) - lo) / (hi - lo))
    return STYLES.index(genome['style']), np.asarray(values, dtype=float)


def sobol_genomes(count, seed, skip=0):
    sampler = qmc.Sobol(d=1 + len(PARAMETERS), scramble=True, seed=seed)
    if skip:
        sampler.fast_forward(skip)
    return [unit_to_genome(row) for row in sampler.random(count)]


def _kernel(styles_a, x_a, styles_b, x_b, lengthscale, style_rho):
    sq = ((x_a[:, None, :] - x_b[None, :, :]) ** 2).sum(axis=2)
    continuous = np.exp(-.5 * sq / lengthscale**2)
    categorical = np.where(styles_a[:, None] == styles_b[None, :], 1., style_rho)
    return continuous * categorical


def _replicate_noise(y, groups):
    residuals = []
    for group in sorted(set(groups)):
        values = y[np.asarray(groups) == group]
        if len(values) > 1:
            residuals.extend(values - values.mean())
    estimate = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else .1
    return min(.5, max(.06, estimate))


def _select_hyperparameters(styles, x, y, noise):
    best = None
    for lengthscale in (.16, .25, .40, .65):
        for rho in (.05, .25, .50):
            k = _kernel(styles, x, styles, x, lengthscale, rho)
            k.flat[::len(k)+1] += noise**2 + 1e-6
            try:
                factor = cho_factor(k, lower=True, check_finite=False)
                alpha = cho_solve(factor, y, check_finite=False)
                logdet = 2 * np.log(np.diag(factor[0])).sum()
                nll = .5 * float(y @ alpha) + .5 * logdet
            except np.linalg.LinAlgError:
                continue
            if best is None or nll < best[0]:
                best = (nll, lengthscale, rho)
    if best is None:
        return .4, .25
    return best[1], best[2]


def _posterior_draw(train_genomes, raw_y, groups, pool_genomes, rng, rff=192):
    train = [genome_to_unit(g) for g in train_genomes]
    pool = [genome_to_unit(g) for g in pool_genomes]
    st = np.asarray([v[0] for v in train]); xt = np.asarray([v[1] for v in train])
    sp = np.asarray([v[0] for v in pool]); xp = np.asarray([v[1] for v in pool])
    center = float(np.mean(raw_y)); scale = max(float(np.std(raw_y)), 1e-8)
    y = (np.asarray(raw_y, dtype=float) - center) / scale
    noise = _replicate_noise(y, groups)
    lengthscale, rho = _select_hyperparameters(st, xt, y, noise)

    # A finite feature prior supplies a coherent function draw over the full pool.
    omega = rng.normal(size=(len(PARAMETERS), rff)) / lengthscale
    phase = rng.uniform(0, 2*np.pi, size=rff)
    phi_t = np.sqrt(2/rff) * np.cos(xt @ omega + phase)
    phi_p = np.sqrt(2/rff) * np.cos(xp @ omega + phase)
    style_cov = np.full((len(STYLES), len(STYLES)), rho)
    np.fill_diagonal(style_cov, 1.)
    style_root = np.linalg.cholesky(style_cov)
    phi_t = np.einsum('ni,nj->nij', style_root[st], phi_t).reshape(len(xt), -1)
    phi_p = np.einsum('ni,nj->nij', style_root[sp], phi_p).reshape(len(xp), -1)
    weights = rng.normal(size=phi_t.shape[1])
    prior_t, prior_p = phi_t @ weights, phi_p @ weights
    ktt = _kernel(st, xt, st, xt, lengthscale, rho)
    ktt.flat[::len(ktt)+1] += noise**2 + 1e-6
    kpt = _kernel(sp, xp, st, xt, lengthscale, rho)
    noisy_prior_t = prior_t + rng.normal(scale=noise, size=len(prior_t))
    correction = cho_solve(cho_factor(ktt, lower=True, check_finite=False),
                           y - noisy_prior_t, check_finite=False)
    sampled = prior_p + kpt @ correction
    return center + scale * sampled, {
        'lengthscale': lengthscale, 'style_rho': rho,
        'replicate_noise_standardized': noise, 'response_mean': center,
        'response_std': scale, 'rff': rff,
    }


def _quality(compliance):
    return 1 / (1 + compliance / .01)


def propose_batch(observations, count, seed, descriptor_ranges, dims=(4, 4),
                  pool_size=8192, epsilon_indices=()):
    """Draw a constrained QD function and greedily diversify its best candidates."""
    usable = [r for r in observations if r.get('realized_descriptors') is not None
              and r.get('measured_design_volume_fraction') is not None
              and r.get('compliance_J') is not None and r.get('valid')]
    if len(usable) < 3:
        raise ValueError('At least three complete observations are required')
    pool = sobol_genomes(pool_size, seed=seed)
    rng = np.random.default_rng(seed)
    groups = [r.get('replicate_group', r['id']) for r in usable]
    genomes = [r['genome'] for r in usable]
    targets = [np.asarray([r['realized_descriptors'][0] for r in usable]),
               np.asarray([r['realized_descriptors'][1] for r in usable]),
               np.asarray([r['measured_design_volume_fraction'] for r in usable]),
               np.log(np.asarray([r['compliance_J'] for r in usable]))]
    draws, fits = [], []
    for target in targets:
        draw, fit = _posterior_draw(genomes, target, groups, pool, rng)
        draws.append(draw); fits.append(fit)
    d0, d1, volume, log_compliance = draws
    compliance = np.exp(np.clip(log_compliance, math.log(1e-5), math.log(1.)))

    elites = {}
    for row in usable:
        if not row.get('constraints_satisfied'):
            continue
        cell = cell_index(row['realized_descriptors'], dims, descriptor_ranges)
        if cell is not None:
            elites[cell] = min(elites.get(cell, float('inf')), row['compliance_J'])
    cells, acquisition = [], np.full(pool_size, -1e6)
    for i, descriptor in enumerate(zip(d0, d1)):
        cell = cell_index(descriptor, dims, descriptor_ranges)
        cells.append(cell)
        if cell is None or abs(volume[i] - .5) > .025:
            continue
        q = _quality(compliance[i])
        acquisition[i] = 1. + q if cell not in elites else max(0., q - _quality(elites[cell]))
    # A slight uncertainty-independent jitter makes ties deterministic by seed.
    acquisition += rng.uniform(0, 1e-8, size=pool_size)
    unit = np.asarray([[s/2, *x] for s, x in map(genome_to_unit, pool)])
    ranked = np.argsort(acquisition)[::-1]
    candidate_band = ranked[:min(512, len(ranked))]
    chosen = []
    for local_index in range(count):
        if local_index in epsilon_indices:
            available = [i for i in ranked if i not in chosen]
            pick = int(rng.choice(available))
        elif not chosen:
            pick = int(ranked[0])
        else:
            distances = np.min(np.linalg.norm(unit[candidate_band, None, :] -
                                              unit[np.asarray(chosen)][None, :, :], axis=2), axis=1)
            utility = acquisition[candidate_band] + .18 * distances
            utility[np.isin(candidate_band, chosen)] = -np.inf
            pick = int(candidate_band[np.argmax(utility)])
        chosen.append(pick)
    jobs = []
    for i, index in enumerate(chosen):
        jobs.append({'genome': pool[index],
                     'target_cell': list(cells[index]) if cells[index] is not None else None,
                     'posterior_sample': {
                         'descriptors': [float(d0[index]), float(d1[index])],
                         'volume_fraction': float(volume[index]),
                         'compliance_J': float(compliance[index]),
                         'acquisition': float(acquisition[index]),
                         'epsilon_random': i in epsilon_indices}})
    return jobs, {'usable_observations': len(usable), 'pool_size': pool_size,
                  'occupied_verified_cells': len(elites), 'fits': fits,
                  'seed': seed, 'selection': 'constrained Thompson draw + greedy maximin'}
