"""Chance-constrained probabilistic-cell acquisition for RC-BQD."""
from __future__ import annotations

import math

import numpy as np

from raqd_posterior import sobol_genomes
from raqd_v2_posterior import MixedExactGP


def quality(compliance):
    return 1 / (1 + compliance / .01)


def fit_models(observations):
    usable = [r for r in observations if r.get("valid") and r.get("realized_descriptors")
              and r.get("measured_design_volume_fraction") is not None
              and r.get("compliance_J") is not None]
    genomes = [r["genome"] for r in usable]
    targets = {
        "descriptor_0": [r["realized_descriptors"][0] for r in usable],
        "descriptor_1": [r["realized_descriptors"][1] for r in usable],
        "volume": [r["measured_design_volume_fraction"] for r in usable],
        "log_compliance": [math.log(r["compliance_J"]) for r in usable],
    }
    return usable, {name: MixedExactGP().fit(genomes, values) for name, values in targets.items()}


def archive_quality(observations, dims, ranges):
    elite = {}
    for row in observations:
        if not row.get("constraints_satisfied"):
            continue
        indices = []
        for value, cells, (lo, hi) in zip(row["realized_descriptors"], dims, ranges):
            if not lo <= value <= hi:
                indices = []; break
            indices.append(min(cells - 1, int((value-lo)/(hi-lo)*cells)))
        if indices:
            cell = tuple(indices); q = quality(row["compliance_J"])
            elite[cell] = max(elite.get(cell, -np.inf), q)
    return elite


def propose_rcbqd(observations, count, seed, descriptor_ranges, dims=(4, 4),
                  pool_size=8192, posterior_samples=256, min_feasibility=.20,
                  min_modal_cell_probability=.20):
    """Integrate cell membership, feasibility, and improvement over posterior draws."""
    usable, models = fit_models(observations)
    pool = sobol_genomes(pool_size, seed)
    predictions = {name: model.predict(pool, realization=True) for name, model in models.items()}
    elite = archive_quality(usable, dims, descriptor_ranges)
    rng = np.random.default_rng(seed)
    acquisition = np.zeros(pool_size); pof = np.zeros(pool_size)
    modal_cell_probability = np.zeros(pool_size); modal_cells = [None] * pool_size
    chunk = 512
    for start in range(0, pool_size, chunk):
        stop = min(pool_size, start + chunk); n = stop-start
        draws = {}
        for name, (mean, std) in predictions.items():
            draws[name] = rng.normal(mean[start:stop], std[start:stop], size=(posterior_samples, n))
        feasible = np.abs(draws["volume"] - .5) <= .025
        pof[start:stop] = feasible.mean(axis=0)
        utility = np.zeros((posterior_samples, n))
        cell_counts = [dict() for _ in range(n)]
        compliance = np.exp(np.clip(draws["log_compliance"], math.log(1e-5), math.log(1.)))
        q = quality(compliance)
        for sample in range(posterior_samples):
            for local in range(n):
                cell = []
                for value, cells, (lo, hi) in zip(
                        (draws["descriptor_0"][sample, local], draws["descriptor_1"][sample, local]),
                        dims, descriptor_ranges):
                    if not lo <= value <= hi:
                        cell = []; break
                    cell.append(min(cells-1, int((value-lo)/(hi-lo)*cells)))
                if not cell:
                    continue
                cell = tuple(cell)
                cell_counts[local][cell] = cell_counts[local].get(cell, 0) + 1
                if feasible[sample, local]:
                    utility[sample, local] = (1 + q[sample, local] if cell not in elite
                                              else max(0., q[sample, local] - elite[cell]))
        acquisition[start:stop] = utility.mean(axis=0)
        for local, counts in enumerate(cell_counts):
            if counts:
                cell, hits = max(counts.items(), key=lambda pair: pair[1])
                modal_cells[start+local] = cell
                modal_cell_probability[start+local] = hits / posterior_samples
    acquisition[pof < min_feasibility] = -np.inf
    acquisition[modal_cell_probability < min_modal_cell_probability] = -np.inf

    # Greedy diversity in normalized mixed genome space.
    from raqd_posterior import genome_to_unit
    units = np.asarray([[style/(len(set(g["style"] for g in pool))-1), *x]
                        for style, x in map(genome_to_unit, pool)])
    ranked = np.argsort(acquisition)[::-1]
    band = ranked[:min(512, len(ranked))]
    chosen = []
    chosen_cells = set()
    for _ in range(count):
        if not chosen:
            pick = int(ranked[0])
        else:
            distance = np.min(np.linalg.norm(units[band,None,:]-units[np.asarray(chosen)][None,:,:],axis=2),axis=1)
            score = acquisition[band] + .05 * distance
            score[np.isin(band, chosen)] = -np.inf
            # Fill distinct behavior cells before spending a batch slot twice in one cell.
            unused_cell = np.asarray([modal_cells[i] not in chosen_cells for i in band])
            if np.any(unused_cell & np.isfinite(score)):
                score[~unused_cell] = -np.inf
            pick = int(band[np.argmax(score)])
        if not np.isfinite(acquisition[pick]):
            raise RuntimeError("No candidate satisfies the minimum modeled feasibility probability")
        chosen.append(pick)
        chosen_cells.add(modal_cells[pick])
    jobs = [{"genome": pool[i], "target_cell": list(modal_cells[i]) if modal_cells[i] else None,
             "posterior": {"expected_archive_gain": float(acquisition[i]),
                           "feasibility_probability": float(pof[i]),
                           "modal_cell_probability": float(modal_cell_probability[i]),
                           "volume_mean": float(predictions["volume"][0][i]),
                           "volume_std": float(predictions["volume"][1][i])}}
            for i in chosen]
    diagnostics = {"usable_observations": len(usable), "pool_size": pool_size,
                   "posterior_samples": posterior_samples,
                   "minimum_feasibility_probability": min_feasibility,
                   "minimum_modal_cell_probability": min_modal_cell_probability,
                   "occupied_verified_cells": len(elite),
                   "models": {name: model.diagnostics() for name, model in models.items()},
                   "selection": "probabilistic cell expected improvement × joint chance constraint"}
    return jobs, diagnostics
