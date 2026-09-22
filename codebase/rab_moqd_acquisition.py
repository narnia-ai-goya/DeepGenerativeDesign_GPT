"""Monte Carlo cell-conditioned hypervolume acquisition for RAB-MOQD."""
from __future__ import annotations

import math

import numpy as np

from raqd_posterior import STYLES, genome_to_unit, sobol_genomes
from raqd_v2_posterior import MixedExactGP


def cell_of(descriptors, ranges=None, dims=(4, 4), thresholds=None):
    if thresholds is not None:
        return tuple(int(np.searchsorted(edge, value, side="right"))
                     for value, edge in zip(descriptors, thresholds))
    cell = []
    for value, (lo, hi), count in zip(descriptors, ranges, dims):
        if value < lo or value > hi:
            return None
        cell.append(min(count-1, int((value-lo)/(hi-lo)*count)))
    return tuple(cell)


def nondominated(points):
    result = []
    for point in points:
        if any(all(other[i] <= point[i] for i in range(2)) and other != point
               for other in points):
            continue
        result.append(tuple(point))
    return result


def hypervolume_2d(points, reference=(1.0, 1.0)):
    points = nondominated([tuple(np.minimum(point, reference)) for point in points
                           if point[0] < reference[0] and point[1] < reference[1]])
    previous_y = reference[1]
    total = 0.0
    for x, y in sorted(points):
        if y < previous_y:
            total += (reference[0]-x) * (previous_y-y)
            previous_y = y
    return float(total)


def objective_point(compliance, volume, objective_bounds):
    (c_lo, c_hi), (v_lo, v_hi) = objective_bounds
    return ((compliance-c_lo)/(c_hi-c_lo), (volume-v_lo)/(v_hi-v_lo))


def archive_fronts(observations, descriptor_ranges, objective_bounds, dims=(4, 4),
                   cell_thresholds=None):
    fronts = {}
    for row in observations:
        if not row.get("valid"):
            continue
        cell = cell_of(row["realized_descriptors"], descriptor_ranges, dims,
                       thresholds=cell_thresholds)
        if cell is None:
            continue
        point = objective_point(row["compliance_J"],
                                row["measured_design_volume_fraction"], objective_bounds)
        fronts.setdefault(cell, []).append(point)
    return {cell: nondominated(points) for cell, points in fronts.items()}


def propose_rab_moqd(observations, count, seed, descriptor_ranges, objective_bounds,
                     dims=(4, 4), pool_size=8192, posterior_samples=128,
                     sample_posterior=True, joint_standardized_residuals=None,
                     cell_thresholds=None):
    usable = [row for row in observations if row.get("valid") and row.get("realized_descriptors")]
    genomes = [row["genome"] for row in usable]
    targets = {
        "descriptor_0": [row["realized_descriptors"][0] for row in usable],
        "descriptor_1": [row["realized_descriptors"][1] for row in usable],
        "volume": [row["measured_design_volume_fraction"] for row in usable],
        "log_compliance": [math.log(row["compliance_J"]) for row in usable],
    }
    models = {name: MixedExactGP().fit(genomes, values) for name, values in targets.items()}
    pool = sobol_genomes(pool_size, seed, skip=128)
    prediction = {name: model.predict(pool, realization=True) for name, model in models.items()}
    fronts = archive_fronts(usable, descriptor_ranges, objective_bounds, dims,
                            cell_thresholds=cell_thresholds)
    base_hv = {cell: hypervolume_2d(points) for cell, points in fronts.items()}

    style_counts = {style: [0, 0] for style in STYLES}
    for row in observations:
        style_counts[row["genome"]["style"]][1] += 1
        style_counts[row["genome"]["style"]][0] += int(bool(row.get("valid")))
    validity_probability = {style: (valid+2)/(total+3)
                            for style, (valid, total) in style_counts.items()}

    rng = np.random.default_rng(seed)
    acquisition = np.zeros(pool_size)
    modal_probability = np.zeros(pool_size)
    modal_cells = [None] * pool_size
    mean_hvi = np.zeros(pool_size)
    for start in range(0, pool_size, 256):
        stop = min(pool_size, start+256); width = stop-start
        if joint_standardized_residuals is not None:
            residuals = np.asarray(joint_standardized_residuals, dtype=float)
            sampled_rows = rng.integers(0, len(residuals), size=(posterior_samples, width))
            draws = {name: mean[start:stop][None, :] + std[start:stop][None, :]
                     * residuals[sampled_rows, output_index]
                     for output_index, (name, (mean, std)) in enumerate(prediction.items())}
        elif sample_posterior:
            draws = {name: rng.normal(mean[start:stop], std[start:stop],
                                      size=(posterior_samples, width))
                     for name, (mean, std) in prediction.items()}
        else:
            draws = {name: np.broadcast_to(mean[start:stop], (posterior_samples, width))
                     for name, (mean, _) in prediction.items()}
        gain = np.zeros((posterior_samples, width))
        counts = [dict() for _ in range(width)]
        compliance = np.exp(np.clip(draws["log_compliance"], math.log(1e-5), math.log(1.0)))
        for sample in range(posterior_samples):
            for local in range(width):
                cell = cell_of((draws["descriptor_0"][sample, local],
                                draws["descriptor_1"][sample, local]), descriptor_ranges, dims,
                               thresholds=cell_thresholds)
                if cell is None:
                    continue
                counts[local][cell] = counts[local].get(cell, 0)+1
                point = objective_point(compliance[sample, local],
                                        draws["volume"][sample, local], objective_bounds)
                gain[sample, local] = max(0.0, hypervolume_2d(fronts.get(cell, [])+[point])
                                          - base_hv.get(cell, 0.0))
        mean_hvi[start:stop] = gain.mean(axis=0)
        for local, cell_counts in enumerate(counts):
            if cell_counts:
                cell, hits = max(cell_counts.items(), key=lambda pair: pair[1])
                modal_cells[start+local] = cell
                modal_probability[start+local] = hits/posterior_samples
        acquisition[start:stop] = mean_hvi[start:stop] * np.array([
            validity_probability[genome["style"]] for genome in pool[start:stop]])

    units = np.asarray([[style/(len(STYLES)-1), *values]
                        for style, values in map(genome_to_unit, pool)])
    ranked = np.argsort(acquisition)[::-1]
    band = ranked[:min(768, len(ranked))]
    chosen, chosen_cells = [], set()
    for _ in range(count):
        score = acquisition[band].copy()
        if chosen:
            distance = np.min(np.linalg.norm(
                units[band, None, :]-units[np.asarray(chosen)][None, :, :], axis=2), axis=1)
            score += .02*distance
        score[np.isin(band, chosen)] = -np.inf
        distinct = np.array([modal_cells[index] not in chosen_cells for index in band])
        if np.any(distinct & np.isfinite(score)): score[~distinct] = -np.inf
        pick = int(band[np.argmax(score)])
        if not np.isfinite(score.max()) or acquisition[pick] <= 0:
            raise RuntimeError("No candidate has positive expected cell-wise hypervolume improvement")
        chosen.append(pick); chosen_cells.add(modal_cells[pick])

    jobs = []
    for index in chosen:
        jobs.append({
            "genome": pool[index], "target_cell": list(modal_cells[index]),
            "posterior": {
                "expected_hypervolume_improvement": float(mean_hvi[index]),
                "validity_probability": validity_probability[pool[index]["style"]],
                "modal_cell_probability": float(modal_probability[index]),
                **{f"{name}_{which}": float(values[part][index])
                   for name, values in prediction.items()
                   for which, part in (("mean", 0), ("std", 1))},
            },
        })
    diagnostics = {
        "usable_observations": len(usable), "pool_size": pool_size,
        "posterior_samples": posterior_samples,
        "posterior_sampling": sample_posterior,
        "joint_residual_bootstrap": joint_standardized_residuals is not None,
        "cell_assignment": "open_thresholds" if cell_thresholds is not None else "closed_ranges",
        "cell_thresholds": cell_thresholds,
        "occupied_cells": len(fronts),
        "cell_pareto_sizes": {str(cell): len(front) for cell, front in fronts.items()},
        "validity_probability_by_style": validity_probability,
        "models": {name: model.diagnostics() for name, model in models.items()},
        "selection": (("group-conformal joint-residual" if joint_standardized_residuals is not None
                       else "expected" if sample_posterior else "plug-in posterior-mean")
                      + " cell-conditioned 2D hypervolume improvement x empirical-Beta validity probability"),
    }
    return jobs, diagnostics
