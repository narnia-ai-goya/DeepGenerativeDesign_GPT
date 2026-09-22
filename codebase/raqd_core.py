"""Core data structures for realization-aware quality-diversity experiments."""
from __future__ import annotations

from collections import Counter
import math


def cell_index(descriptors, dims, ranges):
    """Return a grid cell without clipping out-of-calibration descriptors."""
    if len(descriptors) != len(dims) or len(dims) != len(ranges):
        raise ValueError('Descriptor dimension mismatch')
    cell = []
    for value, bins, bounds in zip(descriptors, dims, ranges):
        lo, hi = bounds
        if bins < 1 or hi <= lo:
            raise ValueError('Invalid archive specification')
        if not math.isfinite(value) or not lo <= value <= hi:
            return None
        cell.append(min(bins - 1, int((value - lo) / (hi - lo) * bins)))
    return tuple(cell)


def normalized_gap(target_cell, realized_descriptors, dims, ranges):
    """Euclidean gap from the requested cell center in normalized archive units."""
    if target_cell is None:
        return None
    if len(target_cell) != len(dims):
        raise ValueError('Target cell dimension mismatch')
    center = []
    normalized = []
    for target, value, bins, (lo, hi) in zip(target_cell, realized_descriptors, dims, ranges):
        if not 0 <= target < bins:
            raise ValueError('Target cell outside archive')
        center.append((target + .5) / bins)
        normalized.append((value - lo) / (hi - lo))
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(center, normalized)))


class RealizationArchive:
    """Paired intent/verified archive with an explicit target-to-result transition log."""

    def __init__(self, dims=(4, 4), ranges=((0., 1.), (0., 1.))):
        self.dims = tuple(dims)
        self.ranges = tuple(tuple(x) for x in ranges)
        self.attempts = []
        self.elites = {}
        self.transitions = Counter()

    def record(self, result):
        target = tuple(result['target_cell']) if result.get('target_cell') is not None else None
        descriptors = result.get('realized_descriptors')
        realized = cell_index(descriptors, self.dims, self.ranges) if descriptors is not None else None
        enriched = {**result,
                    'realized_cell': list(realized) if realized is not None else None,
                    'realization_gap': normalized_gap(target, descriptors, self.dims, self.ranges)
                    if target is not None and descriptors is not None else None}
        self.attempts.append(enriched)
        if target is not None:
            self.transitions[(target, realized)] += 1
        eligible = (result.get('valid', False) and result.get('constraints_satisfied', False)
                    and realized is not None and math.isfinite(result['compliance_J'])
                    and result['compliance_J'] > 0)
        inserted = False
        if eligible:
            previous = self.elites.get(realized)
            if previous is None or result['compliance_J'] < previous['compliance_J']:
                self.elites[realized] = enriched
                inserted = True
        return enriched, inserted

    def summary(self):
        targeted = [a for a in self.attempts if a.get('target_cell') is not None]
        hits = sum(a['realized_cell'] == a['target_cell'] for a in targeted)
        gaps = [a['realization_gap'] for a in targeted if a['realization_gap'] is not None]
        return {
            'attempts': len(self.attempts),
            'verified_elites': len(self.elites),
            'total_cells': math.prod(self.dims),
            'verified_coverage': len(self.elites) / math.prod(self.dims),
            'targeted_attempts': len(targeted),
            'target_hit_rate': hits / len(targeted) if targeted else None,
            'mean_realization_gap': sum(gaps) / len(gaps) if gaps else None,
        }
