"""Small dependency-free MAP-Elites archive. Objectives are minimized."""
import math
import random

PARAMETERS = {'cfg': (3.0, 9.0), 'sp_cfg': (3.0, 7.0),
              'sp_guide_w_peak': (20.0, 80.0)}


class GridArchive:
    def __init__(self, dims=(4, 4), ranges=((0., 1.), (0., 1.))):
        self.dims, self.ranges, self.elites = tuple(dims), tuple(ranges), {}
        if len(dims) != len(ranges) or any(n < 1 for n in dims):
            raise ValueError('Invalid archive dimensions')
        if any(hi <= lo for lo, hi in ranges):
            raise ValueError('Invalid descriptor ranges')

    def index(self, descriptors):
        if len(descriptors) != len(self.dims):
            raise ValueError('Descriptor dimension mismatch')
        index = []
        for x, n, (lo, hi) in zip(descriptors, self.dims, self.ranges):
            if not math.isfinite(x) or not lo <= x <= hi:
                return None
            index.append(min(n - 1, int((x - lo) / (hi - lo) * n)))
        return tuple(index)

    def add(self, result):
        if not result.get('valid'):
            return False
        objective = result['compliance_J']
        if not math.isfinite(objective) or objective <= 0:
            return False
        cell = self.index(result['descriptors'])
        if cell is None:
            return False
        old = self.elites.get(cell)
        if old is None or objective < old['compliance_J']:
            self.elites[cell] = result
            return True
        return False

    def summary(self):
        return {'occupied_cells': len(self.elites),
                'total_cells': math.prod(self.dims),
                'coverage': len(self.elites) / math.prod(self.dims),
                'best_compliance_J': min((e['compliance_J'] for e in self.elites.values()), default=None),
                # Positive bounded quality, fixed 0.01 J scale for this load case.
                'qd_score': sum(1 / (1 + e['compliance_J'] / .01) for e in self.elites.values()),
                'elites': [{'cell': list(k), **v} for k, v in sorted(self.elites.items())]}


def propose(rng: random.Random, bank, archive=None):
    parents = list(archive.elites.values()) if archive else []
    if not parents:
        return {'style': rng.choice(bank),
                **{k: rng.uniform(lo, hi) for k, (lo, hi) in PARAMETERS.items()}}, None
    parent = rng.choice(parents)
    genome = dict(parent['genome'])
    if rng.random() < .35:
        genome['style'] = rng.choice(bank)
    for k, (lo, hi) in PARAMETERS.items():
        genome[k] = min(hi, max(lo, genome[k] + rng.gauss(0, .20 * (hi - lo))))
    return genome, parent['id']


def descriptors_from_occupancy(occupied, upper):
    """Both arrays refer to the SAME fixed design-only CAD sample points."""
    import numpy as np
    occupied, upper = np.asarray(occupied, bool), np.asarray(upper, bool)
    if occupied.shape != upper.shape or occupied.ndim != 1 or not len(occupied):
        raise ValueError('Nonempty matching one-dimensional samples required')
    if not occupied.any():
        raise ValueError('Empty design material')
    return [float(occupied.mean()), float((occupied & upper).sum() / occupied.sum())]
