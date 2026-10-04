"""Audit whether a thresholded tetra density field connects all chair BC patches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from run_chair_msh_simp_topopt_2026_10_04 import SPEC


def matched_threshold(rho: np.ndarray, volumes: np.ndarray,
                      target_liters: float, void_density: float = .001) -> float:
    target = target_liters / 1000
    needed = (target - void_density*float(volumes.sum())) / (1-void_density)
    order = np.argsort(-rho)
    count = int(np.searchsorted(np.cumsum(volumes[order]), needed))
    return float((rho[order[count-1]]+rho[order[count]])/2)


def audit(rho: np.ndarray, geometry: dict, threshold: float) -> dict:
    tets = geometry['tets']
    nodes = geometry['nodes']
    volumes = geometry['volumes']
    centers = geometry['centroids']
    solid = rho >= threshold
    selected = tets[solid]
    faces = np.sort(np.concatenate([selected[:, [0, 1, 2]],
                                    selected[:, [0, 1, 3]],
                                    selected[:, [0, 2, 3]],
                                    selected[:, [1, 2, 3]]]), axis=1)
    owner = np.tile(np.arange(len(selected)), 4)
    _, inverse = np.unique(faces, axis=0, return_inverse=True)
    order = np.argsort(inverse)
    matching = np.flatnonzero(inverse[order][1:] == inverse[order][:-1])
    start = owner[order[matching]]
    end = owner[order[matching+1]]
    graph = coo_matrix((np.ones(2*len(start)),
                        (np.r_[start, end], np.r_[end, start])),
                       shape=(len(selected), len(selected))).tocsr()
    ncomp, labels = connected_components(graph)
    comp_volume = np.bincount(labels, weights=volumes[solid])
    main = int(comp_volume.argmax())
    spec = np.load(SPEC / 'voxel.npz')
    at = np.clip(np.floor((centers[solid]-spec['origin']) /
                          spec['pitch_xyz']).astype(int), 0, 63)
    xyz = centers[solid]
    masks = {key: spec[key][at[:, 0], at[:, 1], at[:, 2]].astype(bool)
             for key in ('fix', 'load', 'back_load')}
    patches = {}
    for key in ('load', 'back_load'):
        m = masks[key]
        patches[key] = {'cells': int(m.sum()),
                        'main_fraction': float(np.mean(labels[m] == main)) if m.any() else 0.}
    for xsign in (-1, 1):
        for ysign in (-1, 1):
            key = f'foot_x{xsign:+d}_y{ysign:+d}'
            m = masks['fix'] & (xyz[:, 0]*xsign > 0) & (xyz[:, 1]*ysign > 0)
            patches[key] = {'cells': int(m.sum()),
                            'main_fraction': float(np.mean(labels[m] == main)) if m.any() else 0.}
    return {'threshold': threshold, 'solid_tetrahedra': int(solid.sum()),
            'binary_volume_liters': float(np.dot(np.where(solid, 1., .001), volumes)*1000),
            'components': int(ncomp),
            'main_volume_fraction': float(comp_volume.max()/comp_volume.sum()),
            'patches': patches,
            'all_bc_connected': all(x['main_fraction'] == 1. for x in patches.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rho', required=True)
    parser.add_argument('--geometry', required=True)
    parser.add_argument('--target-liters', type=float, default=25.442)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rho = np.load(args.rho)
    geo = np.load(args.geometry)
    cutoff = matched_threshold(rho, geo['volumes'], args.target_liters)
    result = audit(rho, geo, cutoff)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
