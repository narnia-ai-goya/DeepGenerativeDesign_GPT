"""Connect disconnected chair BC patches through density-guided tetra paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from audit_chair_msh_topopt_connectivity_2026_10_04 import audit, matched_threshold
from run_chair_msh_simp_topopt_2026_10_04 import SPEC


def adjacency(tets: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, csr_matrix]:
    n = len(tets)
    faces = np.sort(np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
                                    tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), axis=1)
    owner = np.tile(np.arange(n), 4)
    # Lexicographic sort is sufficient to identify shared tetra faces.
    order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    sorted_faces = faces[order]
    paired = np.flatnonzero(np.all(sorted_faces[1:] == sorted_faces[:-1], axis=1))
    u = owner[order[paired]]
    v = owner[order[paired+1]]
    base = csr_matrix((np.ones(len(u)*2), (np.r_[u, v], np.r_[v, u])), shape=(n, n))
    return u, v, base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rho', required=True)
    parser.add_argument('--geometry', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--target-liters', type=float, default=25.442)
    parser.add_argument('--tube-radius-mm', type=float, default=18.)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rho = np.load(args.rho)
    z = np.load(args.geometry)
    geometry = {k: z[k] for k in z.files}
    centers, volumes, tets = geometry['centroids'], geometry['volumes'], geometry['tets']
    threshold = matched_threshold(rho, volumes, args.target_liters)
    solid = rho >= threshold
    u, v, graph = adjacency(tets, centers)
    ids = np.flatnonzero(solid)
    ncomp, labels = connected_components(graph[ids][:, ids])
    size = np.bincount(labels, weights=volumes[ids])
    main = ids[labels == size.argmax()]
    in_main = np.zeros(len(rho), bool)
    in_main[main] = True
    spec = np.load(SPEC / 'voxel.npz')
    at = np.clip(np.floor((centers-spec['origin']) / spec['pitch_xyz']).astype(int), 0, 63)
    fix = spec['fix'][at[:, 0], at[:, 1], at[:, 2]].astype(bool)
    bc = spec['bc'][at[:, 0], at[:, 1], at[:, 2]].astype(bool)
    length = np.linalg.norm(centers[u]-centers[v], axis=1)
    # Shortest paths favor existing high-density cells; a small geometric cost
    # discourages unnecessary detours. Costs are directed into the destination.
    uv = length * (.05 + (1-rho[v])**2)
    vu = length * (.05 + (1-rho[u])**2)
    weighted = csr_matrix((np.r_[uv, vu], (np.r_[u, v], np.r_[v, u])),
                          shape=graph.shape)
    distance, predecessor, _ = dijkstra(weighted, directed=True, indices=main,
                                         min_only=True, return_predecessors=True)
    paths = []
    for xs in (-1, 1):
        for ys in (-1, 1):
            foot = fix & (centers[:, 0]*xs > 0) & (centers[:, 1]*ys > 0)
            candidates = np.flatnonzero(foot & ~in_main)
            if not len(candidates):
                continue
            destination = int(candidates[np.argmin(distance[candidates])])
            route = [destination]
            while predecessor[route[-1]] >= 0:
                route.append(int(predecessor[route[-1]]))
            if not in_main[route[-1]]:
                raise RuntimeError(f'foot ({xs},{ys}) has no route to main component')
            paths.append({'foot': [xs, ys], 'route_cells': len(route),
                          'length_mm': float(np.linalg.norm(np.diff(centers[route], axis=0), axis=1).sum()*1000),
                          'route': route})
    if not paths:
        print('All feet already connected; no repair needed')
        return
    path_cells = np.unique(np.concatenate([p['route'] for p in paths]))
    tube = np.zeros(len(rho), bool)
    tree = cKDTree(centers)
    for neighbors in tree.query_ball_point(centers[path_cells], args.tube_radius_mm/1000):
        tube[neighbors] = True
    passive = bc | tube
    if volumes[passive].sum() >= args.target_liters/1000:
        raise RuntimeError('required BC+connector mass exceeds target volume')
    repaired = rho.copy()
    repaired[passive] = 1.0
    free = ~passive
    # Restore the exact target volume by reducing free densities while leaving
    # the path and all BC patches intact.
    lo, hi = 0.0, 1.0
    for _ in range(70):
        alpha = (lo+hi)/2
        trial = np.maximum(.001, rho[free]*alpha)
        total = volumes[passive].sum()+np.dot(trial, volumes[free])
        if total > args.target_liters/1000:
            hi = alpha
        else:
            lo = alpha
    repaired[free] = np.maximum(.001, rho[free]*lo)
    np.save(out / 'rho_connected_seed.npy', repaired)
    np.save(out / 'passive_connector_mask.npy', passive)
    before = audit(rho, geometry, threshold)
    after = audit(repaired, geometry,
                  matched_threshold(repaired, volumes, args.target_liters))
    summary = {'input_rho': args.rho, 'target_liters': args.target_liters,
               'paths': [{k: v for k, v in path.items() if k != 'route'} for path in paths],
               'path_tetrahedra': int(len(path_cells)), 'tube_tetrahedra': int(tube.sum()),
               'connector_volume_liters': float(volumes[tube].sum()*1000),
               'passive_volume_liters': float(volumes[passive].sum()*1000),
               'before': before, 'after_seed': after}
    (out / 'connector_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
