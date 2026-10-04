"""Threshold, retain the BC-connected component, and regrow to matched mass."""
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import meshio
import numpy as np
from scipy.sparse.csgraph import connected_components

from audit_chair_msh_topopt_connectivity_2026_10_04 import audit, matched_threshold
from repair_chair_msh_topopt_connections_2026_10_04 import adjacency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--rho', required=True)
    parser.add_argument('--geometry', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--target-liters', type=float, default=25.442)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rho = np.load(args.rho)
    g = np.load(args.geometry)
    geo = {k: g[k] for k in g.files}
    volumes = geo['volumes']
    threshold = matched_threshold(rho, volumes, args.target_liters)
    selected = rho >= threshold
    _, _, graph = adjacency(geo['tets'], geo['centroids'])
    ids = np.flatnonzero(selected)
    _, labels = connected_components(graph[ids][:, ids])
    by_volume = np.bincount(labels, weights=volumes[ids])
    keep = np.zeros(len(rho), bool)
    keep[ids[labels == by_volume.argmax()]] = True
    target_solid = ((args.target_liters/1000 - .001*float(volumes.sum()))/.999)
    current = float(volumes[keep].sum())
    if current > target_solid:
        raise RuntimeError('main component already exceeds target mass')
    frontier = []
    seen = keep.copy()
    def offer(node: int) -> None:
        start, stop = graph.indptr[node], graph.indptr[node+1]
        for neighbor in graph.indices[start:stop]:
            if not seen[neighbor]:
                seen[neighbor] = True
                heapq.heappush(frontier, (-float(rho[neighbor]), int(neighbor)))
    for node in np.flatnonzero(keep):
        offer(int(node))
    added = 0
    while current < target_solid and frontier:
        _, node = heapq.heappop(frontier)
        keep[node] = True
        current += float(volumes[node])
        added += 1
        offer(node)
    if current < target_solid:
        raise RuntimeError('could not regrow connected material to target mass')
    binary = np.where(keep, 1., .001)
    np.save(out / 'rho_connected_binary.npy', binary)
    meshio.write(out / 'connected_binary.vtu',
                 meshio.Mesh(geo['nodes'],
                             [meshio.CellBlock('tetra', geo['tets'])],
                             cell_data={'density': [binary]}))
    result = audit(binary, geo, .5)
    result.update({'source_rho': args.rho,
                   'source_threshold': threshold,
                   'added_tetrahedra_to_match_volume': added,
                   'target_liters': args.target_liters,
                   'connected_binary_vtu': str(out / 'connected_binary.vtu')})
    (out / 'connected_binary_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    if not result['all_bc_connected'] or result['components'] != 1:
        raise RuntimeError('connected binary validation failed')


if __name__ == '__main__':
    main()
