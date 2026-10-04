"""Matched-volume SIMP/OC baseline directly on the chair envelope tetra mesh."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import meshio
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SPEC = (BASE / 'text_reasoned_front_axes_2026-10-03/long_qd_fea_2026-10-03'
        / 'envelope_plus16_spec_2026-10-03')
OUT = (BASE / 'text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03'
       / 'llm_text_proposal_round_03_2026-10-03'
       / 'simultaneous_load_dense_sparse_2026-10-04'
       / 'diagnostics/msh_simp_topopt_2026-10-04')
FENICS = Path('/home/goya/miniconda3/envs/fenics/bin/python')
SOLVER = ROOT / 'codebase/code/fenics_fea_bracket.py'
MESH = SPEC / 'fea_domain/chair_035.msh'
TARGET_LITERS = 25.442


def run_fea(iteration: int, rho: np.ndarray, *, geometry: bool = False) -> dict:
    folder = OUT / f'iter_{iteration:02d}'
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / 'rho.npy', rho)
    command = [str(FENICS), str(SOLVER), '--domain-dir', str(SPEC / 'fea_domain'),
               '--density', str(folder / 'rho.npy'), '--output', str(folder / 'dc.npy'),
               '--mesh-cache', str(MESH), '--mesh-size', '.035', '--penal', '2',
               '--E0', '1', '--load-magnitude', '800',
               '--second-load-stl', str(SPEC / 'fea_domain_back/load.stl'),
               '--second-load-magnitude', '200', '--second-load-mode', 'y']
    if geometry:
        command += ['--geom-out', str(OUT / 'fenics_geometry.npz')]
    env = {**os.environ, 'LOAD_MODE': '-z', 'BC_SURFACE_DIST': '1',
           'BC_DIST': '.025', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    with (folder / 'fea.log').open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'FEA iteration {iteration} failed: '
                           f'{(folder / "fea.log").read_text()[-1200:]}')
    info = json.loads((folder / 'dc_info.json').read_text())
    log = (folder / 'fea.log').read_text()
    if ('Load combination: simultaneous' not in log or
        'cell-based ρ mode' not in log):
        raise RuntimeError(f'FEA iteration {iteration} did not use simultaneous '
                           'loads and direct tetra-element densities')
    return info


def filter_matrix(centroids: np.ndarray, radius: float) -> tuple[csr_matrix, np.ndarray]:
    pairs = cKDTree(centroids).query_pairs(radius, output_type='ndarray')
    distance = np.linalg.norm(centroids[pairs[:, 0]] - centroids[pairs[:, 1]], axis=1)
    weight = radius - distance
    n = len(centroids)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
    values = np.concatenate([weight, weight, np.full(n, radius)])
    H = csr_matrix((values, (rows, cols)), shape=(n, n))
    return H, np.asarray(H.sum(axis=1)).ravel()


def oc_step(rho: np.ndarray, dc: np.ndarray, volumes: np.ndarray,
            passive: np.ndarray, target_volume: float, move: float) -> np.ndarray:
    lo, hi = 0.0, float(np.max(-dc[~passive] / volumes[~passive]))
    hi = max(hi, 1.0)
    for _ in range(75):
        lam = (lo + hi) / 2
        factor = np.sqrt(np.maximum(1e-16, -dc / np.maximum(lam * volumes, 1e-30)))
        trial = np.maximum(.001, np.maximum(rho-move,
                           np.minimum(1.0, np.minimum(rho+move, rho*factor))))
        trial[passive] = 1.0
        if np.dot(trial, volumes) > target_volume:
            lo = lam
        else:
            hi = lam
    return trial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=25)
    parser.add_argument('--filter-radius', type=float, default=.045)
    parser.add_argument('--move', type=float, default=.12)
    parser.add_argument('--resume-at', type=int, default=1,
                        help='resume from an already-saved iteration rho.npy')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    mesh = meshio.read(MESH)
    tetra = np.vstack([block.data for block in mesh.cells if block.type == 'tetra'])
    n_tets = len(tetra)
    # Bootstrap the FEniCS ordering; meshio's tetra order need not match it.
    init = np.full(n_tets, TARGET_LITERS / 1000 / .2803244613227307,
                   dtype=np.float64)
    if not (OUT / 'fenics_geometry.npz').exists():
        info = run_fea(0, init, geometry=True)
        print('bootstrap FEA', info['compliance'], flush=True)
    geometry = np.load(OUT / 'fenics_geometry.npz')
    centers = geometry['centroids']
    volumes = geometry['volumes']
    assert len(centers) == n_tets
    spec = np.load(SPEC / 'voxel.npz')
    cell = np.floor((centers-spec['origin'])/spec['pitch_xyz']).astype(int)
    cell = np.clip(cell, 0, 63)
    passive = spec['bc'][cell[:, 0], cell[:, 1], cell[:, 2]].astype(bool)
    total_volume = float(volumes.sum())
    target_volume = TARGET_LITERS / 1000
    passive_volume = float(volumes[passive].sum())
    if passive_volume >= target_volume:
        raise RuntimeError('mandatory BC volume exceeds mass target')
    rho = np.full(n_tets, (target_volume - passive_volume) /
                  (total_volume-passive_volume), dtype=np.float64)
    rho[passive] = 1.0
    H, Hs = filter_matrix(centers, args.filter_radius)
    print(f'FEM elements={n_tets}, envelope={total_volume*1000:.3f} L, '
          f'target={TARGET_LITERS:.3f} L, BC={passive_volume*1000:.3f} L, '
          f'filter radius={args.filter_radius*1000:.1f} mm, nnz={H.nnz}', flush=True)
    if args.resume_at > 1:
        rho = np.load(OUT / f'iter_{args.resume_at:02d}/rho.npy')
        history = [row for row in json.loads((OUT / 'history.json').read_text())
                   if row['iteration'] < args.resume_at]
        print(f'resuming at iteration {args.resume_at} with {len(history)} '
              'validated prior solves', flush=True)
    else:
        history = []
    for iteration in range(args.resume_at, args.iterations+1):
        info = run_fea(iteration, rho)
        dc = np.load(OUT / f'iter_{iteration:02d}/dc.npy')
        if dc.shape != rho.shape or not np.all(np.isfinite(dc)):
            raise RuntimeError(f'invalid FEA sensitivity at iteration {iteration}')
        # Classical mesh-independent sensitivity filter (distance weighting).
        dcf = np.asarray(H @ (rho * dc)).ravel() / np.maximum(rho*Hs, 1e-16)
        dcf[passive] = 0.0
        updated = oc_step(rho, dcf, volumes, passive, target_volume, args.move)
        row = {'iteration': iteration, 'compliance': info['compliance'],
               'volume_liters': float(np.dot(rho, volumes)*1000),
               'rho_change_max': float(np.max(np.abs(updated-rho))),
               'rho_gt_0p5': int(np.count_nonzero(rho>.5)),
               'density_min': float(rho.min()), 'density_max': float(rho.max())}
        history.append(row)
        (OUT / 'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print(json.dumps(row), flush=True)
        rho = updated
    final_info = run_fea(args.iterations+1, rho)
    np.save(OUT / 'rho_final_fenics_order.npy', rho)
    # Meshio-order export for visualization, matched by tetra centroids.
    q = mesh.points[tetra]
    meshio_centers = q.mean(axis=1)
    d, idx = cKDTree(centers).query(meshio_centers)
    if float(d.max()) > 1e-8 or len(np.unique(idx)) != len(idx):
        raise RuntimeError('FEniCS and meshio tetra order could not be matched')
    visual = meshio.Mesh(mesh.points, [meshio.CellBlock('tetra', tetra)],
                         cell_data={'density': [rho[idx]]})
    meshio.write(OUT / 'optimized_density.vtu', visual)
    summary = {'mesh': str(MESH), 'target_liters': TARGET_LITERS,
               'envelope_liters': total_volume*1000,
               'passive_bc_liters': passive_volume*1000,
               'iterations': args.iterations, 'filter_radius_mm': args.filter_radius*1000,
               'penal': 2, 'seat_newtons': 800, 'back_newtons': 200,
               'final_volume_liters': float(np.dot(rho,volumes)*1000),
               'initial_compliance': history[0]['compliance'],
               'final_compliance': final_info['compliance'],
               'optimized_density': str(OUT / 'optimized_density.vtu'),
               'rho_fenics': str(OUT / 'rho_final_fenics_order.npy')}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('FINAL', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
