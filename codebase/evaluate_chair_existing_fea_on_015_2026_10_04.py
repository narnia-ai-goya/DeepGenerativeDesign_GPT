"""Evaluate the existing generator FEA-on repeat on one 15 mm chair FEM mesh."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.ndimage import generate_binary_structure, label
from scipy.spatial.transform import Rotation
from skimage.measure import marching_cubes
import trimesh

from evaluate_chair_protected_parts_qd_2026_10_03 import ANCHOR, preview
from make_chair_domain import ROOT
from run_chair_existing_fea_on_015_2026_10_04 import OLD, OUT, SPEC

sys.path.insert(0, str(ROOT / 'codebase/code'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402


SOLVER = ROOT / 'codebase/code/fenics_fea_bracket.py'
FENICS = '/home/goya/miniconda3/envs/fenics/bin/python'


def prepare(name: str, source: Path, spec, origin, pitch, env, bc, anchor_occ, protected):
    case = OUT / 'evaluation' / name
    case.mkdir(parents=True, exist_ok=True)
    cal = json.loads((ROOT / 'experiments/chair/sofa_style_2026-09-28/single_view_spec_2026-10-03/specification.json').read_text())['native_frame_registration']
    rotation = Rotation.from_euler('x', cal['rotation_x_degrees'], degrees=True).as_matrix()
    raw = trimesh.load(source, force='mesh', process=False)
    mesh = max(raw.split(only_watertight=False), key=lambda part: abs(part.volume))
    mesh.vertices = ((mesh.vertices - np.asarray(cal['source_center_m'])) @ rotation.T *
                     cal['uniform_scale'] + np.asarray(cal['physical_center_m']))
    mesh.export(case / 'aligned_main.obj')
    preview(mesh, case / 'raw_preview.png')
    generated = voxel_centers_inside(mesh, 64, origin, pitch).astype(bool)
    candidate = (anchor_occ & protected) | (generated & ~protected)
    realized = (candidate & env) | bc
    np.savez_compressed(case / 'realized_occupancy.npz', occupied=realized)
    vv, ff, _, _ = marching_cubes(np.pad(realized, 1), .5)
    combined = trimesh.Trimesh(vertices=origin + (vv - .5)*pitch, faces=ff, process=False)
    combined.export(case / 'combined_voxel.obj')
    preview(combined, case / 'evaluated_preview.png')
    return {'name': name, 'source_mesh': str(source), 'aligned_mesh': str(case / 'aligned_main.obj'),
            'raw_preview': str(case / 'raw_preview.png'),
            'evaluated_preview': str(case / 'evaluated_preview.png'),
            'mass_liters': float(realized.sum() * np.prod(pitch) * 1000),
            'components_6conn': int(label(realized, generate_binary_structure(3, 1))[1]),
            'repair_fraction': float(((realized & ~candidate).sum() +
                                      (candidate & ~realized).sum()) / max(1, realized.sum())),
            'seat_bc_fraction_raw': float((candidate & spec['load']).sum() / spec['load'].sum()),
            'back_bc_fraction_raw': float((candidate & spec['back_load']).sum() /
                                          spec['back_load'].sum())}


def solve(row, indices, nodes):
    case = OUT / 'evaluation' / row['name']
    occ = np.load(case / 'realized_occupancy.npz')['occupied']
    rho = np.where(occ[indices[:, 0], indices[:, 1], indices[:, 2]], 1., .001)
    np.save(case / 'rho.npy', rho)
    np.save(case / 'nodes.npy', nodes)
    cmd = [FENICS, str(SOLVER), '--domain-dir', str(SPEC / 'fea_domain'),
           '--density', str(case / 'rho.npy'), '--nodes', str(case / 'nodes.npy'),
           '--output', str(case / 'dc.npy'), '--mesh-cache',
           str(SPEC / 'fea_domain/chair_015.msh'), '--mesh-size', '.015',
           '--penal', '2', '--E0', '1', '--load-magnitude', '800',
           '--second-load-stl', str(SPEC / 'fea_domain_back/load.stl'),
           '--second-load-magnitude', '200', '--second-load-mode', 'y']
    env = {**os.environ, 'LOAD_MODE': '-z', 'BC_SURFACE_DIST': '1', 'BC_DIST': '.025',
           'FEA_MAX_NODE_MAP_DISTANCE': '.04', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'FEA_KSP_TYPE': 'gmres', 'FEA_PC_TYPE': 'gamg',
           'FEA_KSP_RTOL': '1e-7', 'FEA_KSP_MAX_IT': '1500'}
    with (case / 'fea.log').open('w') as stream:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"FEA {row['name']} failed: {(case / 'fea.log').read_text()[-2000:]}")
    row['compliance_15mm'] = json.loads((case / 'dc_info.json').read_text())['compliance']
    row['fea_log'] = str(case / 'fea.log')
    print(row['name'], row['mass_liters'], row['compliance_15mm'], flush=True)


def main():
    spec = np.load(SPEC / 'voxel.npz')
    env, bc = spec['bracket'].astype(bool), spec['bc'].astype(bool)
    origin, pitch = spec['origin'], spec['pitch_xyz']
    anchor = trimesh.load(ANCHOR, force='mesh', process=False)
    anchor_occ = voxel_centers_inside(anchor, 64, origin, pitch).astype(bool)
    z = origin[2] + (np.arange(64) + .5)*pitch[2]
    protected = (z <= .61)[None, None, :] | spec['back_load'].astype(bool)
    rows = []
    for name, source in [('off', OLD / 'sparse_d0_s0/generation/mesh.obj'),
                         ('old_on', OLD / 'sparse_d1_s1/generation/mesh.obj'),
                         ('new_dense_on', OUT / 'dense/generation/mesh.obj'),
                         ('new_sparse_on', OUT / 'sparse/generation/mesh.obj')]:
        rows.append(prepare(name, source, spec, origin, pitch, env, bc,
                            anchor_occ, protected))
    indices = np.argwhere(env)
    nodes = origin + (indices + .5)*pitch
    for row in rows:
        solve(row, indices, nodes)
    (OUT / 'evaluation/summary.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
