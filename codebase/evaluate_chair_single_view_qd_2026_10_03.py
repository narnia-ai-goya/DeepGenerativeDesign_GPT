#!/usr/bin/env python3
"""Evaluate fixed-image chair candidates against the registered physical spec."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from make_chair_domain import ROOT

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'single_view_qd_2026-10-03'
SPEC = BASE / 'single_view_spec_2026-10-03'
SOURCE = BASE / 'minimal_direct3ds2_2026-10-02'
sys.path.insert(0, str(ROOT / 'codebase/code'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402


def evaluate(case_name: str) -> dict:
    case = OUT / case_name
    case.mkdir(parents=True, exist_ok=True)
    seed = (42 if case_name == 'seed_42' else
            int(json.loads((case / 'config.json').read_text())['seed']))
    physical_mesh = case / 'aligned_main.obj'
    if case_name == 'seed_42':
        physical_mesh = SOURCE / 'aligned_sparse_main.obj'
    elif not physical_mesh.exists():
        raw = trimesh.load(case / 'generation/mesh.obj', force='mesh', process=False)
        cal = json.loads((SPEC / 'specification.json').read_text())['native_frame_registration']
        rot = Rotation.from_euler('x', cal['rotation_x_degrees'], degrees=True).as_matrix()
        raw.vertices = ((raw.vertices-np.asarray(cal['source_center_m'])) @ rot.T *
                        cal['uniform_scale'] + np.asarray(cal['physical_center_m']))
        main = max(raw.split(only_watertight=False), key=lambda part: abs(part.volume))
        main.export(physical_mesh)
    mesh = trimesh.load(physical_mesh, force='mesh', process=False)
    d = np.load(SPEC / 'voxel.npz')
    cached_occupancy = case / 'occupancy.npz'
    occupied = (np.load(cached_occupancy)['occupied'] if cached_occupancy.exists() else
                voxel_centers_inside(mesh, 64, d['origin'], d['pitch_xyz']))
    x,y,z = [d['origin'][i] + (np.arange(64)+.5)*d['pitch_xyz'][i] for i in range(3)]
    # Designer-readable shape phenotype, measured only from the final 3D mesh.
    # Y-Z opening between seat and arm, projected along chair width (X).
    side = occupied.any(axis=0)
    region = side[np.ix_((y >= -.16)&(y <= .12), (z >= .61)&(z <= .76))]
    side_open_fraction = float(1.0 - region.mean())
    # Top-to-lower width ratio of the backrest: tapered versus broad upper rim.
    X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
    widths = []
    for lo, hi in ((.70, .76), (.84, .90)):
        patch = occupied & (Z >= lo) & (Z < hi) & (Y > .12)
        coordinates = X[patch]
        widths.append(float(np.percentile(coordinates,95)-np.percentile(coordinates,5))
                      if len(coordinates) else 0.)
    backrest_taper_ratio = widths[1]/widths[0] if widths[0] else 0.
    cover = {}
    for key in ('fix','load','back_load'):
        target = d[key].astype(bool)
        cover[key] = float((occupied & target).sum()/max(1,target.sum()))
    spec = json.loads((SPEC / 'specification.json').read_text())
    foot_cover = []
    for xc,yc in spec['foot_centers_m']:
        mask=((X-xc)**2+(Y-yc)**2<=.038**2)&(Z>=.005)&(Z<=.035)
        foot_cover.append(float((occupied&mask).sum()/max(1,mask.sum())))
    item = {
        'id': case_name, 'seed': seed,
        'source_image': str(case / 'input_image.png' if (case / 'input_image.png').exists()
                            else BASE / 'open_arm/input/v00_front_lo.png'),
        'mesh': str(physical_mesh), 'watertight': bool(mesh.is_watertight),
        'mesh_components': len(mesh.split(only_watertight=False)),
        'volume_liters': float(abs(mesh.volume)*1000),
        'occupied_voxels': int(occupied.sum()),
        'outside_envelope_voxels': int((occupied & ~d['bracket']).sum()),
        'outside_envelope_fraction': float((occupied & ~d['bracket']).sum()/max(1,occupied.sum())),
        'seat_coverage': cover['load'], 'back_coverage': cover['back_load'],
        'foot_coverage': foot_cover,
        'side_open_fraction': side_open_fraction,
        'backrest_taper_ratio': backrest_taper_ratio,
    }
    np.savez_compressed(case / 'occupancy.npz', occupied=occupied)
    (case / 'metrics.json').write_text(json.dumps(item, indent=2)+'\n')
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('cases', nargs='+')
    args = parser.parse_args()
    for case_name in args.cases:
        if case_name.isdigit():
            case_name = f'seed_{case_name}'
        print(json.dumps(evaluate(case_name), indent=2), flush=True)


if __name__ == '__main__':
    main()
