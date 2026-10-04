#!/usr/bin/env python3
"""Check chair BC/envelope geometry and their 64³ rasterization."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))
from make_bc_proper_pysdf import voxel_centers_inside


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data_real/chair'


def main() -> None:
    data = np.load(OUT / 'voxel.npz')
    meshes = {
        key: trimesh.load(OUT / filename, force='mesh', process=True)
        for key, filename in {
            'env': 'original_DesignSpace.stl', 'fix': 'fixed.stl',
            'load': 'load.stl', 'keepout': 'occupant_keepout.stl'
        }.items()
    }
    raw = {name: voxel_centers_inside(mesh, 64, data['origin'], data['pitch_xyz'])
           for name, mesh in meshes.items()}
    checks = {
        'envelope_watertight': bool(meshes['env'].is_watertight),
        'fixed_watertight': bool(meshes['fix'].is_watertight),
        'load_watertight': bool(meshes['load'].is_watertight),
        'envelope_components': len(meshes['env'].split()),
        'fixed_components': len(meshes['fix'].split()),
        'load_components': len(meshes['load'].split()),
        'pitch_mm': float(data['pitch_xyz'][0] * 1000),
        'envelope_voxels': int(data['bracket'].sum()),
        'fixed_voxels': int(data['fix'].sum()),
        'load_voxels': int(data['load'].sum()),
        'bc_voxels': int(data['bc'].sum()),
        'raw_fixed_outside_envelope': int((raw['fix'] & ~raw['env']).sum()),
        'raw_load_outside_envelope': int((raw['load'] & ~raw['env']).sum()),
        'keepout_intersection_voxels': int((raw['keepout'] & raw['env']).sum()),
        'fixed_load_overlap_voxels': int((data['fix'] & data['load']).sum()),
        'envelope_voxel_components': int(label(data['bracket'])[1]),
        'keepout_voxels': int(raw['keepout'].sum()),
    }
    passed = (all(checks[key] for key in ('envelope_watertight', 'fixed_watertight',
                                          'load_watertight'))
              and checks['envelope_components'] == 1
              and checks['fixed_components'] == 4
              and checks['load_components'] == 1
              and checks['envelope_voxel_components'] == 1
              and all(checks[key] == 0 for key in (
                  'raw_fixed_outside_envelope', 'raw_load_outside_envelope',
                  'keepout_intersection_voxels', 'fixed_load_overlap_voxels')))
    checks['passed'] = passed
    (OUT / 'validation.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
