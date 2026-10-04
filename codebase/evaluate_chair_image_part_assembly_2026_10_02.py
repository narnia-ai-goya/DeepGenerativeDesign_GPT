#!/usr/bin/env python3
"""Matched 800 N chair FEM proxy for image-first part assembly candidates."""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from make_chair_domain import ROOT

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'semantic_parts_from_image_2026-10-02/visual_hull_assembly'
SPEC = BASE / 'registered_spec_2026-10-02'
FEM = BASE / 'form_embodies_multi_2026-10-02'


def main() -> None:
    spec = np.load(SPEC / 'native_frame_spec.npz')
    indices = np.argwhere(spec['bracket'].astype(bool))
    native_nodes = spec['origin'] + (indices + .5) * spec['pitch_xyz']
    cal = json.loads((SPEC / 'calibration.json').read_text())['native_to_physical']
    source = np.asarray(cal['source_center_m'])
    center = np.asarray(cal['physical_center_m'])
    rotation = Rotation.from_euler('x', cal['rotation_x_degrees'], degrees=True).as_matrix()
    physical_nodes = (native_nodes - source) @ rotation.T * float(cal['uniform_scale']) + center
    occupancy = np.load(OUT / 'occupancy.npz')
    origin = occupancy['origin']
    pitch = float(occupancy['pitch'])
    rows = []
    for name in ('assembled_contact_repaired', 'assembled_domain_clipped'):
        occ = occupancy[name]
        ijk = np.floor((physical_nodes - (origin - pitch / 2)) / pitch).astype(int)
        valid = ((ijk >= 0) & (ijk < np.asarray(occ.shape))).all(axis=1)
        chosen = np.zeros(len(ijk), dtype=bool)
        chosen[valid] = occ[ijk[valid, 0], ijk[valid, 1], ijk[valid, 2]]
        folder = OUT / ('fea_' + name)
        folder.mkdir(exist_ok=True)
        np.save(folder / 'nodes.npy', native_nodes.astype(np.float64))
        np.save(folder / 'rho.npy', np.where(chosen, 1., .001).astype(np.float64))
        cmd = ['/home/goya/miniconda3/envs/fenics/bin/python',
               str(ROOT / 'codebase/code/fenics_fea_bracket.py'),
               '--domain-dir', str(FEM / 'native_fea_domain'),
               '--density', str(folder / 'rho.npy'), '--nodes', str(folder / 'nodes.npy'),
               '--output', str(folder / 'dc.npy'),
               '--mesh-cache', str(FEM / 'native_fea_shared.msh'),
               '--mesh-size', '0.035', '--penal', '2', '--E0', '1.0',
               '--load-magnitude', '800']
        env = dict(os.environ, LOAD_MODE='-y', FEA_MAX_NODE_MAP_DISTANCE='0.03')
        with (folder / 'fea.log').open('w') as log:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        item = {'name': name, 'mesh': str(OUT / f'{name}.obj'),
                'occupied_domain_voxels': int(chosen.sum()),
                'domain_voxels': len(chosen), 'fea_exit_code': proc.returncode,
                'fea_log': str(folder / 'fea.log')}
        if proc.returncode == 0:
            item['compliance_proxy'] = float(json.loads((folder / 'dc_info.json').read_text())['compliance'])
        rows.append(item)
        (OUT / 'fea_metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(item), flush=True)


if __name__ == '__main__':
    main()
