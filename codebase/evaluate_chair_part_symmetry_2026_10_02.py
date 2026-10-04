#!/usr/bin/env python3
"""Matched post-generation voxel FEM proxy for the chair symmetry pilot."""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import trimesh
from pysdf import SDF
from scipy.spatial.transform import Rotation

from make_chair_domain import ROOT

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
PILOT = BASE / 'part_symmetry_pilot_2026-10-02'
FEM = BASE / 'form_embodies_multi_2026-10-02'
SPEC = BASE / 'registered_spec_2026-10-02'


def main() -> None:
    spec = np.load(SPEC / 'native_frame_spec.npz')
    indices = np.argwhere(spec['bracket'].astype(bool))
    native_nodes = spec['origin'] + (indices + .5) * spec['pitch_xyz']
    cal = json.loads((SPEC / 'calibration.json').read_text())['native_to_physical']
    source = np.asarray(cal['source_center_m'])
    center = np.asarray(cal['physical_center_m'])
    rotation = Rotation.from_euler('x', cal['rotation_x_degrees'], degrees=True).as_matrix()
    physical_nodes = (native_nodes - source) @ rotation.T * float(cal['uniform_scale']) + center
    rows = []
    for name in ('original_voxelized', 'right_to_left', 'left_to_right'):
        mesh_path = PILOT / f'{name}.obj'
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        sdf = SDF(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
        inside = sdf(physical_nodes.astype(np.float32)) > 0
        check = PILOT / f'fea_{name}'
        check.mkdir(exist_ok=True)
        np.save(check / 'nodes.npy', native_nodes.astype(np.float64))
        np.save(check / 'rho.npy', np.where(inside, 1., .001).astype(np.float64))
        cmd = ['/home/goya/miniconda3/envs/fenics/bin/python',
               str(ROOT / 'codebase/code/fenics_fea_bracket.py'),
               '--domain-dir', str(FEM / 'native_fea_domain'),
               '--density', str(check / 'rho.npy'), '--nodes', str(check / 'nodes.npy'),
               '--output', str(check / 'dc.npy'),
               '--mesh-cache', str(FEM / 'native_fea_shared.msh'),
               '--mesh-size', '0.035', '--penal', '2', '--E0', '1.0',
               '--load-magnitude', '800']
        env = dict(os.environ, LOAD_MODE='-y', FEA_MAX_NODE_MAP_DISTANCE='0.03')
        with (check / 'fea.log').open('w') as log:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        item = {'name': name, 'mesh': str(mesh_path), 'occupied_domain_voxels': int(inside.sum()),
                'domain_voxels': len(inside), 'fea_exit_code': proc.returncode,
                'fea_log': str(check / 'fea.log')}
        if proc.returncode == 0:
            info = json.loads((check / 'dc_info.json').read_text())
            item['compliance_proxy'] = float(info['compliance'])
        rows.append(item)
        (PILOT / 'fea_metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(item), flush=True)


if __name__ == '__main__':
    main()
