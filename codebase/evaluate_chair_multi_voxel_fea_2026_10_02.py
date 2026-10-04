#!/usr/bin/env python3
"""Post-generation 64³ voxel FEM diagnostic on the six chair sparse meshes."""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import trimesh
from pysdf import SDF

from make_chair_domain import ROOT

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'form_embodies_multi_2026-10-02'
FENICS = '/home/goya/miniconda3/envs/fenics/bin/python'
SCRIPT = ROOT / 'codebase/code/fenics_fea_bracket.py'
SPEC = BASE / 'registered_spec_2026-10-02/native_frame_spec.npz'


def main():
    spec = np.load(SPEC)
    indices = np.argwhere(spec['bracket'].astype(bool))
    nodes = spec['origin'] + (indices + .5) * spec['pitch_xyz']
    rows = []
    for case in ('open_arm', 'solid_side', 'diagonal_braced'):
        modes = ['fea_off', 'fea_on']
        if case in ('open_arm', 'solid_side'):
            modes += ['dense_on_sparse_off', 'dense_off_sparse_on']
        for mode in modes:
            directory = OUT / (f'{case}_{mode}' if mode.startswith('dense_') else f'{case}_{mode}_sparse')
            source = directory / 'generation/mesh.obj'
            if not source.exists():
                continue
            mesh = trimesh.load(source, force='mesh')
            signed = SDF(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
            occupied = signed(nodes.astype(np.float32)) > 0
            check = directory / 'voxel_fea_800N'
            check.mkdir(exist_ok=True)
            np.save(check / 'nodes.npy', nodes.astype(np.float64))
            np.save(check / 'rho.npy', np.where(occupied, 1., 0.001).astype(np.float64))
            cmd = [FENICS, str(SCRIPT), '--domain-dir', str(OUT / 'native_fea_domain'),
                   '--density', str(check / 'rho.npy'), '--nodes', str(check / 'nodes.npy'),
                   '--output', str(check / 'dc.npy'), '--mesh-cache', str(OUT / 'native_fea_shared.msh'),
                   '--mesh-size', '0.035', '--penal', '2', '--E0', '1.0',
                   '--load-magnitude', '800']
            env = dict(os.environ, LOAD_MODE='-y', FEA_MAX_NODE_MAP_DISTANCE='0.03')
            with (check / 'fea.log').open('w') as log:
                proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            info_path = check / 'dc_info.json'
            item = {'case': case, 'mode': mode, 'mesh': str(source),
                    'occupied_domain_voxels': int(occupied.sum()),
                    'domain_voxels': len(nodes), 'occupancy_fraction': float(occupied.mean()),
                    'fea_exit_code': proc.returncode, 'fea_info': str(info_path),
                    'fea_log': str(check / 'fea.log')}
            if proc.returncode == 0:
                info = json.loads(info_path.read_text())
                item['compliance_proxy'] = float(info['compliance'])
            rows.append(item)
            (OUT / 'voxel_fea_800N.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(json.dumps(item), flush=True)


if __name__ == '__main__':
    main()
