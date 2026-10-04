"""Replay the sparse chair stage once to capture its pre-refiner mesh and volume."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from make_chair_domain import ROOT
from run_chair_dual_load_fea_2026_10_04 import INPUT
from run_chair_existing_fea_on_015_2026_10_04 import OUT, SPEC
from run_chair_sparse_fea_loop_2026_10_03 import command
from run_connectivity_qd_sampling import generation_env


def main():
    folder = OUT / 'diagnostics/sparse_volume_transition'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'generation'
    if not (output / 'mesh_pre_refiner.obj').exists():
        env = generation_env(6)
        env.update({'VANILLA': '0', 'FEA_LOAD_MAGNITUDE': '800',
                    'FEA_MAX_NODE_MAP_DISTANCE': '.04', 'BC_SURFACE_DIST': '1',
                    'BC_DIST': '.025', 'FEA_WORK_DIR': str(folder / 'fea_work'),
                    'FEA_KSP_TYPE': 'gmres', 'FEA_PC_TYPE': 'gamg',
                    'FEA_KSP_RTOL': '1e-7', 'FEA_KSP_MAX_IT': '1500',
                    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                    'MKL_NUM_THREADS': '1', 'D3DS2_SAVE_PRE_REFINER': '1'})
        for key in ('FEA_SECOND_LOAD_STL', 'FEA_SECOND_LOAD_MAGNITUDE',
                    'FEA_SECOND_LOAD_MODE'):
            env.pop(key, None)
        config = OUT / 'sparse/config.json'
        with (folder / 'generation.log').open('w') as stream:
            proc = subprocess.run(command(config, INPUT, output), cwd=ROOT, env=env,
                                  stdout=stream, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f'Generation failed: {folder / "generation.log"}')
    pre = trimesh.load(output / 'mesh_pre_refiner.obj', force='mesh', process=False)
    post = trimesh.load(output / 'mesh.obj', force='mesh', process=False)
    reference = trimesh.load(OUT / 'sparse/generation/mesh.obj', force='mesh', process=False)
    dense = trimesh.load(OUT / 'dense/generation/mesh.obj', force='mesh', process=False)
    grid = np.load(SPEC / 'native_frame_spec.npz')
    pitch64 = np.asarray(grid['pitch_xyz'])
    # sparse2mesh diagnostic is in normalized [-1,1] coordinates; saved final
    # mesh was transformed to native world coordinates after the refiner.
    native_jacobian = float(np.prod(32*pitch64))
    cal = json.loads((OUT.parents[4] / 'single_view_spec_2026-10-03/specification.json').read_text())['native_frame_registration']
    physical_jacobian = float(cal['uniform_scale'])**3
    result = {'dense_native_L': float(abs(dense.volume)*1000),
              'pre_refiner_native_L': float(abs(pre.volume)*native_jacobian*1000),
              'post_refiner_native_L': float(abs(post.volume)*1000),
              'original_sparse_native_L': float(abs(reference.volume)*1000),
              'dense_physical_L': float(abs(dense.volume)*physical_jacobian*1000),
              'pre_refiner_physical_L': float(abs(pre.volume)*native_jacobian*physical_jacobian*1000),
              'post_refiner_physical_L': float(abs(post.volume)*physical_jacobian*1000),
              'original_sparse_physical_L': float(abs(reference.volume)*physical_jacobian*1000),
              'pre_refiner_mesh': str(output / 'mesh_pre_refiner.obj'),
              'post_refiner_mesh': str(output / 'mesh.obj'),
              'generation_log': str(folder / 'generation.log')}
    (folder / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
