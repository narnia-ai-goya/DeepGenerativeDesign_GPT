"""Evaluate the unmodified sparse chair OBJ by sampling its physical surface on FEM cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from pysdf import SDF
from scipy.spatial.transform import Rotation

from run_chair_existing_fea_on_015_2026_10_04 import OUT, OLD, SPEC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', choices=('off', 'old_on', 'new_sparse_on'),
                        default='new_sparse_on')
    name = parser.parse_args().name
    case = OUT / 'evaluation' / name / 'direct_full_mesh_15mm'
    case.mkdir(parents=True, exist_ok=True)
    source = (OUT / 'sparse/generation/mesh.obj' if name == 'new_sparse_on' else
              OLD / ('sparse_d0_s0' if name == 'off' else 'sparse_d1_s1') /
              'generation/mesh.obj')
    mesh = trimesh.load(source, force='mesh', process=False)
    cal = json.loads((OUT.parents[4] / 'single_view_spec_2026-10-03/specification.json').read_text())['native_frame_registration']
    rotation = Rotation.from_euler('x', cal['rotation_x_degrees'], degrees=True).as_matrix()
    mesh.vertices = ((mesh.vertices - np.asarray(cal['source_center_m'])) @ rotation.T *
                     cal['uniform_scale'] + np.asarray(cal['physical_center_m']))
    mesh.export(case / 'aligned_full.obj')
    geometry = np.load(OUT.parent / 'diagnostics/msh_simp_topopt_015_2026-10-04/fenics_geometry.npz')
    centroids = geometry['centroids']
    volumes = geometry['volumes']
    if len(centroids) != 388856 or not mesh.is_watertight:
        raise RuntimeError('unexpected FEM mesh or non-watertight source')
    sdf = SDF(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    rho = np.empty(len(centroids), dtype=np.float64)
    for lo in range(0, len(centroids), 25000):
        hi = min(len(centroids), lo+25000)
        rho[lo:hi] = np.where(sdf(centroids[lo:hi].astype(np.float32)) > 0, 1., .001)
        print(f'sampled {hi:,}/{len(centroids):,}', flush=True)
    np.save(case / 'rho_cells.npy', rho)
    summary = {'source_obj': str(source), 'n_vertices': len(mesh.vertices),
               'n_faces': len(mesh.faces), 'watertight': bool(mesh.is_watertight),
               'fea_mesh': str(SPEC / 'fea_domain/chair_015.msh'),
               'inside_cells': int(np.count_nonzero(rho > .5)),
               'solid_volume_liters': float(volumes[rho > .5].sum()*1000),
               'method': 'complete physical aligned sparse mesh.obj, including all components; signed-distance sampling at 15 mm FEM cell centroids; no protected anchor, BC union, envelope voxel repair, or smoothing'}
    (case / 'geometry_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
