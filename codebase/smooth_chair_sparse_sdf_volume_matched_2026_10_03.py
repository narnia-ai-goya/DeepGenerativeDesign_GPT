#!/usr/bin/env python3
"""Volume-matched SDF smoothing, avoiding the mass gain from fixed-iso blurring."""
from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
import trimesh

from make_chair_domain import ROOT

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'sparse_wrinkle_pilot_2026-10-03'
CAPTURE = OUT / 'sdf_capture'


def main() -> None:
    data = np.load(CAPTURE / 'raw_sdf.npz')
    sdf = data['sdf']
    level = float(data['mc_threshold'])
    origin = data['env_origin']
    pitch = data['env_pitch']
    baseline = trimesh.load(CAPTURE / 'generation/mesh.obj', force='mesh', process=False)
    lo = np.maximum(0, np.floor((baseline.bounds[0] - origin) / pitch - 12).astype(int))
    hi = np.minimum(sdf.shape, np.ceil((baseline.bounds[1] - origin) / pitch + 13).astype(int))
    crop = sdf[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    fraction = float((crop < level).mean())
    rows = []
    for sigma in (2.0, 3.0, 4.0):
        case = OUT / f'sdf_sigma{int(sigma)}_volmatch'
        generation = case / 'generation'
        generation.mkdir(parents=True, exist_ok=True)
        smooth = gaussian_filter(crop, sigma=sigma, mode='nearest')
        matched_level = float(np.quantile(smooth, fraction))
        verts, faces, _, _ = marching_cubes(smooth, level=matched_level)
        verts = origin + (verts + lo + .5) * pitch
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        target = generation / 'mesh.obj'
        mesh.export(target)
        row = {'name': case.name, 'sigma_vox512': sigma,
               'sigma_mm': float(sigma * pitch.mean() * 1000),
               'original_iso': level, 'volume_matched_iso': matched_level,
               'target_occupied_fraction_in_crop': fraction,
               'mesh': str(target), 'faces': int(len(mesh.faces))}
        (case / 'run.json').write_text(json.dumps(row, indent=2) + '\n')
        rows.append(row)
        print(row, flush=True)
    (OUT / 'sdf_volume_match_runs.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
