#!/usr/bin/env python3
"""Register angular bracket images to the physical BCs and rasterize 64² targets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from PIL import Image

from diagnose_angular_stage_fidelity import (CASES, EXP, OUT, bc_centers,
                                            input_markers, phys_to_pixel, warp_image)
from run_semantic_qd_sparse_round import transform_matrix


TARGET_OUT = EXP / 'image_projection_guidance_2026-09-24'


def main() -> None:
    TARGET_OUT.mkdir(parents=True, exist_ok=True)
    transform = transform_matrix()
    manifest = {}
    for name in ('triangular_truss', 'staggered_chevron'):
        case = CASES[name]
        config = json.loads((case / 'config_dense_top.json').read_text())
        mesh_config = config['stages']['mesh']
        npz = np.load(mesh_config['bc_proper'])
        origin = np.asarray(npz['origin'], dtype=np.float64)
        pitch = np.asarray(npz['pitch_xyz'], dtype=np.float64)
        image = np.asarray(Image.open(case / 'input/그림1.png').convert('RGB'))
        source_markers = input_markers(image)
        physical_markers = bc_centers(config)
        registered, residual = warp_image(image, source_markers, physical_markers, 'affine')
        mask_800 = registered.min(axis=2) < 230
        soft_800 = ndi.gaussian_filter(mask_800.astype(np.float32), sigma=7.0)

        ii, jj = np.indices((64, 64))
        model_xyz = np.column_stack((origin[0] + (ii.ravel() + 0.5) * pitch[0],
                                     origin[1] + (jj.ravel() + 0.5) * pitch[1],
                                     np.full(ii.size, origin[2] + 32 * pitch[2]),
                                     np.ones(ii.size)))
        physical_xy = (model_xyz @ transform)[:, :2]
        pixel_xy = phys_to_pixel(physical_xy)
        target = ndi.map_coordinates(soft_800, [pixel_xy[:, 1], pixel_xy[:, 0]],
                                     order=1, mode='constant', cval=0).reshape(64, 64)
        envelope_2d = np.asarray(npz['bracket'], bool).any(axis=2)
        bc_2d = np.asarray(npz['bc'], bool).any(axis=2)
        bc_guard = ndi.binary_dilation(bc_2d, iterations=2)
        valid = envelope_2d & ~bc_guard
        confidence = np.clip(np.abs(target - 0.5) * 2, 0.2, 1.0)
        weight = valid.astype(np.float32) * confidence.astype(np.float32)
        path = TARGET_OUT / f'{name}_target.npz'
        np.savez(path, target=target.astype(np.float32), weight=weight,
                 source_image=str(case / 'input/그림1.png'),
                 marker_rmse_px=float(np.sqrt(np.mean(residual ** 2))))
        Image.fromarray(registered).save(TARGET_OUT / f'{name}_registered.png')
        # Voxel array axes are (physical X, physical Y); display top with +Y upward.
        display_target = np.flipud(target.T)
        display_weight = np.flipud(weight.T)
        for label, array in (('target', display_target), ('weight', display_weight)):
            Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255)).resize((512, 512),
                Image.Resampling.NEAREST).save(TARGET_OUT / f'{name}_{label}.png')
        manifest[name] = {'target': str(path), 'registered_image': str(TARGET_OUT / f'{name}_registered.png'),
                          'marker_rmse_px': float(np.sqrt(np.mean(residual ** 2))),
                          'supervised_voxels_2d': int(valid.sum()),
                          'target_mean_on_supervised': float(target[valid].mean())}
    (TARGET_OUT / 'targets.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(TARGET_OUT / 'targets.json')


if __name__ == '__main__':
    main()
