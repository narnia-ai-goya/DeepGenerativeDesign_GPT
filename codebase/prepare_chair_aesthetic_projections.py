#!/usr/bin/env python3
"""Register axis-aligned front/right chair images to the 64³ physical frame."""
from __future__ import annotations

import json
import sys

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter, map_coordinates

from make_chair_aesthetic_reference import OUT, DOMAIN, ROOT, render_metal
sys.path.insert(0, str(ROOT / 'codebase/code'))
from make_bc_proper_pysdf import voxel_centers_inside
from cond_render_pv import camera_from_elev_azim


def sample_target(image: np.ndarray, physical_points: np.ndarray,
                  eye: np.ndarray, center: np.ndarray, up: np.ndarray,
                  half_extent: float) -> np.ndarray:
    forward = (center-eye) / np.linalg.norm(center-eye)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)
    rel = physical_points-center
    size = image.shape[0]
    col = (.5 + (rel @ right)/(2*half_extent))*(size-1)
    row = (.5 - (rel @ true_up)/(2*half_extent))*(size-1)
    soft = gaussian_filter((image.min(axis=2)<210).astype(np.float32), sigma=.75)
    return map_coordinates(soft, [row, col], order=1, mode='constant', cval=0)


def main() -> None:
    mesh = trimesh.load(OUT / 'reference.obj', force='mesh')
    env = trimesh.load(DOMAIN / 'original_DesignSpace.stl', force='mesh')
    grid = np.load(DOMAIN / 'voxel.npz')
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents))*1.5
    scale = float(env.extents.max())/2*1.15
    origin, pitch = grid['origin'], grid['pitch_xyz']
    axes = [origin[i]+(np.arange(64)+.5)*pitch[i] for i in range(3)]
    reference_vox = voxel_centers_inside(mesh, 64, origin, pitch)
    result = {}
    manifest = {}
    for name, elev, azim, first_axis, projection_axis in (
            ('front', 0, 0, 0, 1), ('right', 0, 90, 1, 0)):
        eye, up = camera_from_elev_azim(center, radius, elev, azim)
        image = render_metal(mesh, eye, center, up, scale/1.15)
        Image.fromarray(image).save(OUT / f'v_{name}_axis.png')
        a, z = np.meshgrid(axes[first_axis], axes[2], indexing='ij')
        points = np.empty((64, 64, 3), np.float64)
        points[..., first_axis] = a
        points[..., 2] = z
        points[..., projection_axis] = center[projection_axis]
        target = sample_target(image, points.reshape(-1, 3), eye, center, up,
                               scale).reshape(64, 64)
        target = np.clip(target, 0, 1)
        reference_projection = reference_vox.any(axis=projection_axis)
        envelope_projection = grid['bracket'].any(axis=projection_axis)
        weight = binary_dilation(envelope_projection, iterations=2).astype(np.float32)
        predicted = target>.5
        alignment_iou = float((predicted&reference_projection).sum()
                              / max((predicted|reference_projection).sum(), 1))
        result[f'target_{name}'] = target.astype(np.float32)
        result[f'weight_{name}'] = weight
        Image.fromarray(np.uint8(np.flipud(target.T)*255)).resize(
            (512, 512), Image.Resampling.NEAREST).save(OUT / f'target_{name}.png')
        manifest[name] = {'axis': projection_axis,
                          'pixel_to_voxel_alignment_iou': alignment_iou,
                          'supervised_pixels': int(weight.sum()),
                          'target_positive_pixels': int(predicted.sum())}
    result['active_threshold'] = np.float32(.1)
    target_path = OUT / 'front_right_projection_targets.npz'
    np.savez(target_path, **result)
    (OUT / 'projection_registration.json').write_text(
        json.dumps({'target': str(target_path), 'views': manifest}, indent=2)+'\n')
    print(OUT / 'projection_registration.json')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
