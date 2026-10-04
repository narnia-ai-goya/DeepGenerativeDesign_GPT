#!/usr/bin/env python3
"""Build camera-aligned orthographic silhouette targets for chair dense guidance."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter

from make_chair_domain import ROOT

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim  # noqa: E402
sys.path.insert(0, str(ROOT / 'codebase/code'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402


BASE = ROOT / 'experiments/chair/image_concepts_2026-09-27/diagonal_braced'
INPUT = BASE / 'multiview_registered/input'
DOMAIN = ROOT / 'data_real/chair'
SPECS = (('front', 'v00_front_lo', 15, 0),
         ('right', 'v02_right_lo', 15, 90),
         ('top', 'v_top', 85, 0))


def camera_grid(center: np.ndarray, eye: np.ndarray, up: np.ndarray,
                half_extent: float, origin: np.ndarray, pitch: np.ndarray,
                depth_half: float, n_rays: int = 160) -> np.ndarray:
    forward = (center-eye) / np.linalg.norm(center-eye)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)
    pixel = np.arange(64, dtype=np.float64) + .5
    horizontal = (pixel/64-.5)*2*half_extent
    vertical = (.5-pixel/64)*2*half_extent
    hh, vv = np.meshgrid(horizontal, vertical, indexing='xy')
    plane = center + hh[..., None]*right + vv[..., None]*true_up
    depth = np.linspace(-depth_half, depth_half, n_rays)
    world = plane[None] + depth[:, None, None, None]*forward
    voxel = (world-origin)/pitch-.5
    normalized = 2*voxel/63-1
    # torch grid_sample's (W,H,D) is (Z,Y,X) for our array (X,Y,Z).
    return normalized[..., [2, 1, 0]].astype(np.float16)


def downsample_mask(image: Image.Image) -> np.ndarray:
    dark = (np.asarray(image.convert('RGB')).min(axis=2) < 210).astype(np.uint8)*255
    return np.asarray(Image.fromarray(dark).resize((64, 64), Image.Resampling.BOX),
                      dtype=np.float32)/255


def main(base: str | None = None) -> None:
    output_base = BASE if base is None else ROOT / base
    input_dir = INPUT if base is None else output_base / 'input'
    env = trimesh.load(DOMAIN / 'original_DesignSpace.stl', force='mesh')
    reference = trimesh.load(
        ROOT / 'experiments/chair/aesthetic_reference_2026-09-27/reference.obj', force='mesh')
    voxel_data = np.load(DOMAIN / 'voxel.npz')
    origin = voxel_data['origin'].astype(np.float64)
    pitch = voxel_data['pitch_xyz'].astype(np.float64)
    center = env.bounds.mean(axis=0)
    radius = float(np.linalg.norm(env.extents))*1.5
    half_extent = float(env.extents.max())/2*1.15
    depth_half = float(np.linalg.norm(env.extents))
    source_occ = voxel_centers_inside(reference, 64, origin, pitch)
    source_field = torch.from_numpy(source_occ.astype(np.float32))[None, None]
    arrays: dict[str, np.ndarray] = {'active_threshold': np.float32(.1)}
    checks = {}
    for name, view, elev, azim in SPECS:
        low = downsample_mask(Image.open(input_dir / f'{view}.png'))
        target = np.clip(gaussian_filter(low, sigma=.35), 0, 1)
        near = binary_dilation(target > .05, iterations=8)
        weight = np.where(near, 1.0, .25).astype(np.float32)
        eye, up = camera_from_elev_azim(center, radius, elev, azim)
        grid = camera_grid(center, eye, up, half_extent, origin, pitch, depth_half)
        with torch.no_grad():
            grid_t = torch.from_numpy(grid.astype(np.float32))[None]
            sampled = F.grid_sample(source_field, grid_t, mode='bilinear',
                                    padding_mode='zeros', align_corners=True)[0, 0]
            source_projection = sampled.amax(dim=0).numpy() > .5
        source_image = Image.open(
            ROOT / 'experiments/chair/aesthetic_reference_2026-09-27/input' / f'{view}.png')
        source_target = downsample_mask(source_image) > .5
        alignment = float((source_projection & source_target).sum()
                          / max((source_projection | source_target).sum(), 1))
        arrays[f'target_{name}'] = target.astype(np.float32)
        arrays[f'weight_{name}'] = weight
        arrays[f'camera_grid_{name}'] = grid
        Image.fromarray((target*255).astype(np.uint8)).resize(
            (512, 512), Image.Resampling.NEAREST).save(output_base / f'camera_target_{name}.png')
        checks[name] = {'reference_camera_registration_iou': round(alignment, 4),
                        'target_positive_pixels': int((target > .5).sum()),
                        'supervised_weight_sum': round(float(weight.sum()), 1),
                        'camera_elev_deg': elev, 'camera_azim_deg': azim}
    output = output_base / 'camera_projection_targets.npz'
    np.savez_compressed(output, **arrays)
    (output_base / 'camera_projection_registration.json').write_text(
        json.dumps({'target': str(output), 'views': checks}, indent=2) + '\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', help='project-relative case folder containing input/')
    main(parser.parse_args().base)
