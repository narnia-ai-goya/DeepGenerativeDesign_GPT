#!/usr/bin/env python3
"""Trace angular image features through top-dense, multiview-dense, sparse and final."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyvista as pv
import scipy.ndimage as ndi
import trimesh
from PIL import Image, ImageDraw
from scipy.interpolate import RBFInterpolator
from skimage.morphology import skeletonize

from run_connectivity_qd_sampling import ROOT
from run_semantic_qd_sparse_round import transform_matrix


EXP = ROOT / 'experiments/bracket/angular_image_qd_loop_2026-09-24'
OUT = EXP / 'stage_fidelity_2026-09-24'
CASES = {
    'triangular_truss': EXP / 'round_00/triangular_truss',
    'x_brace': EXP / 'round_00/x_brace',
    'off_axis_spine': EXP / 'round_01/off_axis_spine',
    'staggered_chevron': EXP / 'round_01/staggered_chevron',
}
STAGES = ('top_dense', 'multiview_dense', 'sparse_aligned', 'final')
N = 800
CENTER = np.array([0.014, -0.073])
SCALE = 0.112
PIXELS_PER_M = N / (2 * SCALE)


def phys_to_pixel(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points)
    return np.column_stack((N / 2 + (points[:, 0] - CENTER[0]) * PIXELS_PER_M,
                            N / 2 - (points[:, 1] - CENTER[1]) * PIXELS_PER_M))


def component_centers(mask: np.ndarray, min_area: int) -> list[tuple[float, float, int]]:
    labels, count = ndi.label(mask)
    centers = []
    for i in range(1, count + 1):
        area = int((labels == i).sum())
        if area >= min_area:
            y, x = ndi.center_of_mass(mask, labels, i)
            centers.append((float(x), float(y), area))
    return centers


def input_markers(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(float)
    r, g, b = rgb.transpose(2, 0, 1)
    red = component_centers((r > 100) & (r > g * 1.4) & (r > b * 1.4), 80)
    green = component_centers((g > 80) & (g > r * 1.3) & (g > b * 1.2), 50)
    if len(red) != 4 or len(green) != 2:
        raise ValueError(f'Expected 4 red/2 green markers, found {red} / {green}')
    red = sorted(red, key=lambda c: c[1])
    red = sorted(red[:2], key=lambda c: c[0]) + sorted(red[2:], key=lambda c: c[0])
    green = sorted(green, key=lambda c: c[1])
    return np.array([[x, y] for x, y, _ in red + green])


def bc_centers(config: dict) -> np.ndarray:
    refs = []
    for kind, wanted in (('fix', 4), ('load', 2)):
        mesh = trimesh.load(config['stages']['post'][kind], force='mesh')
        parts = [part for part in mesh.split(only_watertight=False) if abs(part.volume) > 1e-8]
        if len(parts) != wanted:
            raise ValueError(f'{kind}: expected {wanted} components, got {len(parts)}')
        centers = [part.center_mass[:2] for part in parts]
        centers = sorted(centers, key=lambda c: -c[1])
        if kind == 'fix':
            centers = sorted(centers[:2], key=lambda c: c[0]) + sorted(centers[2:], key=lambda c: c[0])
        refs.extend(centers)
    return phys_to_pixel(np.array(refs))


def affine_inverse(source: np.ndarray, target: np.ndarray):
    # Target render pixel -> source image pixel.
    coeff = np.linalg.lstsq(np.c_[target, np.ones(len(target))], source, rcond=None)[0]
    forward = np.linalg.lstsq(np.c_[source, np.ones(len(source))], target, rcond=None)[0]
    residual = np.linalg.norm(np.c_[source, np.ones(len(source))] @ forward - target, axis=1)
    return lambda points: np.c_[points, np.ones(len(points))] @ coeff, residual


def warp_image(image: np.ndarray, source: np.ndarray, target: np.ndarray,
               mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == 'affine':
        inverse, residual = affine_inverse(source, target)
    elif mode == 'tps':
        inverse = RBFInterpolator(target, source, kernel='thin_plate_spline', degree=1)
        residual = np.linalg.norm(inverse(target) - source, axis=1)
    else:
        raise ValueError(mode)
    yy, xx = np.indices((N, N))
    points = np.column_stack((xx.ravel(), yy.ravel()))
    source_points = np.concatenate([inverse(chunk) for chunk in np.array_split(points, 16)])
    coords = [source_points[:, 1].reshape(N, N), source_points[:, 0].reshape(N, N)]
    warped = np.stack([ndi.map_coordinates(image[:, :, channel], coords, order=1, mode='constant', cval=255)
                       for channel in range(3)], axis=-1).astype(np.uint8)
    return warped, residual


def render_top(mesh_path: Path, output: Path) -> np.ndarray:
    mesh = pv.read(mesh_path)
    plotter = pv.Plotter(off_screen=True, window_size=(N, N))
    plotter.set_background('white')
    plotter.add_mesh(mesh, color='black', lighting=False, smooth_shading=False)
    camera_center = (CENTER[0], CENTER[1], 0.031)
    plotter.camera_position = [(camera_center[0], camera_center[1], camera_center[2] + 1),
                               camera_center, (0, 1, 0)]
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = SCALE
    rgb = plotter.screenshot(str(output))
    plotter.close()
    return np.asarray(rgb).min(axis=2) < 200


def align_mesh(source: Path, target: Path, matrix: np.ndarray) -> None:
    if target.exists():
        return
    mesh = trimesh.load(source, force='mesh', process=False)
    vertices = np.c_[mesh.vertices, np.ones(len(mesh.vertices))] @ matrix
    trimesh.Trimesh(vertices, mesh.faces, process=False).export(target)


def iou(a: np.ndarray, b: np.ndarray, region: np.ndarray | None = None) -> float:
    if region is not None:
        a, b = a & region, b & region
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def void_mask(mask: np.ndarray, exclude_bc: np.ndarray) -> np.ndarray:
    voids = ndi.binary_fill_holes(mask) & ~mask & ~exclude_bc
    labels, n = ndi.label(voids)
    sizes = np.bincount(labels.ravel())
    return (labels > 0) & (sizes[labels] >= 100)


def summary_for_pair(reference: np.ndarray, stage: np.ndarray,
                     design_region: np.ndarray, bc_mask: np.ndarray) -> dict:
    input_outer = ndi.binary_fill_holes(reference)
    stage_outer = ndi.binary_fill_holes(stage)
    input_void = void_mask(reference, bc_mask)
    stage_void = void_mask(stage, bc_mask)
    return {
        'solid_iou_design_region': round(iou(reference, stage, design_region), 4),
        'outer_iou': round(iou(input_outer, stage_outer), 4),
        'enclosed_void_iou': round(iou(input_void, stage_void), 4),
        'void_recall': round(float((input_void & stage_void).sum() / max(input_void.sum(), 1)), 4),
        'input_void_area_px': int(input_void.sum()),
        'stage_void_area_px': int(stage_void.sum()),
        'stage_solid_area_px': int((stage & design_region).sum()),
    }


def input_widths(mask: np.ndarray, region: np.ndarray, pixel_mm: float) -> dict:
    skeleton = skeletonize(mask)
    width_px = 2 * ndi.distance_transform_edt(mask)[skeleton & region]
    width_mm = width_px * pixel_mm
    pitch_mm = 3.75357917
    return {'median_mm': round(float(np.median(width_mm)), 2),
            'p10_mm': round(float(np.quantile(width_mm, 0.1)), 2),
            'fraction_centerline_under_one_dense_voxel': round(float((width_mm < pitch_mm).mean()), 4),
            'fraction_centerline_under_two_dense_voxels': round(float((width_mm < 2 * pitch_mm).mean()), 4),
            'dense_pitch_mm': pitch_mm}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matrix = transform_matrix()
    results = {'method': {'render': 'fixed physical top camera 800x800',
                          'input_mask': 'RGB minimum < 230',
                          'stage_mask': 'black unlit mesh render < 200',
                          'registration': 'six BC markers; affine sensitivity and exact thin-plate-spline',
                          'design_region': 'all pixels excluding 10 mm disks at four fix centers and 8 mm disks at two load centers',
                          'void_definition': 'enclosed white components >= 100 pixels, excluding BC disks'},
               'cases': {}}
    contact_rows = []
    for name, case in CASES.items():
        out = OUT / name
        out.mkdir(exist_ok=True)
        config = json.loads((case / 'config_full.json').read_text())
        image = np.asarray(Image.open(case / 'input/그림1.png').convert('RGB'))
        source = input_markers(image)
        target = bc_centers(config)
        aligned = {}
        registration = {}
        for mode in ('affine', 'tps'):
            rgb, residual = warp_image(image, source, target, mode)
            Image.fromarray(rgb).save(out / f'input_registered_{mode}.png')
            aligned[mode] = rgb.min(axis=2) < 230
            registration[mode] = {'marker_residual_px': [round(float(x), 2) for x in residual],
                                  'marker_rmse_px': round(float(np.sqrt(np.mean(residual ** 2))), 2)}
        yy, xx = np.indices((N, N))
        bc_mask = np.zeros((N, N), bool)
        for i, (cx, cy) in enumerate(target):
            radius_px = (10 if i < 4 else 8) * PIXELS_PER_M / 1000
            bc_mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px ** 2
        design = ~bc_mask
        stage_sources = {
            'top_dense': case / 'dense_top/mesh.obj',
            'multiview_dense': case / 'generation/mesh_dense.obj',
            'sparse_aligned': case / 'post/mesh_physical_aligned.obj',
            'final': case / 'post/final.obj',
        }
        stage_masks = {}
        for stage, mesh_source in stage_sources.items():
            mesh_path = mesh_source
            if stage in ('top_dense', 'multiview_dense'):
                mesh_path = out / f'{stage}_physical.obj'
                align_mesh(mesh_source, mesh_path, matrix)
            stage_masks[stage] = render_top(mesh_path, out / f'{stage}_top.png')
        stage_results = {}
        for mode, reference in aligned.items():
            stage_results[mode] = {
                stage: summary_for_pair(reference, mask, design, bc_mask)
                for stage, mask in stage_masks.items()}
        pairwise = {f'{first}_to_{second}': summary_for_pair(stage_masks[first], stage_masks[second], design, bc_mask)
                    for first, second in zip(STAGES, STAGES[1:])}
        # Affine x/y scale gives a conservative physical width approximation.
        widths = input_widths(aligned['affine'], design, 1000 / PIXELS_PER_M)
        results['cases'][name] = {'source_input': str(case / 'input/그림1.png'),
                                  'registration': registration,
                                  'input_marker_px': source.round(2).tolist(),
                                  'physical_bc_marker_px': target.round(2).tolist(),
                                  'input_widths_affine': widths,
                                  'image_to_stage': stage_results,
                                  'stage_to_stage': pairwise,
                                  'stage_meshes': {k: str(v) for k, v in stage_sources.items()}}
        # Fixed-size contact row with the registered image and all four geometry stages.
        images = [Image.open(out / 'input_registered_affine.png').convert('RGB')]
        images += [Image.open(out / f'{stage}_top.png').convert('RGB') for stage in STAGES]
        row = Image.new('RGB', (N * len(images), N + 45), 'white')
        draw = ImageDraw.Draw(row)
        for i, (label, image_item) in enumerate(zip(('input',) + STAGES, images)):
            row.paste(image_item, (i * N, 45))
            draw.text((i * N + 15, 12), f'{name}: {label}', fill='black')
        row.save(out / 'stage_contact.png')
        contact_rows.append(row)
    all_contact = Image.new('RGB', (N * 5, (N + 45) * len(contact_rows)), 'white')
    for i, row in enumerate(contact_rows):
        all_contact.paste(row, (0, i * (N + 45)))
    all_contact.save(OUT / 'all_stage_contact.png')
    (OUT / 'metrics.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
    print(OUT / 'metrics.json')


if __name__ == '__main__':
    main()
