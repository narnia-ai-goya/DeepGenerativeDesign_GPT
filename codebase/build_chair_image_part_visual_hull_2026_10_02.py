#!/usr/bin/env python3
"""Image-first chair parts: two-view masks, symmetric frame, shared-BC assembly.

The sources are imagegen edits of the original chair images. No existing 3D
chair mesh or model training is used to construct these parts.  Explicit
pixel-to-metre registration is recorded below for reproducibility.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import measure

from make_chair_domain import ROOT
from validate_chair_bc_geometry import audit

sys.path.insert(0, str(ROOT / 'codebase/code/conditioning'))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402

BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
OUT = BASE / 'semantic_parts_from_image_2026-10-02'
IMAGES = OUT / 'inputs'
BC = BASE / 'consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
PITCH = 0.004


def foreground(filename: str) -> np.ndarray:
    rgb = np.asarray(Image.open(IMAGES / filename).convert('RGB'))
    mask = rgb.min(axis=2) < 225
    # The white background may contain weak shadows. Keep the product component.
    labels, n = ndimage.label(mask)
    if not n:
        raise RuntimeError(f'No foreground in {filename}')
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    mask = labels == np.argmax(counts)
    return ndimage.binary_closing(mask, iterations=1)


def lookup(mask: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    ix = np.rint(px).astype(int).clip(0, mask.shape[1] - 1)
    iy = np.rint(py).astype(int).clip(0, mask.shape[0] - 1)
    return mask[iy, ix]


def to_mesh(occ: np.ndarray, origin: np.ndarray) -> trimesh.Trimesh:
    field = np.pad(occ.astype(np.uint8), 1)
    vertices, faces, *_ = measure.marching_cubes(field, level=.5, spacing=(PITCH,) * 3)
    vertices += origin - PITCH
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def sample_bc(bc: np.lib.npyio.NpzFile, key: str, xyz: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    origin = bc['origin']
    pitch = bc['pitch_xyz']
    indices = [np.floor((axis - origin[d]) / pitch[d]).astype(int) for d, axis in enumerate(xyz)]
    valid = [((idx >= 0) & (idx < 64)) for idx in indices]
    indices = [idx.clip(0, 63) for idx in indices]
    return (bc[key][indices[0][:, None, None], indices[1][None, :, None], indices[2][None, None, :]] &
            valid[0][:, None, None] & valid[1][None, :, None] & valid[2][None, None, :])


def main() -> None:
    out = OUT / 'visual_hull_assembly'
    out.mkdir(parents=True, exist_ok=True)
    x = np.arange(-.28 + PITCH / 2, .28, PITCH)
    y = np.arange(-.26 + PITCH / 2, .28, PITCH)
    z = np.arange(0 + PITCH / 2, .92, PITCH)
    origin = np.array([x[0], y[0], z[0]])

    side = foreground('side_frame_imagegen.png')
    front = foreground('seat_back_imagegen.png')
    central_side = foreground('seat_back_right_imagegen.png')

    # Side frame: front/rear foot centres at image x≈380/820 map to y≈−.187/+ .187 m.
    px_side = 600 + y / .00085
    py_side = 1225 - z / .92 * 1125
    side_yz = lookup(side, px_side[:, None], py_side[None, :])
    left = ((x[:, None, None] >= -.225) & (x[:, None, None] <= -.185)) & side_yz[None, :, :]
    right = left[::-1, :, :].copy()  # exact geometric mirror about x=0.

    # Center body visual hull. Piecewise z registration preserves thin seat and
    # the high backrest instead of using each image's full bounding box.
    px_front = 625 + x / .19 * 400
    py_front = np.interp(z, [.455, .52, .65, .92], [1010, 765, 620, 245], left=1254, right=0)
    front_xz = lookup(front, px_front[:, None], py_front[None, :])
    px_cside = 630 + y / .18 * 485
    py_cside = np.interp(z, [.455, .52, .65, .92], [960, 765, 620, 210], left=1254, right=0)
    cside_yz = lookup(central_side, px_cside[:, None], py_cside[None, :])
    center = front_xz[:, None, :] & cside_yz[None, :, :]

    bc = np.load(BC)
    fix = sample_bc(bc, 'fix', (x, y, z))
    seat_load = sample_bc(bc, 'load', (x, y, z))
    back_load = sample_bc(bc, 'back_load', (x, y, z))
    left |= fix & (x[:, None, None] < 0)
    right |= fix & (x[:, None, None] > 0)
    center |= seat_load | back_load
    raw = center | left | right
    interface_left = int(np.count_nonzero(center & left))
    interface_right = int(np.count_nonzero(center & right))
    print('interface overlap voxels', interface_left, interface_right, flush=True)

    # SNAP3D-inspired contact check. Repair only when there is insufficient
    # geometric overlap; here the selected shared interfaces already overlap.
    if min(interface_left, interface_right) < 500:
        seam = (np.abs(x[:, None, None] + .19) <= .012) | (np.abs(x[:, None, None] - .19) <= .012)
        near = ndimage.binary_dilation(raw, structure=ndimage.generate_binary_structure(3, 1))
        assembled = np.where(seam, near, raw)
        repair_applied = True
    else:
        assembled = raw.copy()
        repair_applied = False
    bracket = sample_bc(bc, 'bracket', (x, y, z))
    clipped = assembled & (bracket | fix | seat_load | back_load)
    parts = {'center': center, 'side_left': left, 'side_right_mirrored': right,
             'assembled_raw': raw, 'assembled_contact_repaired': assembled,
             'assembled_domain_clipped': clipped}
    np.savez_compressed(out / 'occupancy.npz', origin=origin, pitch=PITCH,
                        assembled_contact_repaired=assembled,
                        assembled_domain_clipped=clipped)
    metrics = {'source_images': {name: str(IMAGES / name) for name in
              ('side_frame_imagegen.png', 'seat_back_imagegen.png', 'seat_back_right_imagegen.png')},
              'pitch_m': PITCH, 'side_is_mirrored': True,
              'contact_repair_applied': repair_applied,
              'interface_overlap_voxels': {'left': interface_left, 'right': interface_right},
              'parts': {}}
    for name, occ in parts.items():
        labels, n = ndimage.label(occ, structure=ndimage.generate_binary_structure(3, 1))
        mesh = to_mesh(occ, origin)
        path = out / f'{name}.obj'
        mesh.export(path)
        mesh.export(out / f'{name}.glb')
        metrics['parts'][name] = {'obj': str(path), 'volume_liters': float(occ.sum() * PITCH**3 * 1000),
                                  '6_connected_components': int(n),
                                  'largest_fraction': float(np.bincount(labels.ravel())[1:].max() / occ.sum()),
                                  'watertight': bool(mesh.is_watertight)}
    metrics['bc_audit'] = audit(out / 'assembled_contact_repaired.obj', BC)
    metrics['bc_audit_clipped'] = audit(out / 'assembled_domain_clipped.obj', BC)
    metrics['outside_envelope_fraction'] = float(np.count_nonzero(assembled & ~bracket) / assembled.sum())
    metrics['clipped_components'] = metrics['parts']['assembled_domain_clipped']['6_connected_components']
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')

    domain = trimesh.load(ROOT / 'data_real/chair/original_DesignSpace.stl', force='mesh')
    cam_center = domain.bounds.mean(axis=0)
    radius = float(np.linalg.norm(domain.extents)) * 1.5
    fit = float(domain.extents.max()) / 2
    views = [('front', 15, 0), ('right', 15, 90), ('top', 85, 0)]
    tile, head = 360, 28
    sheet = Image.new('RGB', (tile * 3, (tile + head) * 5), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(('center', 'side_left', 'side_right_mirrored',
                                'assembled_contact_repaired', 'assembled_domain_clipped')):
        mesh = trimesh.load(out / f'{name}.obj', force='mesh', process=False)
        for col, (label, elev, azim) in enumerate(views):
            eye, up = camera_from_elev_azim(cam_center, radius, elev, azim)
            rgb = render_lit(mesh, eye, cam_center, up, size=tile,
                             fit_extent=fit, margin=1.15, color=(.52, .57, .62))
            sheet.paste(Image.fromarray(rgb), (col * tile, row * (tile + head) + head))
            draw.text((col * tile + 8, row * (tile + head) + 5), f'{name} / {label}', fill='black')
    sheet.save(out / 'comparison.png')
    print(json.dumps({'metric_path': str(out / 'metrics.json'),
                      'components': metrics['parts']['assembled_contact_repaired']['6_connected_components'],
                      'bc_pass': metrics['bc_audit']['bc_geometry_pass'],
                      'clipped_components': metrics['clipped_components'],
                      'clipped_bc_pass': metrics['bc_audit_clipped']['bc_geometry_pass'],
                      'outside_envelope_fraction': metrics['outside_envelope_fraction']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
