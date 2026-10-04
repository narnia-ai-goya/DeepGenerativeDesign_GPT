#!/usr/bin/env python3
"""Build a physical chair specification registered to the best single-view mesh.

The pretrained Direct3D-S2 mesh supplies only functional landmarks (four feet,
seat height, backrest location).  The envelope is a broad, separate design
domain, so future shape exploration is not restricted to the exact mesh.
Coordinates are metres: X left/right, Y front/back (+Y is back), Z up.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SOURCE = BASE / 'minimal_direct3ds2_2026-10-02/aligned_sparse_main.obj'
OUT = BASE / 'single_view_spec_2026-10-03'
OLD_GRID = BASE / 'consistent_reference_2026-09-28/backrest_load_ablation/voxel.npz'
sys.path.insert(0, str(ROOT / 'codebase/code'))
from make_bc_proper_pysdf import voxel_centers_inside  # noqa: E402


def box(limits: tuple[tuple[float, float], ...]) -> trimesh.Trimesh:
    lo = np.array([v[0] for v in limits])
    hi = np.array([v[1] for v in limits])
    shape = trimesh.creation.box(extents=hi-lo)
    shape.apply_translation((lo+hi)/2)
    return shape


def sloped_back_load(xlim: tuple[float, float], zlim: tuple[float, float]) -> trimesh.Trimesh:
    # Seat occupant pushes the front face of the backrest toward +Y.
    # The panel slopes rearward with increasing height.
    z0, z1 = zlim
    def mid(z: float) -> float:
        return .225 + .20 * (z - .75)
    vertices = np.array([(x, mid(z)+dy, z)
                         for x in xlim for z in (z0, z1) for dy in (-.009, .009)])
    hull = trimesh.convex.convex_hull(vertices)
    return hull


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = trimesh.load(SOURCE, force='mesh', process=False)
    verts = source.vertices
    if len(source.split(only_watertight=False)) != 1:
        raise ValueError('The registered reference is expected to have one component.')

    # Measure each physical foot from the low, four-quadrant vertex cloud.
    low = verts[(verts[:, 2] >= .025) & (verts[:, 2] < .075)]
    foot_centers = []
    for xs in (-1, 1):
        for ys in (-1, 1):
            group = low[(low[:, 0] * xs > .12) & (low[:, 1] * ys > .11)]
            if len(group) < 100:
                raise ValueError(f'Foot landmark absent in quadrant {xs},{ys}')
            foot_centers.append((float(np.median(group[:, 0])),
                                 float(np.median(group[:, 1]))))
    seat_cloud = verts[(np.abs(verts[:, 0]) < .13) &
                       (verts[:, 1] > -.12) & (verts[:, 1] < .08) &
                       (verts[:, 2] > .54) & (verts[:, 2] < .65)]
    seat_top = float(np.percentile(seat_cloud[:, 2], 90))

    # Expand the observed chair extents by roughly 8% in the horizontal plane.
    # Keep the under-seat, back and side-wing design subdomains explicit.
    env_parts = [
        box(((-.285, .285), (-.265, .295), (0, .610))),
        box(((-.285, .285), (.105, .295), (.550, .945))),
        box(((-.285, -.170), (-.265, .295), (.550, .790))),
        box(((.170, .285), (-.265, .295), (.550, .790))),
    ]
    envelope = trimesh.boolean.union(env_parts, engine='manifold')
    if not isinstance(envelope, trimesh.Trimesh):
        raise RuntimeError('Envelope union failed')

    feet = []
    for x, y in foot_centers:
        foot = trimesh.creation.cylinder(radius=.038, height=.030, sections=64)
        foot.apply_translation((x, y, .020))
        feet.append(foot)
    fixed = trimesh.util.concatenate(feet)
    # In this volumetric solver the load volume sits just below the contact
    # surface, inside the seat slab, rather than floating above the surface.
    seat_load = box(((-.150, .150), (-.115, .080),
                     (seat_top-.037, seat_top-.008)))
    back_load = sloped_back_load((-.080, .080), (.720, .830))
    keepout = box(((-.165, .165), (-.140, .100), (.610, .900)))

    outputs = {'envelope': envelope, 'fixed': fixed, 'seat_load': seat_load,
               'back_load': back_load, 'occupant_keepout': keepout}
    for name, shape in outputs.items():
        shape.export(OUT / f'{name}.stl')

    # Keep the existing world grid for direct comparison with the earlier BC.
    old = np.load(OLD_GRID)
    origin, pitch = old['origin'], old['pitch_xyz']
    R = 64
    x = origin[0] + (np.arange(R)+.5)*pitch[0]
    y = origin[1] + (np.arange(R)+.5)*pitch[1]
    z = origin[2] + (np.arange(R)+.5)*pitch[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    envelope_mask = (((abs(X) <= .285) & (Y >= -.265) & (Y <= .295) & (Z >= 0) & (Z <= .610)) |
                     ((abs(X) <= .285) & (Y >= .105) & (Y <= .295) & (Z >= .550) & (Z <= .945)) |
                     ((abs(X) >= .170) & (abs(X) <= .285) & (Y >= -.265) & (Y <= .295) &
                      (Z >= .550) & (Z <= .790)))
    fix_masks = [((X-xc)**2+(Y-yc)**2 <= .038**2) & (Z >= .005) & (Z <= .035)
                 for xc, yc in foot_centers]
    fix_mask = np.logical_or.reduce(fix_masks)
    seat_mask = ((abs(X) <= .150) & (Y >= -.115) & (Y <= .080) &
                 (Z >= seat_top-.037) & (Z <= seat_top-.008))
    back_mid = .225 + .20*(Z-.75)
    back_mask = ((abs(X) <= .080) & (Z >= .720) & (Z <= .830) &
                 (abs(Y-back_mid) <= .009))
    keepout_mask = ((abs(X) <= .165) & (Y >= -.140) & (Y <= .100) &
                    (Z >= .610) & (Z <= .900))
    bc = fix_mask | seat_mask | back_mask
    if np.any(bc & ~envelope_mask) or np.any(keepout_mask & envelope_mask):
        raise ValueError('Functional patches or occupant keepout conflict with envelope')
    reference = voxel_centers_inside(source, R, origin, pitch)
    patch_metrics = {}
    for name, patch in [('seat_load', seat_mask), ('back_load', back_mask),
                        *[(f'foot_{i+1}', f) for i, f in enumerate(fix_masks)]]:
        n = int(patch.sum())
        patch_metrics[name] = {'target_voxels': n,
                               'covered_voxels': int((patch & reference).sum()),
                               'coverage': float((patch & reference).sum()/n) if n else 0.}
    if any(v['target_voxels'] == 0 for v in patch_metrics.values()):
        raise ValueError('One or more BC patches vanish at the comparison-grid resolution')
    np.savez_compressed(OUT / 'voxel.npz', bracket=envelope_mask, bc=bc,
                        design=envelope_mask & ~bc, fix=fix_mask, load=seat_mask,
                        back_load=back_mask, keepout=keepout_mask,
                        origin=origin, pitch_xyz=pitch, pitch=float(pitch[0]))
    # The model's natural chair coordinates differ from physical Z-up.  Reuse
    # the independently measured rigid calibration from the earlier single-
    # view diagnosis so guidance never sees mismatched physical-frame masks.
    registration = json.loads((BASE / 'registered_spec_2026-10-02/calibration.json').read_text())
    transform = registration['native_to_physical']
    source_center = np.array(transform['source_center_m'])
    physical_center = np.array(transform['physical_center_m'])
    rotation = Rotation.from_euler('x', transform['rotation_x_degrees'], degrees=True).as_matrix()
    scale = float(transform['uniform_scale'])
    native_xyz = np.stack((X, Y, Z), axis=-1)
    physical_xyz = (native_xyz-source_center) @ rotation.T * scale + physical_center
    index = ((physical_xyz-origin)/pitch-.5).transpose(3, 0, 1, 2)
    physical_fields = {'bracket': envelope_mask, 'fix': fix_mask,
                       'load': seat_mask, 'back_load': back_mask,
                       'keepout': keepout_mask}
    native_fields = {name: map_coordinates(mask.astype(np.uint8), index,
                     order=0, mode='constant', cval=0) > 0
                     for name, mask in physical_fields.items()}
    native_fields['bc'] = native_fields['fix'] | native_fields['load'] | native_fields['back_load']
    native_fields['design'] = native_fields['bracket'] & ~native_fields['bc']
    if np.any(native_fields['bc'] & ~native_fields['bracket']):
        raise ValueError('Registered BC exits registered envelope')
    np.savez_compressed(OUT / 'native_frame_spec.npz', **native_fields,
                        origin=origin, pitch_xyz=pitch, pitch=float(pitch[0]))
    native_raw = trimesh.load(BASE / 'minimal_direct3ds2_2026-10-02/generation/mesh.obj',
                              force='mesh', process=False)
    native_reference = voxel_centers_inside(native_raw, R, origin, pitch)
    native_coverage = {name: float((native_reference & native_fields[name]).sum() /
                                   native_fields[name].sum())
                       for name in ('fix', 'load', 'back_load')}
    manifest = {
        'name': 'single_view_registered_chair_spec_v1', 'status': 'geometry/BC pilot, no FEA',
        'units': 'm', 'axes': {'x': 'left-right', 'y': 'front-back (+Y back)', 'z': 'up'},
        'source_mesh': str(SOURCE), 'reference_mesh_bounds_m': source.bounds.tolist(),
        'foot_centers_m': foot_centers, 'seat_top_m': seat_top,
        'envelope_bounds_m': envelope.bounds.tolist(),
        'envelope_volume_liters': float(envelope.volume*1000),
        'reference_outside_envelope_voxels': int((reference & ~envelope_mask).sum()),
        'reference_occupied_voxels': int(reference.sum()),
        'functional_patch_coverage': patch_metrics,
        'load_cases_proposed': [
            {'name': 'seated_person', 'force_N': 800, 'direction': [0, 0, -1],
             'patch': 'seat_load'},
            {'name': 'backrest_push', 'force_N': 200, 'direction': [0, 1, 0],
             'patch': 'back_load'},
        ],
        'fix': 'four floor contact pads measured from the reference mesh',
        'envelope_note': 'broad material permission, approximately 8% outer horizontal clearance; not a dilation of the mesh',
        'grid': {'resolution': R, 'origin_m': origin.tolist(), 'pitch_xyz_m': pitch.tolist()},
        'native_frame_registration': transform,
        'native_reference_coverage': native_coverage,
        'native_reference_outside_envelope_voxels': int((native_reference & ~native_fields['bracket']).sum()),
        'files': {name: str(OUT / f'{name}.stl') for name in outputs} |
                 {'voxel': str(OUT / 'voxel.npz'),
                  'native_frame_spec': str(OUT / 'native_frame_spec.npz')},
    }
    (OUT / 'specification.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'output': str(OUT), 'seat_top_m': seat_top,
                      'patches': patch_metrics,
                      'reference_outside_envelope_voxels': manifest['reference_outside_envelope_voxels']},
                     indent=2))


if __name__ == '__main__':
    main()
