"""Isolated +15.8 mm outward chair design envelope with unchanged BC patches."""
from __future__ import annotations

import json
import shutil

import numpy as np
from scipy.ndimage import map_coordinates
import trimesh

from chair_qd_long_protocol_2026_10_03 import BASE, OUT as MAIN
from build_chair_single_view_spec_2026_10_03 import box

OLD = BASE / 'single_view_spec_2026-10-03'
OUT = MAIN / 'envelope_plus16_spec_2026-10-03'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old = np.load(OLD / 'voxel.npz')
    origin, pitch = old['origin'], old['pitch_xyz']
    margin = float(pitch[0])
    # Expand only outward-facing surfaces. The inward edge of the back/side
    # subdomains stays fixed so the occupant keepout is never swallowed.
    x_outer = .285 + margin
    y_front = -.265 - margin
    y_back = .295 + margin
    parts = [
        box(((-x_outer, x_outer), (y_front, y_back), (0, .610))),
        box(((-x_outer, x_outer), (.105, y_back), (.550, .945 + margin))),
        box(((-x_outer, -.170), (y_front, y_back), (.550, .790 + margin))),
        box(((.170, x_outer), (y_front, y_back), (.550, .790 + margin))),
    ]
    envelope = trimesh.boolean.union(parts, engine='manifold')
    if not isinstance(envelope, trimesh.Trimesh) or not envelope.is_watertight:
        raise RuntimeError('Expanded envelope Boolean is invalid')
    envelope.export(OUT / 'envelope.stl')
    x = origin[0] + (np.arange(64) + .5) * pitch[0]
    y = origin[1] + (np.arange(64) + .5) * pitch[1]
    z = origin[2] + (np.arange(64) + .5) * pitch[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    env = (((abs(X) <= x_outer) & (Y >= y_front) & (Y <= y_back) & (Z >= 0) & (Z <= .610)) |
           ((abs(X) <= x_outer) & (Y >= .105) & (Y <= y_back) & (Z >= .550) & (Z <= .945 + margin)) |
           ((abs(X) >= .170) & (abs(X) <= x_outer) & (Y >= y_front) & (Y <= y_back) &
            (Z >= .550) & (Z <= .790 + margin)))
    keepout = old['keepout'].astype(bool)
    bc = old['bc'].astype(bool)
    if np.any(env & keepout):
        raise RuntimeError('Expanded envelope crosses occupant keepout')
    if np.any(bc & ~env):
        raise RuntimeError('Expanded envelope excludes a BC patch')
    np.savez_compressed(OUT / 'voxel.npz',
        bracket=env, bc=bc, design=env & ~bc,
        fix=old['fix'], load=old['load'], back_load=old['back_load'],
        keepout=old['keepout'], origin=origin, pitch_xyz=pitch,
        pitch=float(pitch[0]))
    registration = json.loads((OLD / 'specification.json').read_text())['native_frame_registration']
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_euler('x', registration['rotation_x_degrees'], degrees=True).as_matrix()
    native_xyz = np.stack((X, Y, Z), axis=-1)
    physical_xyz = ((native_xyz - np.asarray(registration['source_center_m'])) @ rotation.T *
                    registration['uniform_scale'] + np.asarray(registration['physical_center_m']))
    index = ((physical_xyz - origin) / pitch - .5).transpose(3, 0, 1, 2)
    native = {}
    for key in ('bracket', 'fix', 'load', 'back_load', 'keepout'):
        source = env if key == 'bracket' else old[key].astype(bool)
        native[key] = map_coordinates(source.astype(np.uint8), index,
                                      order=0, mode='constant', cval=0) > 0
    native['bc'] = native['fix'] | native['load'] | native['back_load']
    native['design'] = native['bracket'] & ~native['bc']
    if np.any(native['bc'] & ~native['bracket']):
        raise RuntimeError('Registered BC leaves the expanded native envelope')
    np.savez_compressed(OUT / 'native_frame_spec.npz', **native,
                        origin=origin, pitch_xyz=pitch, pitch=float(pitch[0]))
    for name, load_source in [('fea_domain', OLD / 'fea_domain/load.stl'),
                              ('fea_domain_back', OLD / 'fea_domain_back/load.stl')]:
        folder = OUT / name
        folder.mkdir(exist_ok=True)
        shutil.copy2(OUT / 'envelope.stl', folder / 'original_DesignSpace.stl')
        shutil.copy2(OLD / 'fea_domain/fixed.stl', folder / 'fixed.stl')
        shutil.copy2(load_source, folder / 'load.stl')
    manifest = {'status': 'isolated expanded chair spec; BC unchanged',
                'parent_spec': str(OLD), 'margin_mm': margin * 1000,
                'expansion': 'outer x/y and upper z surfaces only; inner keepout-facing planes fixed',
                'envelope_voxels_old': int(old['bracket'].sum()),
                'envelope_voxels_new': int(env.sum()),
                'bc_identical': bool(np.array_equal(bc, old['bc'])),
                'keepout_overlap_voxels': int((env & keepout).sum()),
                'files': {key: str(OUT / key) for key in ('envelope.stl', 'voxel.npz',
                                                         'native_frame_spec.npz', 'fea_domain', 'fea_domain_back')}}
    (OUT / 'specification.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
