#!/usr/bin/env python3
"""Prepare the t=0.5 anisotropically stretched caliper generation domain.

The generator operates in a normalized isotropic cube.  Stretch the *entire*
domain before voxelization (envelope, fix and load together), rather than
stretching a generated mesh afterwards, so 64³ dense coordinates and the
conditioning renders remain in the same frame.  t=0.5 interpolates between
the identity and the full equalizing scale used by the existing t=0.3 domain.
"""
import subprocess
import json
from pathlib import Path
import numpy as np
import trimesh

ROOT = Path('/home/goya/SDL/3d_qd')
SRC = ROOT / 'data_real/caliper'
DST = ROOT / 'data_real/caliper_stretch50'
T = 0.5
FULL = np.array([2.063356267355791, 1.0, 2.69108271744613])
SCALE = 1.0 + T * (FULL - 1.0)


def main():
    DST.mkdir(parents=True, exist_ok=True)
    M = np.eye(4); M[0, 0], M[1, 1], M[2, 2] = SCALE
    for name in ('original_DesignSpace.stl', 'fixed.stl', 'load.stl'):
        mesh = trimesh.load(SRC / name, force='mesh', process=False)
        mesh.apply_transform(M)
        mesh.export(DST / name)
    np.save(DST / 'stretch_scale.npy', SCALE)
    # Keep the ordinary caliper cavity semantics: this source load STL denotes
    # the piston-bore volume and must be subtracted in post-processing.
    cfg = json.loads((ROOT / 'codebase/configs/caliper.json').read_text())
    text = json.dumps(cfg).replace('caliper/', 'caliper_stretch50/')
    cfg = json.loads(text)
    cfg['name'] = 'caliper_stretch50'
    cfg['stretch_scale'] = SCALE.tolist()
    for section in ('mesh', 'post', 'fea'):
        for key in ('fix_stl', 'load_stl', 'fix', 'load', 'bracket_stl'):
            value = cfg['stages'].get(section, {}).get(key)
            if isinstance(value, str):
                cfg['stages'][section][key] = value.replace('_remesh.stl', '.stl')
    (ROOT / 'codebase/configs/caliper_stretch50.json').write_text(json.dumps(cfg, indent=2) + '\n')
    cmd = [
        '/home/goya/miniconda3/envs/direct3ds2/bin/python',
        str(ROOT / 'codebase/code/make_bc_proper_pysdf.py'),
        '--bracket', str(DST / 'original_DesignSpace.stl'),
        '--fix', str(DST / 'fixed.stl'), '--load', str(DST / 'load.stl'),
        '--res', '64', '--margin', '.05', '--dilate-bracket', '1',
        '--out', str(DST / 'voxel.npz'),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    data = np.load(DST / 'voxel.npz')
    print(f'[done] scale={SCALE.tolist()} envelope={int(data["bracket"].sum())} '
          f'({100*data["bracket"].mean():.2f}% of 64³) -> {DST}')


if __name__ == '__main__':
    main()
