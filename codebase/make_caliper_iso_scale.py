#!/usr/bin/env python3
"""Prepare an isotropic, aspect-preserving caliper grid with less outer padding."""
import json
import subprocess
from pathlib import Path

ROOT = Path('/home/goya/SDL/3d_qd')
OUT = ROOT / 'data_real/caliper_iso_margin01'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # No mesh transform: this raises the effective isotropic scale solely by
    # reducing the empty cube margin 5% -> 1%, retaining the original aspect.
    cmd = ['/home/goya/miniconda3/envs/direct3ds2/bin/python',
           str(ROOT / 'codebase/code/make_bc_proper_pysdf.py'),
           '--bracket', str(ROOT / 'data_real/caliper/original_DesignSpace.stl'),
           '--fix', str(ROOT / 'data_real/caliper/fixed.stl'),
           '--load', str(ROOT / 'data_real/caliper/load.stl'),
           '--res', '64', '--margin', '.01', '--dilate-bracket', '1',
           '--out', str(OUT / 'voxel.npz')]
    subprocess.run(cmd, check=True, cwd=ROOT)
    cfg = json.loads((ROOT / 'codebase/configs/caliper.json').read_text())
    cfg['name'] = 'caliper_iso_margin01'
    cfg['stages']['mesh']['bc_proper'] = str(OUT.relative_to(ROOT) / 'voxel.npz')
    cfg['stages']['mesh']['bracket_occ'] = str(OUT.relative_to(ROOT) / 'voxel.npz')
    cfg['stages']['mesh'].update({'fea_w': 0.0, 'sp_fea_w': 0.0})
    # Geometry/conditioning/BC meshes deliberately remain the original caliper.
    (ROOT / 'codebase/configs/caliper_iso_margin01.json').write_text(json.dumps(cfg, indent=2) + '\n')


if __name__ == '__main__':
    main()
