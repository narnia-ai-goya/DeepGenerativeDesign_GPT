#!/usr/bin/env python3
"""Post-only caliper ablation: isolate clip-wedge cleanup from generation."""
import copy
import json
from pathlib import Path

from run_from_image import ROOT, DATA_ROOT, PY_D3D, CODE, run


SOURCE = (ROOT.parent / 'experiments/caliper_parameter_sweep_2026-09-17'
          / 'caliper_parameter_sweep')
OUT = ROOT.parent / 'experiments/caliper_tearing_ablation_2026-09-17' / 'post_only'


def main():
    base = json.loads((ROOT / 'configs/caliper.json').read_text())
    # Two representative inputs distinguish a small boolean wedge from genuine
    # sparse fragmentation (control is 2 raw components; guarded_mid is 149).
    for source in ('control', 'guarded_mid'):
        raw = SOURCE / source / 'gen/mesh.obj'
        if not raw.exists():
            raise FileNotFoundError(raw)
        for radius_mm in (0.4, 0.6):
            case = OUT / f'{source}_wedge{int(radius_mm * 100):02d}'
            case.mkdir(parents=True, exist_ok=True)
            cfg = copy.deepcopy(base)
            cfg['name'] = f'caliper_{source}_wedge{radius_mm:.1f}'
            cfg['stages']['post'].update({'wedge_trim': True,
                                          'wedge_radius': radius_mm / 1000.0})
            cp = case / 'config.json'
            cp.write_text(json.dumps(cfg, indent=2) + '\n')
            hyb, final = case / 'hybrid.obj', case / 'final.obj'
            print(f'## {source} / wedge {radius_mm:.1f} mm', flush=True)
            rc = run([PY_D3D, str(CODE / 'post_hybrid_union_clip.py'), '--config', str(cp),
                      '--in', str(raw), '--out', str(hyb)], case / 'post.log', cwd=str(DATA_ROOT))
            if rc:
                print(f'  post FAILED rc={rc}', flush=True); continue
            rc = run([PY_D3D, str(CODE / 'surface_remesh_pre.py'), '--config', str(cp),
                      '--in', str(hyb), '--out', str(final)], case / 'remesh.log', cwd=str(DATA_ROOT))
            print(f'  {"OK" if rc == 0 and final.exists() else "FAILED"}: {final}', flush=True)


if __name__ == '__main__':
    main()
