"""Weld + isotropic-remesh the domain/BC STLs into clean watertight equilateral-triangle meshes.

Raw envelope/fix/load STLs are typically unwelded triangle soup (V == 3*F, every
triangle independent) and, for low-poly pegs, visibly faceted. Feeding those into
post_hybrid_union_clip lets the facets/soup show up as fan/wrinkle artifacts at the
peg↔body weld. This step is a one-time preprocessing that produces *_remesh.stl:

  1. weld coincident vertices (merge_close_vertices) → watertight
  2. isotropic explicit remeshing → uniform (near-equilateral) triangles, smooths facets

Output: <domain-dir>/<basename>_remesh.stl for the envelope (= design-domain) / fix / load STLs
  — i.e. the `*_remesh.stl` files shipped under data/<domain>/, which stages.post reads as
  `bracket_stl` / `fix` / `load` (post_hybrid_union_clip --bracket-stl/--fix/--load).

One-time input preparation, NOT part of a reproduction run: the outputs are already shipped.
The commands below are the ones the shipped files were produced with (see code/README.md,
"Input preparation"); the envelope basename is `original_DesignSpace`, not the default
`envelope`, so pass it explicitly.

Usage:
  # from the package root, generation env active (needs pymeshlab)
  python code/remesh_domain_bc.py --domain-dir data/motor_mount \
      --envelope original_DesignSpace --edge-envelope-mm 1.5 --edge-peg-mm 1.0
"""
import argparse
import sys
from pathlib import Path
import pymeshlab

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_config import apply_config


def remesh_one(src, dst, edge_mm, iters=12):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(src))
    # weld soup → watertight
    ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(0.001))
    ms.meshing_remove_duplicate_faces()
    for f in ('meshing_remove_duplicate_vertices', 'meshing_remove_null_faces',
              'meshing_remove_folded_faces', 'meshing_repair_non_manifold_edges'):
        try:
            getattr(ms, f)()
        except Exception:
            pass
    # isotropic → equilateral triangles (also smooths low-poly facets)
    ms.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PureValue(edge_mm / 1000.0), iterations=iters, adaptive=False)
    for f in ('meshing_remove_null_faces', 'meshing_repair_non_manifold_edges'):
        try:
            getattr(ms, f)()
        except Exception:
            pass
    ms.save_current_mesh(str(dst))
    m = ms.current_mesh()
    return m.vertex_number(), m.face_number()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None, help='run config JSON; stages.prep fills envelope / '
                                                  'edge_envelope_mm / edge_peg_mm (CLI still overrides)')
    ap.add_argument('--domain-dir', required=True, help='dir holding envelope/fixed/load .stl')
    ap.add_argument('--envelope', default='original_DesignSpace', help='envelope (design-domain) stl basename (no ext)')
    ap.add_argument('--fix', default='fixed')
    ap.add_argument('--load', default='load')
    ap.add_argument('--edge-envelope-mm', type=float, default=1.5)
    ap.add_argument('--edge-peg-mm', type=float, default=1.0)
    ap.add_argument('--suffix', default='_remesh', help='output basename suffix')
    apply_config(ap, stage='prep')      # stages.prep → argparse defaults (edges / envelope basename)
    args = ap.parse_args()
    d = Path(args.domain_dir)
    jobs = [(args.envelope, args.edge_envelope_mm), (args.fix, args.edge_peg_mm), (args.load, args.edge_peg_mm)]
    for nm, edge in jobs:
        src = d / f'{nm}.stl'
        if not src.exists():
            print(f'  skip {src} (missing)', flush=True)
            continue
        dst = d / f'{nm}{args.suffix}.stl'
        v, f = remesh_one(src, dst, edge)
        print(f'  {dst.name}: edge~{edge}mm  V={v:,} F={f:,}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
