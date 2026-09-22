"""Standalone surface-remesh pass BEFORE FEA prep.

The mesh-level hybrid boolean leaves sliver triangles along the collar/body/
envelope seams; on some styles these break even the downstream remesher
(gmsh PLC faults at every escalation). This pass rebuilds a clean, uniform
surface first:

    cleanup (null/folded/T-vertex/non-manifold) -> isotropic remesh (uniform)
    -> cleanup -> largest component -> pymeshfix -> watertight STL

Usage:
    python surface_remesh_pre.py --in mesh_hybrid.obj --out mesh_fea_ready.stl \
        [--edge-mm 1.5]
"""
import argparse
from pathlib import Path
import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None, help='run config JSON; stages.post fills edge_mm / laplacian_*')
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--edge-mm', type=float, default=1.5)
    ap.add_argument('--laplacian-iters', type=int, default=1, help='Laplacian smoothing iterations (0 = off)')
    ap.add_argument('--laplacian-lamb', type=float, default=0.5, help='Laplacian smoothing lambda')
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_config import apply_config
    apply_config(ap, stage='post')          # stages.post → edge_mm / laplacian_iters / laplacian_lamb
    args = ap.parse_args()

    import pymeshlab, pymeshfix
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(args.inp)

    def _try(name, **kw):
        try:
            getattr(ms, name)(**kw)
            return True
        except Exception:
            return False

    ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(0.001))
    ms.meshing_remove_duplicate_faces()
    _try('meshing_remove_duplicate_vertices')
    _try('meshing_remove_null_faces')
    _try('meshing_remove_folded_faces')
    _try('meshing_remove_t_vertices')
    _try('meshing_remove_unreferenced_vertices')
    _try('meshing_repair_non_manifold_edges')
    _try('meshing_repair_non_manifold_vertices')
    _try('meshing_remove_connected_component_by_diameter',
         mincomponentdiag=pymeshlab.PercentageValue(5.0))
    # paper ref: Supplementary, verification chain, isotropic remesh (~1.5 mm) to a clean
    #            watertight surface.
    ms.meshing_isotropic_explicit_remeshing(
        iterations=10, targetlen=pymeshlab.PureValue(args.edge_mm / 1000.0), adaptive=False)
    _try('meshing_remove_null_faces')
    _try('meshing_remove_folded_faces')
    _try('meshing_repair_non_manifold_edges')
    _try('meshing_repair_non_manifold_vertices')
    _try('meshing_close_holes', maxholesize=2000)

    tmp = str(Path(args.out).with_suffix('.tmp.stl'))
    ms.save_current_mesh(tmp)
    t = trimesh.load(tmp, force='mesh', process=True)
    t.merge_vertices()
    comps = t.split(only_watertight=False)
    if len(comps) > 1:
        t = max(comps, key=lambda c: len(c.faces))
        t.remove_unreferenced_vertices()
    fx = pymeshfix.MeshFix(t.vertices.astype(np.float64), t.faces.astype(np.int32))
    fx.repair()
    r = trimesh.Trimesh(np.asarray(fx.points), np.asarray(fx.faces), process=True)
    rc = r.split(only_watertight=False)
    if len(rc) > 1:
        r = max(rc, key=lambda c: len(c.faces))
        r.remove_unreferenced_vertices()
    if r.is_watertight and r.volume < 0:
        r.invert()
    Path(tmp).unlink(missing_ok=True)
    # Laplacian smoothing: relaxes the remesh staircase, preserves features (config-driven).
    if args.laplacian_iters > 0:
        r = trimesh.smoothing.filter_laplacian(r, lamb=args.laplacian_lamb, iterations=args.laplacian_iters)
    r.export(args.out)
    print(f'[surface_remesh_pre] V={len(r.vertices):,} F={len(r.faces):,} '
          f'watertight={r.is_watertight} vol={abs(r.volume)*1e9:,.0f}mm3 -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
