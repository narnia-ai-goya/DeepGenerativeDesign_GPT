"""FEA-ready mesh prep + tet FEA for the generated part surface.

The pipeline calls this on `final.obj` — the remeshed/smoothed surface it delivers. It also
accepts the raw boolean `hybrid.obj` (manifold3d; 46k-844k faces depending on domain and style),
which on the chunky domains is watertight but too dense / seam-heavy for gmsh HXT to tet
directly, and pymeshfix on that raw dense mesh discards whole peg regions. Either way this
two-step prep produces a clean, coarse, watertight manifold that gmsh tets cleanly AND keeps
the fix/load peg regions so BC facet detection works:

  1. pymeshlab isotropic explicit remesh → uniform edge (default 2.5mm)
     + merge close verts + repair non-manifold + close holes.
  2. pymeshfix on the (now coarse, clean) mesh → guaranteed watertight,
     geometry preserved (bbox intact) since input is already clean.
  3. hand off to fea_tet_from_mesh.py --no-repair (gmsh HXT + dolfinx).

Usage:
  python fea_prep_and_run.py --config configs/<domain>.json --in <run>/final.obj --out-dir <run>/fea \
      [--target-edge-mm 2.5] [--tet-size 0.005]

Requires FENICS_PY to point at the fenics env python: the tet + solve step runs as a
subprocess (pymeshlab and gmsh/dolfinx cannot share one interpreter, see prep_fea_mesh).
"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import trimesh

FENICS_PY = os.environ.get('FENICS_PY', 'python')
HERE = Path(__file__).resolve().parent   # this package's code/ directory (sibling scripts live here)


def _prep_once(in_mesh, out_stl, target_edge_mm):
    # NB: keep pymeshlab function-local, and hand the tet+solve step to fea_tet_from_mesh.py as a
    # SUBPROCESS (below). In the fenics env pymeshlab loads a system libstdc++ that lacks the
    # CXXABI_1.3.15 / GLIBCXX_3.4.31 symbols conda's gmsh and dolfinx need, so importing pymeshlab
    # first in the same process makes `import gmsh` / `import dolfinx` fail. Process separation,
    # not import order, is what keeps this robust.
    import pymeshlab, pymeshfix
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(in_mesh))

    def _try_filter(name, **kw):
        try: getattr(ms, name)(**kw); return True
        except Exception: return False

    # ── (1) PRE-remesh cleanup — tidy the raw hybrid surface so remesh has clean input ──
    ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(0.001))
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices() if _try_filter('meshing_remove_duplicate_vertices') else None
    _try_filter('meshing_remove_null_faces')                     # zero-area faces
    _try_filter('meshing_remove_folded_faces')                   # flipped/folded slivers
    _try_filter('meshing_remove_t_vertices')                     # T-junction slivers
    _try_filter('meshing_remove_unreferenced_vertices')
    _try_filter('meshing_repair_non_manifold_edges')
    _try_filter('meshing_repair_non_manifold_vertices')
    # drop tiny disconnected components (floaters) before remesh
    _try_filter('meshing_remove_connected_component_by_diameter',
                mincomponentdiag=pymeshlab.PercentageValue(5.0))

    # ── (2) surface isotropic remesh (uniform, fine) ──
    ms.meshing_isotropic_explicit_remeshing(
        iterations=8, targetlen=pymeshlab.PureValue(target_edge_mm / 1000.0), adaptive=False)

    # ── (3) POST-remesh cleanup → watertight surface for tet ──
    _try_filter('meshing_remove_null_faces')
    _try_filter('meshing_remove_folded_faces')
    _try_filter('meshing_repair_non_manifold_edges')
    _try_filter('meshing_repair_non_manifold_vertices')
    _try_filter('meshing_close_holes', maxholesize=1000)
    tmp = str(Path(out_stl).with_suffix('.remesh.stl'))
    ms.save_current_mesh(tmp)
    t = trimesh.load(tmp, force='mesh', process=False); t.merge_vertices()
    comps = t.split(only_watertight=False)
    if len(comps) > 1:
        t = max(comps, key=lambda c: len(c.faces))
    mf = pymeshfix.MeshFix(t.vertices.astype(np.float64), t.faces.astype(np.int32))
    mf.repair()
    r = trimesh.Trimesh(vertices=np.asarray(mf.points), faces=np.asarray(mf.faces), process=True)
    r.remove_unreferenced_vertices()
    # keep largest component (pymeshfix may leave a few shells)
    rc = r.split(only_watertight=False)
    if len(rc) > 1:
        r = max(rc, key=lambda c: len(c.faces)); r.remove_unreferenced_vertices()
    if r.is_watertight and r.volume < 0:
        r.invert()
    try: os.remove(tmp)
    except Exception: pass
    return r


def prep_fea_mesh(in_mesh, out_stl, target_edge_mm=2.5):
    """Remesh + pymeshfix at a SINGLE edge length → clean manifold, export STL.
    Edge escalation for gmsh-robustness is driven by the caller (_try)."""
    r = _prep_once(in_mesh, out_stl, target_edge_mm)
    ext = (r.bounds[1] - r.bounds[0]) * 1000
    print(f'  prep edge={target_edge_mm}mm: V={len(r.vertices):,} F={len(r.faces):,} '
          f'watertight={r.is_watertight} vol={abs(r.volume)*1e9:,.0f}mm³ bbox={ext.round(1)}', flush=True)
    r.export(str(out_stl))
    return r


def main():
    # paper ref: Supplementary, Evaluation Protocol Details, a posteriori verification
    #            (linear elasticity, E = 110 GPa, nu = 0.3, 1000 N load, fixture clamped;
    #            Gmsh tet + DOLFINx).
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None, help='run config JSON; stages.fea fills fix_stl / '
                    'load_stl / tet_size / force_N / force_dir / E_GPa / nu / target_edge_mm')
    ap.add_argument('--in', dest='inp', required=True,
                    help='surface to analyse — the pipeline passes <run>/final.obj')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--fix-stl', default='data_real/bracket/fixed_remesh.stl')
    ap.add_argument('--load-stl', default='data_real/bracket/load_remesh.stl')
    ap.add_argument('--target-edge-mm', type=float, default=2.5)
    ap.add_argument('--tet-size', type=float, default=0.005)
    ap.add_argument('--force-N', type=float, default=1000.0)
    ap.add_argument('--force-dir', default='0,0,-1')
    ap.add_argument('--E-GPa', default='110')
    ap.add_argument('--nu', default='0.3')
    ap.add_argument('--prep-mode', default='remesh', choices=['remesh', 'direct'],
                    help="'remesh' (default): isotropic re-remesh + edge escalation, robust for "
                         "chunky parts (bracket/mm). 'direct': feed the polished mesh straight to "
                         "fea_tet_from_mesh's own decimate+repair — required for thin lattices (link), "
                         "where an isotropic re-remesh collapses slender members.")
    ap.add_argument('--sane-c', type=float, default=1.0,
                    help='compliance (J) upper bound; a solve above it is treated as a degenerate '
                         'tet mesh, not a soft part. Per-domain via stages.fea.sane_c, because '
                         'compliance scales with the part size and the load case')
    ap.add_argument('--prep-only', action='store_true', help='stop after producing the clean STL')
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_config import apply_config
    apply_config(ap, stage='fea')           # stages.fea → argparse defaults (paths / material / load / tet)
    args = ap.parse_args()

    # Fail fast and loudly on an unset/wrong FENICS_PY. A which() check is not enough — plain
    # 'python' usually resolves to some interpreter on PATH that simply has no gmsh/dolfinx. Left
    # unchecked, every tet attempt spawns it, each dies on import, and the escalation ladder burns
    # minutes of remeshing only to report "no sane summary" — which reads as a mesh problem when it
    # is a config problem. One ~2s import probe up front is worth that.
    if not args.prep_only:
        if not shutil.which(FENICS_PY):
            sys.exit(f'FENICS_PY={FENICS_PY!r} is not an executable — '
                     'export FENICS_PY=<fenics env>/bin/python')
        probe = subprocess.run([FENICS_PY, '-c', 'import gmsh, dolfinx'], capture_output=True, text=True)
        if probe.returncode != 0:
            err = (probe.stderr or '').strip().split('\n')
            sys.exit(f'FENICS_PY={FENICS_PY!r} cannot import gmsh/dolfinx ({err[-1] if err else "?"}) — '
                     'export FENICS_PY=<fenics env>/bin/python')

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    clean_stl = out / 'fea_input_clean.stl'
    summ = out / 'fea_tet_summary.json'
    SANE_C = args.sane_c   # compliance (J) upper bound — above this = degenerate tet mesh (garbage)

    # 'direct' prep mode (thin lattices, e.g. link): skip the pymeshlab isotropic re-remesh — which
    # collapses slender members and makes gmsh HXT abort — and feed the polished mesh straight to
    # fea_tet_from_mesh's own decimate+Taubin+pymeshfix repair, at a FINE tet size matched to the
    # surface (link tet_size 0.0015 m), so slender members survive the a-posteriori verification.
    if args.prep_mode == 'direct':
        print('[fea] direct fea_tet (internal repair, no isotropic re-remesh)', flush=True)
        for algo, name in ((10, 'HXT'), (1, 'Delaunay')):
            if summ.exists():
                summ.unlink()
            cmd = [FENICS_PY, str(HERE / 'fea_tet_from_mesh.py'), '--mesh', str(args.inp),
                   '--fix-stl', args.fix_stl, '--load-stl', args.load_stl,
                   '--mesh-size', str(args.tet_size), '--tet-algo', str(algo),
                   '--force-N', str(args.force_N), '--force-dir', args.force_dir,
                   '--E-GPa', str(args.E_GPa), '--nu', str(args.nu), '--out', str(out)]
            try:
                r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            except Exception as e:
                print(f'  [direct {name}] aborted ({type(e).__name__}) -> next', flush=True)
                continue
            for ln in r2.stdout.strip().split('\n'):
                if any(k in ln for k in ('tet mesh:', 'fix facets', 'load facets',
                                         'compliance', 'u_max', 'ERROR', 'gmsh failed')):
                    print(f'  [direct {name}] ' + ln.strip(), flush=True)
            if r2.returncode != 0 and not summ.exists():
                err = (r2.stderr or '').strip().split('\n')
                print(f'  [direct {name}] child rc={r2.returncode}: ' + (err[-1] if err else '(no stderr)'),
                      flush=True)
            if summ.exists():
                c = json.load(open(summ)).get('compliance', 1e9)
                if 0 < c < SANE_C:
                    print(f'[done] {summ}  (direct {name})', flush=True)
                    return
                print(f'  [direct {name}] compliance out of sane range → discard', flush=True)
                summ.unlink()
        print('[FAIL] direct fea_tet: no sane summary', flush=True)
        return

    print('[prep] remesh + pymeshfix → clean watertight', flush=True)
    r = prep_fea_mesh(args.inp, clean_stl, target_edge_mm=args.target_edge_mm)
    if not r.is_watertight:
        print('  WARN: clean mesh not watertight — gmsh may fail', flush=True)
    if args.prep_only:
        return

    print('[fea] gmsh tet + dolfinx (fenics env)', flush=True)

    def _try(edge_mm, algo, name):
        # (re)build clean mesh at this edge, then tet+FEA with this algo (mesh-size = tet_size fixed)
        prep_fea_mesh(args.inp, clean_stl, target_edge_mm=edge_mm)
        if summ.exists():
            summ.unlink()
        cmd = [FENICS_PY, str(HERE / 'fea_tet_from_mesh.py'),
               '--mesh', str(clean_stl),
               '--fix-stl', args.fix_stl, '--load-stl', args.load_stl,
               '--mesh-size', str(args.tet_size), '--tet-algo', str(algo),
               '--force-N', str(args.force_N), '--force-dir', args.force_dir,
               '--E-GPa', str(args.E_GPa), '--nu', str(args.nu),
               '--no-repair', '--out', str(out)]
        try:
            r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except Exception as e:
            # a hanging / crashing gmsh attempt must not abort the whole escalation
            print(f'  [{name} e={edge_mm}] tet attempt aborted ({type(e).__name__}) -> next', flush=True)
            if summ.exists():
                summ.unlink()
            return False
        for ln in r2.stdout.strip().split('\n'):
            if any(k in ln for k in ('tet mesh:', 'fix facets', 'load facets',
                                     'compliance', 'u_max', 'ERROR', 'gmsh failed')):
                print(f'  [{name} e={edge_mm}] ' + ln.strip(), flush=True)
        if r2.returncode != 0 and not summ.exists():
            # Surface the child's stderr: without this a bad FENICS_PY (child dies on import) looks
            # exactly like a gmsh failure, and the escalation reports "no sane summary" with no cause.
            err = (r2.stderr or '').strip().split('\n')
            print(f'  [{name} e={edge_mm}] child rc={r2.returncode}: ' + (err[-1] if err else '(no stderr)'),
                  flush=True)
        if summ.exists():
            c = json.load(open(summ)).get('compliance', 1e9)
            if 0 < c < SANE_C:
                return True
            print(f'  [{name} e={edge_mm}] compliance={c:.3e} out of sane range → discard', flush=True)
            summ.unlink()
        return False

    # HXT is fast and works for almost all meshes → try it FIRST at the base edge.
    # Only if HXT fails (occasional 'double free' on sliver-heavy meshes) escalate:
    # coarser remesh edges + Delaunay fallback.
    done = False
    attempts = [(args.target_edge_mm, 10, 'HXT'),
                (args.target_edge_mm, 1,  'Delaunay'),
                (3.0, 10, 'HXT'), (3.0, 1, 'Delaunay'),
                (3.5, 1, 'Delaunay'), (4.0, 1, 'Delaunay')]
    for edge_mm, algo, name in attempts:
        if _try(edge_mm, algo, name):
            print(f'[done] {summ}  ({name}, edge={edge_mm}mm)', flush=True)
            done = True
            break

    # Fallback for boxy/chunky parts: the isotropic remesh can leave coplanar (dihedral-0)
    # facets that gmsh rejects. Feed the raw polished mesh through fea_tet_from_mesh's own
    # decimate + Taubin + pymeshfix repair path instead (robust on chunky geometry; a thin
    # lattice would lose members here, but those succeed on the remesh path above).
    def _try_repair(algo, name):
        if summ.exists():
            summ.unlink()
        cmd = [FENICS_PY, str(HERE / 'fea_tet_from_mesh.py'), '--mesh', str(args.inp),
               '--fix-stl', args.fix_stl, '--load-stl', args.load_stl,
               '--mesh-size', str(args.tet_size), '--tet-algo', str(algo),
               '--force-N', str(args.force_N), '--force-dir', args.force_dir,
               '--E-GPa', str(args.E_GPa), '--nu', str(args.nu), '--out', str(out)]
        try:
            r2 = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except Exception as e:
            print(f'  [{name}] repair attempt aborted ({type(e).__name__}) -> next', flush=True)
            if summ.exists():
                summ.unlink()
            return False
        for ln in r2.stdout.strip().split('\n'):
            if any(k in ln for k in ('tet mesh:', 'compliance', 'u_max', 'gmsh failed', 'ERROR')):
                print(f'  [{name}] ' + ln.strip(), flush=True)
        if summ.exists():
            c = json.load(open(summ)).get('compliance', 1e9)
            if 0 < c < SANE_C:
                return True
            summ.unlink()
        return False

    if not done:
        for algo, name in [(10, 'repair+HXT'), (1, 'repair+Delaunay')]:
            if _try_repair(algo, name):
                print(f'[done] {summ}  ({name}, decimate+repair fallback)', flush=True)
                done = True
                break
    if not done:
        print('[FAIL] no sane summary after remesh escalation + repair fallback', flush=True)


if __name__ == '__main__':
    main()
