#!/usr/bin/env python
"""Stage 0 — input preparation: the three raw domain STLs -> every derived input the pipeline reads.

A domain is fully defined by three raw STLs, shipped under `data_real/<domain>/`:

    original_DesignSpace.stl   design domain — the envelope the generated part must stay inside
    fixed.stl                  fixture surface (clamped in FEA)
    load.stl                   load surface

Everything else a run consumes is derived from them, into the same domain directory:

  1. remesh : remesh_domain_bc.py        -> original_DesignSpace_remesh.stl, fixed_remesh.stl,
              load_remesh.stl — welded (raw CAD STLs are unwelded triangle soup, V = 3F) and
              isotropically remeshed to near-equilateral triangles.
  2. voxel  : make_bc_proper_pysdf.py    -> voxel.npz — the 64³ design-domain / boundary-condition
              grid, read by generation as both `bc_proper` and `bracket_occ`.
  3. fea_domain : fea_domain/{original_DesignSpace,fixed,load}.stl — the in-loop FEA's domain
              directory. The FEM solver (fenics_fea_bracket.py) resolves these three fixed
              filenames inside it, so this step stages geometry under those names, picked per
              role (see FEA_DOMAIN_SRC): the design domain **raw**, the fixture/load pegs
              **remeshed**.
  4. fea_mesh : fea_shared.msh — the in-loop FEM tet mesh, built once from fea_domain/ at
              stages.mesh.fea_mesh_size. The design domain is the same for every prompt, so every
              run shares this one file (stages.mesh.fea_mesh_cache) instead of tetrahedralizing
              the domain again per style/variant.

Which mesh a later stage consumes: post-processing (`stages.post`), the a-posteriori FEA
boundary surfaces (`stages.fea`) and the in-loop FEA's peg surfaces all read `*_remesh.stl` —
raw CAD pegs are so sparsely tessellated that boundary nodes on a large flat face fall outside
the solver's 6 mm vertex-proximity test (measured: the link misses ~29% of its peg surface that
way). Two references stay on the **raw** design-domain STL, each for a measured reason:

  - the voxel grid frame — `voxel.npz` (step 2) and `stages.mesh.fea_bracket_stl`. The 64³
    grid's pitch and origin come from that STL's bounding box, and remeshing rounds off sharp
    outline corners (no bbox change for bracket / motor_mount, up to 1.47 mm for the link),
    which would shift the whole generation frame.
  - the in-loop FEM's tetrahedralized domain — `fea_domain/original_DesignSpace.stl`. Gmsh
    meshes every raw CAD tessellation cleanly but rejects the bracket's remeshed domain
    outright ("Invalid boundary mesh (overlapping facets)"): the isotropic remesh can
    self-intersect.

Config-driven like every other stage: the per-domain preparation values live in
`configs/<domain>.json` under `stages.prep` (remesh edge lengths; grid resolution, bbox margin
and the fixture/load dilation for the voxel grid), and this orchestrator passes only `--config`
plus the per-run I/O paths. The configs' data paths point at the prepared domain directory, so
nothing downstream needs to know how these files were made.

Deterministic, domain-only (no style, no prompt, no GPU, no network) — run it once per domain.
Each step is resumable: skipped when its output exists, re-run with `--force`.

Usage:
    python run_prep.py --domain all                          # all steps, all three domains
    python run_prep.py --domain bracket --stage voxel --force
    python run_prep.py --domain all --data-dir data_real     # (default) domain root to prepare
    python run_prep.py --domain all --verify                 # report what is on disk, build nothing
"""
import argparse, json, os, sys, time

from run_from_image import DOMAINS, PY_D3D, PY_FEN, ROOT, DATA_ROOT, EXP, CODE, run

DATA_DIR = 'data_real'                                   # default domain root (raw STLs live here)
RAW = ('original_DesignSpace', 'fixed', 'load')           # the three raw STLs defining a domain
FEA_DOMAIN = 'fea_domain'                                 # in-loop FEA domain dir (staged remesh)


def domain_dir(dom, data_dir):
    """The domain directory being prepared: <data_dir>/<domain>/ (holds raw + derived inputs)."""
    return DATA_ROOT / data_dir / dom


def raw_missing(dd):
    """Raw STLs absent from the domain dir — prep cannot run without all three."""
    return [f'{n}.stl' for n in RAW if not (dd / f'{n}.stl').exists()]


def _log_dir(dom):
    d = EXP / f'{dom}/prep'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _mesh_summary(paths):
    """F counts for the meshes just written (best effort — trimesh is a generation-env dep)."""
    try:
        import trimesh
    except Exception:
        return ''
    out = []
    for p in paths:
        try:
            m = trimesh.load(str(p), force='mesh', process=False)
            out.append(f'{p.name} F={len(m.faces):,}')
        except Exception:
            out.append(f'{p.name} ?')
    return '  '.join(out)


def stage_remesh(dom, c, dd, force):
    """Step 1: weld + isotropic remesh the three raw STLs -> *_remesh.stl (stages.prep edges)."""
    outs = [dd / f'{n}_remesh.stl' for n in RAW]
    if all(o.exists() for o in outs) and not force:
        print(f'[{dom}/remesh] SKIP (*_remesh.stl exist)'); return True
    rc = run([PY_D3D, str(CODE / 'remesh_domain_bc.py'), '--config', str(ROOT / c['config']),
              '--domain-dir', str(dd)], _log_dir(dom) / 'remesh.log', cwd=str(DATA_ROOT))
    ok = all(o.exists() for o in outs)
    print(f'[{dom}/remesh] {"OK" if ok else "FAIL(rc=%d)" % rc}  {_mesh_summary(outs) if ok else _log_dir(dom) / "remesh.log"}')
    return ok


def stage_voxel(dom, c, dd, force):
    """Step 2: rasterize the RAW STLs to the 64³ design-domain / BC grid -> voxel.npz.

    Raw, not remeshed, on purpose: this grid's pitch and origin define the generation frame
    (see the module docstring). Grid resolution / margin / dilations come from stages.prep."""
    out = dd / 'voxel.npz'
    if out.exists() and not force:
        print(f'[{dom}/voxel] SKIP (voxel.npz exists)'); return True
    rc = run([PY_D3D, str(CODE / 'make_bc_proper_pysdf.py'), '--config', str(ROOT / c['config']),
              '--bracket', str(dd / 'original_DesignSpace.stl'),
              '--fix', str(dd / 'fixed.stl'), '--load', str(dd / 'load.stl'),
              '--out', str(out)], _log_dir(dom) / 'voxel.log', cwd=str(DATA_ROOT))
    if not out.exists():
        print(f'[{dom}/voxel] FAIL(rc={rc}) — log: {_log_dir(dom) / "voxel.log"}'); return False
    try:
        import numpy as np
        z = np.load(out)
        n = {k: int(z[k].sum()) for k in ('bracket', 'bc', 'fix', 'load')}
        print(f'[{dom}/voxel] OK  R={z["bracket"].shape[0]}³  pitch={float(z["pitch"]) * 1000:.3f} mm  '
              f'domain={n["bracket"]:,}  bc={n["bc"]:,} (fix {n["fix"]:,} / load {n["load"]:,})')
    except Exception:
        print(f'[{dom}/voxel] OK -> {out}')
    return True


# What each staged file is used for decides whether it may be the remeshed version:
#   original_DesignSpace.stl : gmsh TETRAHEDRALIZES it (fenics_fea_bracket.build_mesh) and it
#       defines the voxel frame → RAW. The isotropic remesh can introduce self-intersecting
#       facets: gmsh refuses the bracket's remeshed domain outright ("Invalid boundary mesh
#       (overlapping facets)"), while every raw CAD tessellation meshes cleanly.
#   fixed.stl / load.stl : only ever read as a VERTEX CLOUD, to pick boundary nodes by
#       proximity (cKDTree over mesh.vertices) → remeshed, like every other consumed mesh.
FEA_DOMAIN_SRC = {'original_DesignSpace': 'original_DesignSpace',   # raw
                  'fixed': 'fixed_remesh', 'load': 'load_remesh'}


def stage_fea_domain(dom, c, dd, force):
    """Step 3: stage the in-loop FEA domain dir under the three filenames the solver expects.

    fenics_fea_bracket.py resolves original_DesignSpace.stl / fixed.stl / load.stl inside its
    --domain-dir, and the generator only copies into that dir when a file is missing — so
    pre-staging here is what decides which geometry the in-loop FEM actually sees. See
    FEA_DOMAIN_SRC above for why the design domain is the raw STL and the pegs are remeshed."""
    import shutil
    fd = dd / FEA_DOMAIN
    outs = [fd / f'{n}.stl' for n in RAW]
    if all(o.exists() for o in outs) and not force:
        print(f'[{dom}/fea_domain] SKIP ({FEA_DOMAIN}/ staged)'); return True
    srcs = [dd / f'{FEA_DOMAIN_SRC[n]}.stl' for n in RAW]
    missing = [s.name for s in srcs if not s.exists()]
    if missing:
        print(f'[{dom}/fea_domain] FAIL: {missing} missing (run --stage remesh first)'); return False
    fd.mkdir(parents=True, exist_ok=True)
    for s, o in zip(srcs, outs):
        shutil.copyfile(s, o)
    print(f'[{dom}/fea_domain] OK  ' + ', '.join(f'{o.name} <- {s.name}' for s, o in zip(srcs, outs)))
    return True


def stage_fea_mesh(dom, c, dd, force):
    """Step 4: build the in-loop FEM tet mesh once per domain -> fea_shared.msh.

    The in-loop FEA tetrahedralizes the *design domain*, which is identical for every prompt of a
    domain, so this is built here and shared: `stages.mesh.fea_mesh_cache` points every run at this
    one file and the solver just reports "reusing cached mesh". Built from fea_domain/ (step 3) at
    `stages.mesh.fea_mesh_size` — the same value the generator would pass, read from the config so
    the two cannot drift. Runs in the FEA environment (FENICS_PY), which is what reads it back."""
    cfg = json.load(open(ROOT / c['config']))
    mesh_stage = cfg.get('stages', {}).get('mesh', {})
    size = mesh_stage.get('fea_mesh_size')
    cache = DATA_ROOT / mesh_stage.get('fea_mesh_cache', f'{DATA_DIR}/{dom}/fea_shared.msh')
    if not size or size <= 0:
        print(f'[{dom}/fea_mesh] SKIP: stages.mesh.fea_mesh_size is unset/auto — the generator '
              f'derives it at run time, so there is no fixed mesh to pre-build'); return True
    if cache.exists() and not force:
        print(f'[{dom}/fea_mesh] SKIP ({cache.name} exists)'); return True
    fd = dd / FEA_DOMAIN
    if not (fd / 'original_DesignSpace.stl').exists():
        print(f'[{dom}/fea_mesh] FAIL: {FEA_DOMAIN}/ not staged (run --stage fea_domain first)'); return False
    if force and cache.exists():
        cache.unlink()
    rc = run([PY_FEN, str(CODE / 'fenics_fea_bracket.py'), '--build-mesh-only',
              '--domain-dir', str(fd), '--mesh-size', str(size), '--mesh-cache', str(cache)],
             _log_dir(dom) / 'fea_mesh.log', cwd=str(DATA_ROOT))
    ok = cache.exists()
    if ok:
        tail = [l for l in open(_log_dir(dom) / 'fea_mesh.log').read().splitlines() if l.startswith('[mesh]')]
        print(f'[{dom}/fea_mesh] OK  {cache.relative_to(DATA_ROOT)}  {tail[-1] if tail else ""}')
    else:
        print(f'[{dom}/fea_mesh] FAIL(rc={rc}) — log: {_log_dir(dom) / "fea_mesh.log"}')
    return ok


STAGES = {'remesh': stage_remesh, 'voxel': stage_voxel,
          'fea_domain': stage_fea_domain, 'fea_mesh': stage_fea_mesh}


def report(dom, dd):
    """One line per expected input: what exists on disk for this domain."""
    items = [f'{n}.stl' for n in RAW] + [f'{n}_remesh.stl' for n in RAW] + \
            ['voxel.npz'] + [f'{FEA_DOMAIN}/{n}.stl' for n in RAW] + ['fea_shared.msh']
    print(f'  {dom}:')
    for it in items:
        p = dd / it
        tag = 'raw  ' if it in [f'{n}.stl' for n in RAW] else 'built'
        print(f'    [{"x" if p.exists() else " "}] {tag}  {it}'
              f'{"" if not p.exists() else f"  ({p.stat().st_size:,} B)"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain', default='all', help='bracket|motor_mount|link|all (comma-separated allowed)')
    ap.add_argument('--stage', default='all', help='remesh|voxel|fea_domain|all (comma-separated allowed)')
    ap.add_argument('--data-dir', default=DATA_DIR,
                    help=f'domain root holding <domain>/*.stl, relative to DATA_ROOT (default: {DATA_DIR})')
    ap.add_argument('--force', action='store_true', help='rebuild even if the output exists')
    ap.add_argument('--verify', action='store_true', help='only report which inputs exist, build nothing')
    args = ap.parse_args()
    doms = list(DOMAINS) if args.domain == 'all' else args.domain.split(',')
    stages = list(STAGES) if args.stage == 'all' else args.stage.split(',')
    unknown = [d for d in doms if d not in DOMAINS] + [s for s in stages if s not in STAGES]
    if unknown:
        sys.exit(f'unknown domain/stage: {unknown}  (domains {list(DOMAINS)}, stages {list(STAGES)})')
    os.chdir(DATA_ROOT)

    if args.verify:
        print(f'== run_prep --verify: {args.data_dir}/ ==')
        for dom in doms:
            report(dom, domain_dir(dom, args.data_dir))
        return

    print(f'== run_prep: domains={doms} stages={stages} data_dir={args.data_dir} force={args.force} ==')
    results = {}
    for dom in doms:
        dd = domain_dir(dom, args.data_dir)
        print(f'\n########## {dom}  ({args.data_dir}/{dom}) ##########')
        miss = raw_missing(dd)
        if miss:
            print(f'  [{dom}] FAIL: raw STL missing in {dd}: {miss} → skipped')
            results[(dom, 'raw')] = False
            continue
        for st in stages:
            t0 = time.time()
            ok = STAGES[st](dom, DOMAINS[dom], dd, args.force)
            results[(dom, st)] = ok
            print(f'    ({time.time() - t0:.0f}s)')
            if not ok:
                print(f'  [{dom}] step {st} failed → stopping remaining steps'); break
    print('\n== summary ==')
    for (dom, st), ok in results.items():
        print(f'  {dom:12} {st:11} {"OK" if ok else "FAIL"}')
    if all(results.values()):
        print('\nInputs ready. The configs point at this domain root, so a run needs nothing else:\n'
              '  python run_from_image.py --domain all --grid        # conditioning image -> mesh -> post -> FEA\n'
              '  python run_conditioning.py --domain bracket --style de_gyroid   # or rebuild conditioning first')


if __name__ == '__main__':
    main()
