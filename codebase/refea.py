#!/usr/bin/env python
"""Re-run the a-posteriori FEA **serially** on meshes already on disk.

Why this exists
---------------
The FEA stage tetrahedralizes each polished mesh with Gmsh. Gmsh's fast HXT algorithm can
crash ("double free" / PLC fault) when several tetrahedralizations run at the same time, so a
*parallel* run (`run_from_image.py --domain all`, or several workers at once) occasionally
leaves a few styles with a failed or missing compliance. The FEA stage already escalates
(HXT -> Delaunay -> coarser remesh -> decimate+repair fallback) and usually recovers, but the
only way to guarantee no concurrency crash is to run one tetrahedralization at a time.

This script does exactly that: it walks the experiment outputs, and for every variant whose
mesh (`final.obj`) exists but whose FEA compliance is missing / out of range, it re-runs the
**same config-driven FEA** (`fea_prep_and_run.py --config ...`) **serially**. Generation and
post-processing meshes are never touched — only the (cheap) FEA is redone.

Usage
-----
    # honours EXP_ROOT exactly like run_from_image.py / run_ablation.py
    python refea.py --domain all                       # fill any missing/failed FEA, all variants
    python refea.py --domain link --variant on,posthoc # a subset
    python refea.py --domain all --force               # recompute every FEA from scratch
"""
import argparse
import os

from run_from_image import (DOMAINS, ROOT, DATA_ROOT, PY_FEN, CODE, FEA_THREADS, compliance_of, sane_c,
                            styles_of, variant_dir, run)


def fea_one(dom, c, vdir, force):
    """Run (or re-run) the config-driven FEA for one variant dir. Returns (status, C_mJ|None)."""
    mesh = vdir / 'final.obj'   # same input as run_from_image.stage_fea — the delivered surface
    feadir = vdir / 'fea'
    if not mesh.exists():
        return ('no-mesh', None)
    if not force:
        cj = compliance_of(vdir, sane_c(c))
        if cj is not None:
            return ('skip', cj * 1e3)
    feadir.mkdir(parents=True, exist_ok=True)
    cmd = [PY_FEN, str(CODE / 'fea_prep_and_run.py'), '--config', str(ROOT / c['config']),
           '--in', str(mesh), '--out-dir', str(feadir)]
    run(cmd, feadir / 'refea.log', env=dict(os.environ, OMP_NUM_THREADS=FEA_THREADS), cwd=str(DATA_ROOT))
    cj = compliance_of(vdir, sane_c(c))
    return ('ok', cj * 1e3) if cj is not None else ('FAIL', None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain', default='all', help='bracket|motor_mount|link|all')
    ap.add_argument('--variant', default='on,off,posthoc', help='comma list from on,off,posthoc')
    ap.add_argument('--force', action='store_true', help='recompute even when a sane compliance exists')
    args = ap.parse_args()

    doms = list(DOMAINS) if args.domain == 'all' else args.domain.split(',')
    variants = args.variant.split(',')
    os.chdir(DATA_ROOT)

    done = fixed = failed = 0
    for dom in doms:
        c = DOMAINS[dom]
        for style in styles_of(dom):
            for v in variants:
                status, cj = fea_one(dom, c, variant_dir(dom, style, v), args.force)
                if status == 'no-mesh':
                    continue
                tag = f'C={cj:.3f} mJ' if cj is not None else '(no result)'
                print(f'  {dom}/{style}/{v}: {status}  {tag}', flush=True)
                done += 1
                if status == 'ok':
                    fixed += 1
                elif status == 'FAIL':
                    failed += 1
    print(f'\n== refea: {done} variants processed, {fixed} (re)computed, {failed} still failing ==')


if __name__ == '__main__':
    main()
