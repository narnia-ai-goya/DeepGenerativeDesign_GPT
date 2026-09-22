#!/usr/bin/env python
"""Image-first reproduction: shipped conditioning image -> mesh -> post -> a-posteriori FEA.

This is the default, lightweight flow. It starts from the multi-view conditioning images
already bundled under data/<domain>/conditioning/<style>/, so it needs NO text-to-image stage
(no SDXL / diffusers / pyvista, no Hugging Face downloads):
    conditioning image  ->  D3D-S2 generation(GPU)  ->  post-processing(post)  ->  a-posteriori FEA(CPU/FEniCS)

To reproduce the conditioning images themselves from their text prompts (the full text-first
chain), use `run_from_text.py` instead. This module is also the shared stage engine that
run_from_text.py and run_ablation.py import.

Each stage is resumable (skipped if output exists, re-run with --force). Generation requires the
Direct3D-S2 backbone (external/Direct3D-S2) and a GPU, and writes mesh.obj into experiments/<domain>/;
post and fea then run on that mesh (post/fea need no GPU once mesh.obj exists). The package ships the
inputs only (domain geometry in data_real/<domain>/, conditioning images in data/<domain>/conditioning/),
so a fresh checkout runs from the bundled conditioning image.

Environments (handled automatically):
  - generation/post : conda env `direct3ds2` (torch). FEA_NORMALIZE=0, seed 42 deterministic.
  - FEA         : conda env `fenics` (dolfinx 0.9). CPU.
  - on NVML driver mismatch, auto-detect libnvidia-ml LD_PRELOAD.

Each domain ships five representative styles (data/<domain>/conditioning/<style>/), so
--domain all covers 3 domains x 5 styles. --domain/--style/--stage all accept 'all' or a
comma-separated list.

Usage:
    python run_from_image.py --domain all                                  # 3 domains x 5 styles, full chain
    python run_from_image.py --domain bracket --style de_crackle --stage all --force
    python run_from_image.py --domain link --stage post,fea                # post+fea (after gen has produced mesh.obj)
"""
import argparse, glob, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent                 # this package (codebase/)
CODE = ROOT / 'code'
# Domain geometry (data_real/) and conditioning images (data/) live NEXT to the package, not inside
# it, so the code tree stays independent of the bundled data. DATA_ROOT overrides that location. It
# is also the cwd every stage runs in, which is what the configs' relative data_real/ paths resolve
# against — keep the two the same or the configs will not find their STLs.
DATA_ROOT = Path(os.environ['DATA_ROOT']).resolve() if os.environ.get('DATA_ROOT') else ROOT.parent
# Direct3D-S2 backbone location: defaults to external/Direct3D-S2 in the parent directory, overridable via env var.
REPO = Path(os.environ['D3DS2_ROOT']).resolve() if os.environ.get('D3DS2_ROOT') else ROOT.parent
# Experiment-output root (generation / post / FEA / grid outputs). Defaults to experiments/ inside the
# package; set EXP_ROOT to redirect all outputs elsewhere (e.g. a sibling dir) and keep the package clean.
EXP = Path(os.environ['EXP_ROOT']).resolve() if os.environ.get('EXP_ROOT') else DATA_ROOT / 'experiments'
# generation/post = current interpreter (assumed run in the direct3ds2 env), FEA = separate fenics env.
# each environment's Python is set via the D3DS2_PY / FENICS_PY env vars (see README).
PY_D3D  = os.environ.get('D3DS2_PY',  sys.executable)
PY_FEN  = os.environ.get('FENICS_PY', 'python')
# CPU threads for the FEM solves (in-loop and a-posteriori). Tuned for the reported 4090 + EPYC
# box, where the GPU stage leaves most cores idle; override per machine with FEA_THREADS.
FEA_THREADS = os.environ.get('FEA_THREADS', '9')
# Compliance (J) upper bound for the 1000 N verification load. A value above this means the tet
# mesh degenerated rather than the part being that soft, so it is reported as a failure.
# Fallback only; the per-domain value is stages.fea.sane_c. The bound has to be per-domain because
# compliance scales with the part and the load case: a self-equilibrating spread load on the caliper
# lands two orders of magnitude below the bracket's single-vector load, so one global 1 J ceiling is
# either too loose there or too tight elsewhere.
SANE_C_DEFAULT = 1.0


def sane_c(c):
    """Compliance (J) upper bound for this domain — above it the tet mesh degenerated rather than
    the part being that soft, so the run is reported as a failure."""
    return float(json.load(open(ROOT / c['config'])).get('stages', {})
                 .get('fea', {}).get('sane_c', SANE_C_DEFAULT))

# ── common generation flags ──────────────────────────────────────────────────────────────────
# All per-domain generation settings — FEM weights, geometric-guidance flags, r_min, load mode,
# data paths, seed and determinism — live in the per-domain config (configs/<domain>.json:
# top-level seed + stages.mesh) and are applied by the generation script via apply_config(stage='mesh').
# The orchestrator injects NOTHING on the CLI except --config and per-run I/O paths.

# ── per-domain settings (config path + validated post/FEA recipe) ─────────────────────────────
# Every per-domain setting now lives in that domain's config JSON (stages.mesh / post / fea and
# top-level seed). The orchestrator only needs to know which config each domain maps to.
DOMAINS = {
  'bracket':     {'config': 'configs/bracket.json'},
  'motor_mount': {'config': 'configs/motor_mount.json'},
  'link':        {'config': 'configs/link.json'},
  'caliper':     {'config': 'configs/caliper.json'},
  # same caliper, but every input mesh comes straight off the 256^3 SDF (marching cubes at level 0,
  # no binarisation, no gaussian, no Taubin, no remesh — edge length 0.773 mm = the grid pitch), and
  # the mounting lugs are subtracted from the design domain so it is nds - load - fix. The shipped
  # `caliper` inputs went through `<0` binarisation + gaussian 0.8 + Taubin 8 + a 4 mm remesh, which
  # left the generation stage clipping against a 4.06 mm envelope while post clipped against a
  # 1.55 mm one; here all three stages read the same file.
  'caliper_nofix': {'config': 'configs/caliper_nofix.json'},
  # dense-stage-only harness: skip_sparse, aug_lag off (drops the V <= vol_target inequality), FEA off.
  # Dense occupancy had been pinned inside 1.5 pp (44.5-46.0%) across five rmin settings with pairwise
  # IoU 0.83-0.93, because aug_lag holds the cap at mu=50 and every other term can only redistribute.
  # l_rmin is also not a "add material" term but closer to a surface-area penalty: filling the design
  # region 100% scores 356.5 while the 52.1%-occupied real result scores 274.8.
  'caliper_dense': {'config': 'configs/caliper_dense.json'},
  # Anisotropically pre-stretched caliper (x1.319, z1.507). The caliper is 89.6x185.0x68.7 mm, so an
  # isotropic 64^3 cube wastes two axes: the envelope is 5.62% of the grid, 30x60x24 voxels, median
  # local thickness 2.8 voxels, 54.2% under 4. A through-hole needs solid-void-solid = 3 voxels
  # across the wall, so over half the envelope cannot host one - which is why lattice patterns only
  # appeared at 35% fill (as windows punched clean through the wall) and vanished when filled, and
  # why ribs (~1 voxel of relief) and slots (4.73 mm ~ 1.5 voxels) never transferred at all
  # (surface relief 0.14-0.28 mm = noise) at dense OR sparse. Stretching to a cube would give 27.31%
  # but 4.86x the envelope voxels and sparse OOM, so t=0.3 lands at 10.47% - between bracket (7.73%)
  # and motor_mount (13.08%). --mode aniso alone is wrong: npz indices are used 1:1 against the
  # model's 64^3 output while the model generates in an isotropic normalised cube, so stretching the
  # grid alone squashes the shape. Stretching the STL keeps renders/npz/generation consistent; only
  # the output mesh needs the inverse scale.
  'caliper_stretch': {'config': 'configs/caliper_stretch.json'},
  # t=0.5 anisotropic stretch: 14.82% dense-grid occupancy (vs original 5.53%).
  'caliper_stretch50': {'config': 'configs/caliper_stretch50.json'},
  # 50-style sweep at the settings that produced dense_vs_sparse.png (tag ffff_):
  # stretched t=0.3 grid, skip_sparse=False, vol_target 0.65, out_w 15, bc_w 3, tw_rmin 0.075.
  'caliper_sp':   {'config': 'configs/caliper_sp.json'},
  'caliper_ffff': {'config': 'configs/caliper_ffff.json'},
}


def styles_of(dom):
    """Style keys to run for a domain = the conditioning subdirectories provided under
    data/<domain>/conditioning/<style>/. Ships five representative styles per domain; add more
    conditioning subdirs (e.g. via run_conditioning.py) and they are picked up automatically."""
    cdir = DATA_ROOT / f'data/{dom}/conditioning'
    if not cdir.exists():
        print(f'  [{dom}] no conditioning images under {cdir} — nothing to run for this domain '
              f'(generate them with run_conditioning.py, or check --domain)')
        return []
    return sorted(p.name for p in cdir.iterdir() if p.is_dir())


def wd_of(dom, style):
    """Per-(domain, style) working directory for generation + post + FEA outputs."""
    return EXP / f'{dom}/{style}/gen'


def abl_dir(dom, style, variant):
    """WHERE run_ablation.py writes a variant. All three variants get their own dir — including
    'on' — so the comparison runs every variant through an identical code path."""
    return EXP / f'{dom}/{style}/abl/{variant}'


def variant_dir(dom, style, variant):
    """WHERE to READ a variant's result back from (refea.py's recovery, render_results_grid.py).

    Prefer abl/<variant>/ when an ablation produced it; otherwise fall back to gen/ for 'on',
    since a plain run_from_image / run_from_text run writes its FEM-guided result there."""
    abl = abl_dir(dom, style, variant)
    if abl.exists():
        return abl
    return wd_of(dom, style) if variant == 'on' else abl


def dir_to_mode(force_dir):
    """Map an a-posteriori `force_dir` vector ("0,0,-1") to the in-loop LOAD_MODE keyword ("-z").
    Returns None for a direction that has no keyword (i.e. not axis-aligned)."""
    if str(force_dir).startswith('spread:'):
        return str(force_dir)          # opposed load: the keyword IS the mode, verbatim on both sides
    try:
        v = [float(x) for x in str(force_dir).split(',')]
    except (TypeError, ValueError):
        return None
    if len(v) != 3 or sum(abs(x) > 1e-9 for x in v) != 1:
        return None
    i = max(range(3), key=lambda k: abs(v[k]))
    return ('' if v[i] > 0 else '-') + 'xyz'[i]


def gen_env(config_path, base_env):
    """Export the in-loop FEM load direction (`stages.mesh.load_mode`) as LOAD_MODE for the FEA
    subprocess, and assert it agrees with the direction the a-posteriori FEA verifies
    (`stages.fea.force_dir`).

    The generation script also exports LOAD_MODE itself from the same config key, so a manual
    `--config` run behaves identically; this stays as the orchestrator-side guard. The two keys
    are separate on purpose (one is a keyword, the other a vector) — they silently drifted once,
    which left the in-loop objective optimizing a different load case than the verified one, so
    the mismatch is now a hard error rather than something to notice in a log."""
    st = json.load(open(config_path)).get('stages', {})
    lm = st.get('mesh', {}).get('load_mode')
    fd = st.get('fea', {}).get('force_dir')
    expect = dir_to_mode(fd)
    if lm and expect and lm != expect:
        sys.exit(f'config mismatch in {config_path}:\n'
                 f'  stages.mesh.load_mode = {lm!r}  (in-loop FEM guidance)\n'
                 f'  stages.fea.force_dir  = {fd!r} → {expect!r}  (a-posteriori verification)\n'
                 f'Set load_mode to {expect!r}, or state why the two load cases differ.')
    if not lm and expect:
        sys.exit(f'config {config_path}: stages.mesh.load_mode is unset, so the in-loop FEM would '
                 f'fall back to the solver default "diag" while verification loads {fd!r} '
                 f'({expect!r}). Set stages.mesh.load_mode = {expect!r}.')
    return dict(base_env, LOAD_MODE=lm) if lm else base_env


def find_nvml_preload():
    """(optional) On kernel/userspace libnvidia-ml version mismatch, find the matching lib → LD_PRELOAD.
    Not needed in normal environments (returns None). Can be set explicitly via the NVML_PRELOAD env var;
    if unset, only standard system lib paths are searched (no machine-specific paths hardcoded)."""
    env_p = os.environ.get('NVML_PRELOAD')
    if env_p:
        return env_p
    try:
        drv = open('/proc/driver/nvidia/version').read().split('Module')[1].split()[0]
    except Exception:
        return None
    for base in ('/usr/lib/x86_64-linux-gnu', '/usr/lib', '/usr/lib64'):
        for p in glob.glob(f'{base}/**/libnvidia-ml.so.{drv}', recursive=True):
            return p
    return None


def run(cmd, log, env=None, cwd=None):
    print('  $', ' '.join(str(c) for c in cmd), flush=True)
    with open(log, 'w') as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              env=env, cwd=cwd).returncode


def stage_gen(dom, c, style, force, wd=None, extra=(), tag=''):
    """Generation. `wd` overrides the output dir, `extra` appends generation flags and `tag`
    labels the log line — that is all run_ablation.py needs to reuse this engine per variant."""
    out = wd or wd_of(dom, style); mesh = out / 'mesh.obj'
    lbl = f'{dom}/{style}{tag}'
    if mesh.exists() and not force:
        print(f'[{lbl}/gen] SKIP (mesh.obj exists)'); return True
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, FEA_NORMALIZE='0', D3DS2_PATCH_FEATS='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', DETERMINISTIC='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               # append, don't clobber a caller's PYTHONPATH
               PYTHONPATH=str(REPO / 'external' / 'Direct3D-S2')
                          + (os.pathsep + os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''),
               # CPU threads for the in-loop FEM BLAS/assembly (the GPU stage leaves cores idle)
               OMP_NUM_THREADS=FEA_THREADS, OPENBLAS_NUM_THREADS=FEA_THREADS,
               MKL_NUM_THREADS=FEA_THREADS,
               # in-loop FEM: fenics python + absolute-path script (if missing, FEA fails every step → generation runs FEA-off)
               FEA_FENICS_BIN=PY_FEN, FEA_FENICS_SCRIPT=str(CODE / 'fenics_fea_bracket.py'))
    env = gen_env(ROOT / c['config'], env)   # LOAD_MODE from config (per-domain FEA load direction)
    pl = find_nvml_preload()
    if pl: env['LD_PRELOAD'] = pl + (':' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
    # Fully config-driven: every generation hyperparameter, data path (bc_proper, bracket_occ,
    # fea_domain_dir, fea_bracket_stl), seed and determinism come from the per-domain config
    # (top-level seed + stages.mesh) via the generation script's apply_config. The orchestrator
    # passes only --config and per-run I/O (target-dir/out + the per-run tet cache). cwd=DATA_ROOT
    # so the config's relative data/ paths resolve against the data root; the backbone loads via D3DS2_ROOT.
    cmd = [PY_D3D, str(CODE / 'generate_with_physics_guidance.py'),
           '--config', str(ROOT / c['config']),
           '--target-dir', str(DATA_ROOT / f'data/{dom}/conditioning/{style}'),
           '--out', str(out)] + list(extra)
    # NB: no --fea-mesh-cache here. The in-loop FEM tet mesh is the DESIGN DOMAIN, identical for
    # every prompt, so run_prep.py builds it once per domain and stages.mesh.fea_mesh_cache points
    # every run at that shared file (the solver locks it and reports "reusing cached mesh").
    rc = run(cmd, out / 'gen.log', env=env, cwd=str(DATA_ROOT))
    ok = rc == 0 and mesh.exists()
    print(f'[{lbl}/gen] {"OK" if ok else "FAIL(rc=%d)"%rc}  -> {mesh}')
    return ok


def stage_post(dom, c, style, force, wd=None, tag=''):
    out = wd or wd_of(dom, style); mesh = out / 'mesh.obj'
    hyb, fin = out / 'hybrid.obj', out / 'final.obj'
    lbl = f'{dom}/{style}{tag}'
    # both outputs must exist to skip: hybrid.obj is the intermediate the remesh reads, and a run
    # dir holding only one of the two is a half-finished post that should be redone.
    if fin.exists() and hyb.exists() and not force:
        print(f'[{lbl}/post] SKIP (hybrid.obj + final.obj exist)'); return True
    if not mesh.exists():
        print(f'[{lbl}/post] FAIL: mesh.obj missing (generate first)'); return False
    env = dict(os.environ)
    cfg = str(ROOT / c['config'])
    # Config-driven post (stages.post): mesh-level union + design-domain clip (per-domain peg/paths)
    # → isotropic remesh + Laplacian smoothing. Orchestrator passes only --config and per-stage I/O.
    rc = run([PY_D3D, str(CODE / 'post_hybrid_union_clip.py'), '--config', cfg,
         '--in', str(mesh), '--out', str(hyb)], out / 'post.log', env=env, cwd=str(DATA_ROOT))
    if rc != 0 or not hyb.exists():
        print(f'[{lbl}/post] FAIL: post_hybrid'); return False
    rc = run([PY_D3D, str(CODE / 'surface_remesh_pre.py'), '--config', cfg,
         '--in', str(hyb), '--out', str(fin)], out / 'remesh.log', env=env, cwd=str(DATA_ROOT))
    ok = rc == 0 and fin.exists()
    print(f'[{lbl}/post] {"OK" if ok else "FAIL"}  -> {fin}')
    return ok


def stage_fea(dom, c, style, force, wd=None, tag=''):
    # a posteriori FEA on `final.obj` — the part the pipeline actually delivers. The input is the
    # remeshed/smoothed surface, not the raw boolean (`hybrid.obj`): validating the delivered
    # geometry is the point, and the smoothed surface also tetrahedralizes far more reliably
    # (measured: every domain lands on the first HXT rung, where hybrid.obj needed remesh
    # escalation or failed outright on fine lattices). fea_prep_and_run.py still prepends its own
    # isotropic remesh + pymeshfix (NOT quadric decimation, so thin members survive), then runs
    # --no-repair gmsh tet + DOLFINx.
    out = wd or wd_of(dom, style); fea_in = out / 'final.obj'; feadir = out / 'fea'
    summ = feadir / 'fea_tet_summary.json'
    lbl = f'{dom}/{style}{tag}'
    if summ.exists() and not force and compliance_of(out, sane_c(c)) is not None:
        d = json.load(open(summ)); print(f'[{lbl}/fea] SKIP  C={d["compliance"]*1e3:.3f} mJ'); return True
    if not fea_in.exists():
        print(f'[{lbl}/fea] FAIL: final.obj missing (post first)'); return False
    feadir.mkdir(parents=True, exist_ok=True)
    # A failed retry must not leave an earlier summary looking like a new result.
    if summ.exists():
        summ.unlink()
    env = dict(os.environ, OMP_NUM_THREADS=FEA_THREADS)
    cmd = [PY_FEN, str(CODE / 'fea_prep_and_run.py'), '--config', str(ROOT / c['config']),
           '--in', str(fea_in), '--out-dir', str(feadir)]   # fixtures/material/load/tet from stages.fea
    rc = run(cmd, feadir / 'fea.log', env=env, cwd=str(DATA_ROOT))
    ok = rc == 0 and compliance_of(out, sane_c(c)) is not None
    if ok:
        d = json.load(open(summ))
        print(f'[{lbl}/fea] OK  C={d["compliance"]*1e3:.3f} mJ  vm_max={d["vm_max"]/1e6:.1f} MPa')
    else:
        print(f'[{lbl}/fea] FAIL(rc={rc}) — log: {feadir/"fea.log"}')
    return ok


STAGES = {'gen': stage_gen, 'post': stage_post, 'fea': stage_fea}


def compliance_of(wd, bound=SANE_C_DEFAULT):
    """Verified compliance (J) read back from a run dir's FEA summary — the single reader used by
    run_ablation.py's comparison table and refea.py's recovery check. None when the summary is
    missing, unreadable, or outside the sane range (see sane_c / SANE_C_DEFAULT)."""
    try:
        c = float(json.load(open(Path(wd) / 'fea' / 'fea_tet_summary.json'))['compliance'])
    except Exception:
        return None
    return c if 0 < c < bound else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain', default='all', help='bracket|motor_mount|link|all')
    ap.add_argument('--style', default='all', help='style key, or "all" for every provided conditioning style')
    ap.add_argument('--stage', default='all', help='gen|post|fea|all (comma-separated allowed)')
    ap.add_argument('--force', action='store_true', help='re-run even if output exists')
    ap.add_argument('--grid', action='store_true',
                    help='after all stages finish, render a per-domain result grid (render_results_grid.py)')
    args = ap.parse_args()
    doms = list(DOMAINS) if args.domain == 'all' else args.domain.split(',')
    stages = ['gen', 'post', 'fea'] if args.stage == 'all' else args.stage.split(',')
    os.chdir(DATA_ROOT)
    print(f'== run_from_image: domains={doms} stages={stages} force={args.force} ==')
    results = {}
    for dom in doms:
        c = DOMAINS[dom]
        styles = styles_of(dom) if args.style == 'all' else args.style.split(',')
        for style in styles:
            print(f'\n########## {dom} / {style} ##########')
            for st in stages:
                t0 = time.time()
                ok = STAGES[st](dom, c, style, args.force)
                results[(dom, style, st)] = ok
                print(f'    ({time.time()-t0:.0f}s)')
                if not ok:
                    print(f'  [{dom}/{style}] stage {st} failed → stopping remaining stages'); break
    print('\n== summary ==')
    for (dom, style, st), ok in results.items():
        print(f'  {dom:12} {style:20} {st:5} {"OK" if ok else "FAIL"}')
    if args.grid:
        from render_results_grid import render_grid, detect_variants
        print('\n== result grids ==')
        for dom in doms:
            out = render_grid(dom, variants=detect_variants(dom))
            print(f'  [{dom}] grid -> {out}')


if __name__ == '__main__':
    main()
