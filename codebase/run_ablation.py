#!/usr/bin/env python
"""FEA ablation: with a single command, generate → post-process → a-posteriori FEA the three variants
fea-on / fea-off / post-hoc, then compare compliance (the primary ablation comparison: does in-loop FEM
guidance lower compliance).

The three variants differ only in generation flags; everything else comes from the per-domain config:
    on      : (no flag — the config's fea_w / sp_fea_w as-is, i.e. in-loop FEM guidance)
    off     : --fea-w 0     --sp-fea-w 0                (baseline)
    posthoc : --fea-w 0     --sp-fea-w 0  --posthoc-fea-steps 60 --posthoc-fea-lr 2e-3

Post-processing and validation (post/FEA) use the same single code and config as run_from_image. Per-variant
output goes to experiments/<dom>/<style>/abl/<variant>/ (resolved by run_from_image.variant_dir, which
refea.py and render_results_grid.py read back). Resumable (skip if present, re-run with --force).

Usage:
    python run_ablation.py --domain bracket                 # all of bracket on/off/posthoc
    python run_ablation.py --domain all --variant on,off     # specific variants only
"""
import argparse, os, time
from run_from_image import DOMAINS, DATA_ROOT, STAGES, abl_dir, compliance_of, styles_of, sane_c

# The three variants differ only in the FEM-guidance flags appended to the generation command.
# 'on' appends nothing: the per-domain config's fea_w / sp_fea_w are the ON setting.
VARIANTS = {
    'on':      [],
    'off':     ['--fea-w', '0', '--sp-fea-w', '0'],
    'posthoc': ['--fea-w', '0', '--sp-fea-w', '0',
                '--posthoc-fea-steps', '60', '--posthoc-fea-lr', '2e-3'],
}


def run_variant(dom, c, style, variant, force):
    """generate → post → a-posteriori FEA for one variant, reusing run_from_image's stage engine
    (same env, same config plumbing, same skip logic) with a per-variant output dir. Returns the
    verified compliance in J, or None if any stage failed."""
    wd = abl_dir(dom, style, variant)
    tag = f'/{variant}'
    if not STAGES['gen'](dom, c, style, force, wd=wd, extra=VARIANTS[variant], tag=tag):
        return None
    if not STAGES['post'](dom, c, style, force, wd=wd, tag=tag):
        return None
    STAGES['fea'](dom, c, style, force, wd=wd, tag=tag)
    return compliance_of(wd, sane_c(c))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain', default='all', help='bracket|motor_mount|link|all')
    ap.add_argument('--style', default='all', help='style key, or "all" for every provided conditioning style')
    ap.add_argument('--variant', default='on,off,posthoc', help='on|off|posthoc (comma-separated)')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    doms = list(DOMAINS) if args.domain == 'all' else args.domain.split(',')
    variants = args.variant.split(',')
    os.chdir(DATA_ROOT)
    res = {}   # (dom, style, variant) -> compliance (mJ) or None
    styles_for = lambda dom: styles_of(dom) if args.style == 'all' else args.style.split(',')
    for dom in doms:
        c = DOMAINS[dom]
        for style in styles_for(dom):
            for v in variants:
                print(f'\n########## {dom} / {style} / {v} ##########'); t0 = time.time()
                C = run_variant(dom, c, style, v, args.force)
                res[(dom, style, v)] = None if C is None else C * 1e3
                print(f'  -> C = {res[(dom,style,v)]}  ({time.time()-t0:.0f}s)')
    # ── comparison table (one row per domain × style) ──
    print('\n== FEA ablation: compliance (mJ) ==')
    print(f'{"domain":12}{"style":22}{"off":>9}{"on":>9}{"posthoc":>9}{"on vs off":>11}{"posthoc vs off":>15}')
    def pct(a, b): return f'{(a-b)/b*100:+.1f}%' if (a is not None and b) else '—'
    fmt = lambda x: f'{x:.3f}' if x is not None else 'FAIL'
    for dom in doms:
        for style in styles_for(dom):
            off, on, ph = res.get((dom, style, 'off')), res.get((dom, style, 'on')), res.get((dom, style, 'posthoc'))
            print(f'{dom:12}{style:22}{fmt(off):>9}{fmt(on):>9}{fmt(ph):>9}{pct(on,off):>11}{pct(ph,off):>15}')


if __name__ == '__main__':
    main()
