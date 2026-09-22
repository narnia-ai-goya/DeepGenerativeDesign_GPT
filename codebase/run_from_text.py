#!/usr/bin/env python
"""Text-first reproduction: style prompt -> SDXL conditioning image -> mesh -> post -> a-posteriori FEA.

The full chain including Stage 1 (conditioning-image generation). For each (domain, style):
  1. text prompt (configs/curated_prompts.json)  --SDXL / IP-Adapter + cross-view attention-->
     multi-view conditioning images written to data/<domain>/conditioning/<style>/
     (via run_conditioning.gen_conditioning), then
  2. that conditioning  --Direct3D-S2 inference-time guidance-->  mesh  ->  post-processing  ->
     a-posteriori FEA (the same stage engine as run_from_image.py).

Stage 1 needs the public SDXL / IP-Adapter weights (auto-downloaded from the Hugging Face Hub
on first run) plus pyvista / compel / diffusers, and a GPU. If you only want to reproduce from
the conditioning images already shipped in the package, use `run_from_image.py` instead (no
text-to-image stage, no HF downloads).

'all' styles = the five curated default styles per domain (configs/curated_prompts.json
`default_styles`); pass --style <key> to pick any of the curated 100/20/20 prompts. Each stage
is resumable: an existing conditioning image / mesh is reused unless --force is given.

Usage:
    python run_from_text.py --domain bracket --style de_crackle
    python run_from_text.py --domain all                        # 3 domains x 5 default styles: prompt -> ... -> FEA
    python run_from_text.py --domain link --style de_gyroid --force
"""
import argparse, json, os, time
from run_from_image import DOMAINS, STAGES, ROOT, DATA_ROOT
from run_conditioning import gen_conditioning

CURATED = json.load(open(ROOT / 'configs/curated_prompts.json'))


def default_styles(dom):
    """The five curated default styles shipped for a domain (configs/curated_prompts.json)."""
    return CURATED[dom]['default_styles']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain', default='all', help='bracket|motor_mount|link|all')
    ap.add_argument('--style', default='all',
                    help='style key (see configs/curated_prompts.json), or "all" for the domain default styles')
    ap.add_argument('--stage', default='all',
                    help='gen|post|fea|all for the post-conditioning pipeline (conditioning always runs first)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--force', action='store_true',
                    help='re-run even if output exists (also regenerates the conditioning image)')
    ap.add_argument('--grid', action='store_true',
                    help='after all stages finish, render a per-domain result grid (render_results_grid.py)')
    args = ap.parse_args()
    doms = list(DOMAINS) if args.domain == 'all' else args.domain.split(',')
    stages = ['gen', 'post', 'fea'] if args.stage == 'all' else args.stage.split(',')
    os.chdir(DATA_ROOT)
    print(f'== run_from_text: domains={doms} stages=conditioning+{stages} force={args.force} ==')
    results = {}
    for dom in doms:
        c = DOMAINS[dom]
        styles = default_styles(dom) if args.style == 'all' else args.style.split(',')
        for style in styles:
            print(f'\n########## {dom} / {style} ##########')
            # Stage 1: text prompt -> conditioning image (SDXL). Skipped if it already exists (unless --force).
            t0 = time.time()
            cond = gen_conditioning(dom, c, style, args.seed, args.force)
            print(f'    (conditioning {time.time()-t0:.0f}s)')
            if not cond:
                print(f'  [{dom}/{style}] conditioning failed -> skipping remaining stages'); continue
            # Stage 2-4: mesh -> post -> FEA (identical engine to run_from_image.py).
            for st in stages:
                t0 = time.time()
                ok = STAGES[st](dom, c, style, args.force)
                results[(dom, style, st)] = ok
                print(f'    ({time.time()-t0:.0f}s)')
                if not ok:
                    print(f'  [{dom}/{style}] stage {st} failed -> stopping remaining stages'); break
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
