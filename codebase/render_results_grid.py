#!/usr/bin/env python
"""Render per-domain result grids: ON (+ OFF / post-hoc) with verified compliance.

Reads the meshes and a posteriori FEA summaries produced under experiments/<domain>/<style>/ and
lays them out one row per style, one column per variant. Each mesh shown is `hybrid.obj`, the
polished mesh the FEA stage actually verified (all three domains), so the rendered shape matches
the reported compliance. Rendering is off-screen (pyvista).

Usage:
    python render_results_grid.py --domain all                       # columns: ON | OFF
    python render_results_grid.py --domain bracket --variants on,off,posthoc
    python render_results_grid.py --domain all --variants auto        # only variants present on disk
Output: experiments/<domain>/grid_<domain>.png (one grid per domain).

Also importable: render_grid(dom, variants) is called by run_from_text.py / run_from_image.py --grid.
"""
import argparse, json, os
from pathlib import Path
from run_from_image import DOMAINS, EXP, styles_of, wd_of, variant_dir


def _compliance(*paths):
    for f in paths:
        try:
            return json.load(open(f))['compliance'] * 1e3
        except Exception:
            pass
    return None


def detect_variants(dom):
    """'on' plus whichever of off/posthoc actually have a verified result on disk."""
    styles = styles_of(dom)
    vs = ['on']
    for v in ('off', 'posthoc'):
        if any((variant_dir(dom, s, v) / 'fea' / 'fea_tet_summary.json').exists()
               for s in styles):
            vs.append(v)
    return tuple(vs)


def render_grid(dom, variants=('on', 'off'), size=360, out=None):
    """Render one domain's result grid. Returns the output path (or None if no styles)."""
    os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
    import pyvista as pv
    import trimesh
    from PIL import Image, ImageDraw, ImageFont
    try:
        pv.start_xvfb()
    except Exception:
        pass

    S = size
    fea_in = 'final.obj'          # the delivered mesh, which is also what stage_fea analyses
    styles = styles_of(dom)
    if not styles:
        return None
    try:
        fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
    except Exception:
        fnt = ImageFont.load_default()

    def blank(txt='n/a'):
        im = Image.new('RGB', (S, S), '#f2f2f2'); ImageDraw.Draw(im).text((6, S // 2), txt, fill='#999'); return im

    def render_mesh(f):
        if not Path(f).exists():
            return blank('pending')
        try:
            m = trimesh.load(str(f), force='mesh', process=False)
        except Exception:
            return blank('load fail')
        p = pv.Plotter(off_screen=True, window_size=(S, S)); p.set_background('white')
        p.add_mesh(pv.wrap(m), color='#b8c2cc', smooth_shading=True)
        p.enable_parallel_projection(); p.camera_position = 'iso'
        return Image.fromarray(p.screenshot(return_img=True))

    cols = list(variants)                          # result meshes only (no conditioning column)
    rows = []
    for s in styles:
        onc = _compliance(wd_of(dom, s) / 'fea' / 'fea_tet_summary.json')
        cells = []
        for i, col in enumerate(cols):
            d = variant_dir(dom, s, col)
            c = _compliance(d / 'fea' / 'fea_tet_summary.json')
            lbl = col.upper() + (f'  C={c:.3f} mJ' if c is not None else '  (n/a)')
            if col != 'on' and onc and c:
                lbl += f'  ({c / onc:.1f}x)'
            if i == 0:
                lbl = f'{s}\n' + lbl                # style name on the first column
            cells.append((render_mesh(d / fea_in), lbl))
        rows.append(cells)

    pad, cap, hdr = 8, 42, 30
    W = len(cols) * S + (len(cols) + 1) * pad
    H = len(rows) * (S + cap) + (len(rows) + 1) * pad + hdr
    g = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(g)
    d.text((pad, 6), f'{dom}  —  ' + ' vs '.join(c.upper() for c in cols)
           + f'   [FEA mesh: {fea_in}]', fill='black', font=fnt)
    for r, cells in enumerate(rows):
        for c, (im, cap_txt) in enumerate(cells):
            x = pad + c * (S + pad); y = hdr + pad + r * (S + cap + pad)
            g.paste(im, (x, y))
            for i, ln in enumerate(cap_txt.split('\n')):
                d.text((x + 4, y + S + 2 + i * 18), ln, fill='black', font=fnt)
    out = Path(out) if out else EXP / f'{dom}/grid_{dom}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    g.save(str(out))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain', default='all', help='bracket|motor_mount|link|all')
    ap.add_argument('--variants', default='on,off',
                    help="comma list from on,off,posthoc (default on,off), or 'auto' to use only the "
                         "variants that have a verified result on disk (what --grid passes)")
    ap.add_argument('--size', type=int, default=360)
    args = ap.parse_args()
    doms = list(DOMAINS) if args.domain == 'all' else args.domain.split(',')
    for dom in doms:
        variants = detect_variants(dom) if args.variants == 'auto' else tuple(args.variants.split(','))
        out = render_grid(dom, variants=variants, size=args.size)
        print(f'  [{dom}] grid ({",".join(variants)}) -> {out}' if out else f'  [{dom}] no styles found, skipped')


if __name__ == '__main__':
    main()
