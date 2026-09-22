"""Convert every non-black (i.e. non-dark-structure) pixel to white.

Purpose: in styled cond images, keep only the black structure and unify grey
BC / shading / background to white. A luminance-based soft threshold gives a
smooth transition without halos at anti-aliasing edges.

  L <= lo  : keep original (keep mode) or pure black (binary mode)
  L >= hi  : pure white
  lo<L<hi  : linear blending (soft band)
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def convert(img: Image.Image, lo: int, hi: int, mode: str,
            protect: np.ndarray | None = None,
            protect_fill: str = 'black') -> Image.Image:
    """protect: bool HxW — True pixels (e.g. BC) are excluded from white conversion.
    protect_fill: 'black' = render BC as pure black (so it reads as material/structure),
                  'original' = keep the original pixels."""
    rgb = np.asarray(img.convert('RGB')).astype(np.float32)
    L = rgb @ np.array([0.299, 0.587, 0.114], np.float32)   # luminance
    t = np.clip((L - lo) / max(hi - lo, 1), 0.0, 1.0)[..., None]  # 0=keep, 1=white
    base = np.zeros_like(rgb) if mode == 'binary' else rgb
    out = base * (1 - t) + 255.0 * t
    if protect is not None:
        fill = np.zeros_like(rgb) if protect_fill == 'black' else rgb
        out = np.where(protect[..., None], fill, out)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True, help='input PNG directory (recursive)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--lo', type=int, default=70,
                    help='luminance at or below this = structure (kept/black)')
    ap.add_argument('--hi', type=int, default=140,
                    help='luminance at or above this = pure white')
    ap.add_argument('--mode', choices=['keep', 'binary'], default='keep',
                    help='keep: preserve original texture in dark areas / binary: pure black-and-white')
    ap.add_argument('--protect-mask-dir', default=None,
                    help='BC protect-mask directory (masks_aligned format). Uses the file stem to '
                         'find <view>_bc.png and keeps that region as original (e.g. v_top.png -> v_top_bc.png)')
    ap.add_argument('--protect-dilate', type=int, default=2,
                    help='outward margin of the protect mask (px)')
    ap.add_argument('--protect-fill', choices=['black', 'original'], default='black',
                    help='BC region handling: black=pure black (default, reads as material) / original=keep original')
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    pngs = sorted(in_dir.rglob('*.png'))
    if not pngs:
        raise SystemExit(f'ERROR: no png under {in_dir}')
    for p in pngs:
        rel = p.relative_to(in_dir)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(p)
        protect = None
        if args.protect_mask_dir:
            mp = Path(args.protect_mask_dir) / f'{p.stem}_bc.png'
            if mp.exists():
                m = Image.open(mp).convert('L')
                if m.size != img.size:
                    m = m.resize(img.size, Image.NEAREST)
                if args.protect_dilate > 0:
                    from PIL import ImageFilter
                    m = m.filter(ImageFilter.MaxFilter(2 * args.protect_dilate + 1))
                protect = np.asarray(m) > 128
            else:
                print(f'  WARN: no protect mask -> converting without protection: {mp}')
        convert(img, args.lo, args.hi, args.mode, protect, args.protect_fill).save(dst)
        tag = ' [BC protected]' if protect is not None else ''
        print(f'  {rel}{tag}')
    print(f'DONE {len(pngs)} files -> {out_dir} (lo={args.lo}, hi={args.hi}, {args.mode})')


if __name__ == '__main__':
    main()
