"""Conditioning step 3 — the IP-Adapter style reference: SDXL text-to-image, then normalized.

The reference is not the raw SDXL image. The recorded recipe is

    SDXL 1024 dense  →  centre crop  →  upscale back to 1024  →  binarize + polarity normalize

so the reference ends up a flat two-tone tile: **black structure on white through-holes**, with the
black area held in a target band (the shipped references measure black_frac 0.29-0.41, median 0.35
across 237 files). That is what the reference has to say to IP-Adapter — "openings this large, this
much material". A raw SDXL render says something else entirely: it is dark, continuous-tone and
fine-grained, which pushes the inpaint step toward a uniform silhouette.

The crop is what sets feature scale: cropping a fraction of the dense render and upscaling makes
the pattern coarser. Thresholding at the `black_frac` quantile then fixes the black/white split
regardless of how bright the render came out, so the reference always carries the same
material-to-void ratio.

Prompt text: the positive is the style's `descriptor` joined with `stages.cond_ref.ref_common`
(a shared "bold black structural lattice on pure white background ..." instruction) and the
negative is `stages.cond_ref.negative`, which suppresses the fine dense mesh SDXL defaults to.
`run_conditioning.py` assembles the positive; every knob lives in `stages.cond_ref`.

Usage:
  python code/conditioning/text2img_style_ref.py --config configs/link.json \
      --prompt "<descriptor>, <ref_common>" --out <ref.png> [--seed 42]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_config import apply_config


def normalize_ref(img, crop_frac, out_size, black_frac):
    """Centre-crop -> upscale -> binarize so that `black_frac` of the tile is black structure.

    The threshold is the black_frac quantile of the luminance histogram, so the split holds
    whatever the render's overall brightness is. Polarity is fixed by construction: darker pixels
    become the black structure, brighter ones the white through-holes."""
    if crop_frac and 0 < crop_frac < 1:
        w, h = img.size
        cw, ch = int(w * crop_frac), int(h * crop_frac)
        left, top = (w - cw) // 2, (h - ch) // 2
        img = img.crop((left, top, left + cw, top + ch))
    img = img.resize((out_size, out_size), Image.LANCZOS)
    a = np.asarray(img.convert('L'))
    # Pick the threshold whose achieved black fraction is closest to the target. A plain quantile
    # overshoots badly when the render has a large flat region at one value (SDXL saturates to pure
    # black): every tied pixel lands on the same side of `<=`. Scanning all 256 cut points and
    # scoring |achieved - target| makes the criterion hold whatever the histogram looks like.
    cdf = np.cumsum(np.bincount(a.ravel(), minlength=256)) / a.size
    thr = int(np.argmin(np.abs(cdf - black_frac)))
    b = np.where(a <= thr, 0, 255).astype(np.uint8)
    return Image.fromarray(b).convert('RGB'), thr, float((b == 0).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=None, help='run config JSON; stages.cond_ref fills every knob')
    ap.add_argument('--prompt', required=True, help='descriptor + ref_common, assembled by the caller')
    ap.add_argument('--out', required=True, help='output reference PNG')
    ap.add_argument('--negative', default=None, help='stages.cond_ref.negative')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--guidance', type=float, default=8.0)
    ap.add_argument('--gen-size', type=int, default=1024, help='SDXL generation resolution')
    ap.add_argument('--crop-frac', type=float, default=0.35,
                    help='centre crop as a fraction of the render (recorded range 0.28-0.42); '
                         'smaller = coarser pattern. 0 or 1 disables the crop.')
    ap.add_argument('--out-size', type=int, default=1024, help='reference size after upscale')
    ap.add_argument('--black-frac', type=float, default=0.35,
                    help='target black (structure) area fraction; recorded band 0.30-0.45')
    ap.add_argument('--raw-out', default=None, help='also save the pre-normalization render here')
    apply_config(ap, stage='cond_ref')
    args = ap.parse_args()
    if not args.negative:
        ap.error('negative missing — set stages.cond_ref.negative in the config')

    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    gen = torch.Generator('cuda').manual_seed(args.seed)
    img = pipe(prompt=args.prompt, negative_prompt=args.negative,
               width=args.gen_size, height=args.gen_size,
               num_inference_steps=args.steps, guidance_scale=args.guidance,
               generator=gen).images[0]

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    if args.raw_out:
        os.makedirs(os.path.dirname(args.raw_out) or '.', exist_ok=True)
        img.save(args.raw_out)
    ref, thr, got = normalize_ref(img, args.crop_frac, args.out_size, args.black_frac)
    ref.save(args.out)
    print(f'  ref: crop {args.crop_frac} -> {args.out_size}^2  threshold {thr:.0f}  '
          f'black_frac {got:.3f} (target {args.black_frac})')
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
