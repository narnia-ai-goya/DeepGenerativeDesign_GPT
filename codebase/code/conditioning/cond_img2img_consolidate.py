"""Conditioning step 6 — SDXL img2img consolidation pass over the normalized views.

The last step of the conditioning chain that produced the shipped images. It re-runs the
luminance-normalized views (step 5's output) through plain SDXL img2img — no mask, no
ControlNet — which unifies tone across the six views and sharpens the carved openings the
masked inpaint left soft.

IP-Adapter is loaded because the pipeline requires image embeds, but the shipped runs used
`ip_scale = 0.0`, i.e. the reference image contributes nothing: the style is already baked into
the input views by step 4. Note `strength` also sets the step count — SDXL img2img runs
`int(steps x strength)` denoising steps, so steps 40 / strength 0.7 = 28 actual steps.

Every value — `ip_scale`, `strength`, `steps`, `guidance`, and the `prompt` / `negative` text —
lives in `configs/<domain>.json` under `stages.cond_img2img`. This pass does NOT take the style
prompt: it is a finishing render pass ("photorealistic 3D render, glossy black anodized metal
surface, ...") applied after the style has already been carved in by step 4.

Usage:
  python code/conditioning/cond_img2img_consolidate.py --config configs/link.json \
      --input-dir <white/>  --out-dir <final conditioning/>  --ref-image <ref.png> \
      --prompt "<positive>" --negative "<negative>"
"""
import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL

# run_config lives in code/, one level up from this package — add it so the script also
# runs standalone (the orchestrator additionally puts code/ on PYTHONPATH).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_config import apply_config


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=None, help='run config JSON; stages.cond_img2img fills '
                                                   'ip_scale / strength / steps / guidance')
    ap.add_argument('--input-dir', required=True, help='normalized views from step 5 (white/)')
    ap.add_argument('--out-dir', required=True, help='final conditioning the generator reads')
    ap.add_argument('--ref-image', default=None,
                    help='IP-Adapter reference. Inert at ip_scale=0; falls back to the first '
                         'input view if the path is missing (the pipeline still needs embeds).')
    ap.add_argument('--prompt', default=None,
                    help='normally comes from stages.cond_img2img.prompt in the config')
    ap.add_argument('--negative', default=None,
                    help='normally comes from stages.cond_img2img.negative in the config')
    ap.add_argument('--ip-scale', type=float, default=0.0)
    ap.add_argument('--strength', type=float, default=0.7)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--guidance', type=float, default=9.0)
    ap.add_argument('--seed', type=int, default=42)
    apply_config(ap, stage='cond_img2img')
    args = ap.parse_args()
    if not args.prompt or not args.negative:
        ap.error('prompt/negative missing — set stages.cond_img2img.prompt and .negative in the '
                 'config (or pass --prompt/--negative)')

    os.makedirs(args.out_dir, exist_ok=True)

    print('loading SDXL Img2Img + IP-Adapter...', flush=True)
    vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16)
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0',
        vae=vae, torch_dtype=torch.float16, variant='fp16')
    pipe.to('cuda')
    pipe.load_ip_adapter(
        'h94/IP-Adapter', subfolder='sdxl_models', weight_name='ip-adapter_sdxl.safetensors')
    pipe.set_ip_adapter_scale(args.ip_scale)
    print(f'  ip_scale={args.ip_scale}, strength={args.strength}, guidance={args.guidance}, '
          f'steps={args.steps} → {int(args.steps * args.strength)} denoising steps', flush=True)

    views = sorted(f for f in os.listdir(args.input_dir) if f.endswith('.png'))
    if not views:
        sys.exit(f'no .png views in {args.input_dir}')
    if args.ref_image and Path(args.ref_image).exists():
        ref = Image.open(args.ref_image).convert('RGB')
    else:
        ref = Image.open(f'{args.input_dir}/{views[0]}').convert('RGB')
        print(f'  ref-image fallback: using {views[0]} from input-dir', flush=True)

    for f in views:
        view = f[:-4]
        img = Image.open(f'{args.input_dir}/{f}').convert('RGB')
        W, H = img.size
        gen = torch.Generator('cuda').manual_seed(args.seed)
        out = pipe(prompt=args.prompt, negative_prompt=args.negative,
                   image=img, ip_adapter_image=ref,
                   strength=args.strength, num_inference_steps=args.steps,
                   guidance_scale=args.guidance, generator=gen,
                   height=H, width=W).images[0]
        out.save(f'{args.out_dir}/{view}.png')
        print(f'  {view} done', flush=True)

    print(f'\nfinal: {len(views)} views → {args.out_dir}/')


if __name__ == '__main__':
    main()
