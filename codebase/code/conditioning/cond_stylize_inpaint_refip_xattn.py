"""SDXL Inpainting 0.1 + external reference IP-Adapter + cross-view self-attention.

Combines two consistency mechanisms in a single forward pass on batched 5 views:
  - External reference image injected via IP-Adapter on cross-attn (attn2) — every
    view's diffusion sees the same external semantic (BOLD/ULTRA reference).
  - Cross-view K/V sharing (xattn) on self-attn (attn1) — 5 views' tokens are
    pooled so each view attends to all other views' latent context.

install_mv_attn(only_self=True) leaves attn2 (IP-Adapter) processors intact and
only swaps attn1 to MultiViewAttnProcessor.

Spatial mask preserves fix/load BC regions identically to refip.
"""
import argparse
import sys
import torch
from pathlib import Path
from PIL import Image, ImageFilter

# Shim for diffusers 0.38 + transformers <4.45: stub the missing Dinov2 classes
# so `autoencoder_rae` (referenced from autoencoders/__init__.py) doesn't crash
# the inpaint pipeline import. We never actually use AutoencoderRAE.
import transformers as _tf
if not hasattr(_tf, "Dinov2WithRegistersConfig"):
    class _StubDinov2:
        pass
    _tf.Dinov2WithRegistersConfig = _StubDinov2
    _tf.Dinov2WithRegistersModel = _StubDinov2

sys.path.insert(0, str(Path(__file__).parent))
NEGATIVE = 'low quality, blurry, distorted, text, watermark, cartoon, anime, painting, colorful, rust, wood, plastic, toy, simple, flat, 2D drawing, overexposed highlights, bright reflective surface, polished chrome, bleached white, washed out, near-white interior, thin frame, dissolved bracket, fragmented body, broken connections, icon, vector graphic, logo, pictogram, flat illustration, minimalist drawing, sticker, sketch, line art, clipart, monoline, high-contrast flat black shape, paper cutout, thin strut, lattice, skeletal, delicate, filigree, web, truss, topology-optimized strut, fine detail, perforated, mesh wireframe, wire frame, spider web, hollow cage, organic shape, blobby, lumpy, wavy outline, irregular bulges, clay sculpture, painterly, soft squishy material, melted plastic, uneven surface, asymmetric organic growth, cylindrical tube, tubular strut, sausage shape, hot dog, round bar, puffy inflated rubber, balloon, marshmallow, pillow'
from cond_stylize_inpaint_xattn import install_mv_attn  # attn1 only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="run config JSON; CLI overrides")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt", required=True, help="positive prompt text")
    ap.add_argument("--negative", default=NEGATIVE, help="negative prompt text")
    ap.add_argument("--prompt-suffix", default="",
                    help="text appended to every style prompt (e.g. extra 3D emphasis)")
    ap.add_argument("--ref-image", required=True, help="external reference PNG/JPG")
    ap.add_argument("--views", default="v00_front_lo,v02_right_lo,v04_back_lo,v06_left_lo,v_top")
    ap.add_argument("--strength", type=float, default=0.95)
    ap.add_argument("--feather", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ip-scale", type=float, default=1.0)
    ap.add_argument("--ip-mode", default="all", choices=["style", "all"])
    ap.add_argument("--base-model", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--xattn-mode", default="full", choices=["full", "anchor"],
                    help="full: all views share K/V mutually (default) / anchor: attend to self+anchor view only")
    ap.add_argument("--anchor-view", default="v_top",
                    help="reference view name when xattn-mode=anchor (must be in views)")
    ap.add_argument("--xattn-end-frac", type=float, default=1.0,
                    help="share K/V only on early steps whose progress is <= this value (1.0=all, default). "
                         "e.g. 0.5 -> share only the first 50%% of steps, later steps free for detail")
    ap.add_argument("--control", default="none", choices=["none", "depth", "canny"],
                    help="geometry-guiding ControlNet: depth(rendered depth) / canny(input edge)")
    ap.add_argument("--depth-dir", default=None,
                    help="per-view depth PNG directory when control=depth")
    ap.add_argument("--cn-scale", type=float, default=0.5,
                    help="ControlNet conditioning scale")
    ap.add_argument("--long-prompt", action='store_true',
                    help="enable compel-based long-prompt (>77 CLIP tokens) support")
    # run_config lives in code/, one level up from this package — add it so the script also
    # runs standalone (the orchestrator additionally puts code/ on PYTHONPATH).
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from run_config import apply_config
    apply_config(ap, stage='cond_stylize')
    args = ap.parse_args()
    # Also read long_prompt from config (stages.cond_stylize.long_prompt) if set
    if args.config:
        try:
            import json as _j
            _cfg = _j.load(open(args.config))
            if _cfg.get('stages', {}).get('cond_stylize', {}).get('long_prompt', False):
                args.long_prompt = True
        except Exception:
            pass

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    n_views = len(views)
    use_cn = args.control != "none"
    print(f"loading SDXL Inpainting + IP-Adapter + xattn "
          f"(n_views={n_views}, xattn={args.xattn_mode}, control={args.control})")
    if use_cn:
        from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
        cn_repo = ("diffusers/controlnet-depth-sdxl-1.0" if args.control == "depth"
                   else "diffusers/controlnet-canny-sdxl-1.0")
        cn = ControlNetModel.from_pretrained(cn_repo, torch_dtype=torch.float16)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            args.base_model, controlnet=cn,
            torch_dtype=torch.float16, variant="fp16",
        )
    else:
        from diffusers import StableDiffusionXLInpaintPipeline
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            args.base_model, torch_dtype=torch.float16, variant="fp16",
        )
    # 1) Load IP-Adapter on attn2 (cross-attn)
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.safetensors",
    )
    scale_cfg = {"down": {"block_2": args.ip_scale}} if args.ip_mode == "style" else args.ip_scale
    pipe.set_ip_adapter_scale(scale_cfg)
    print(f"  IP-Adapter scale: {scale_cfg}")

    # 2) Install xattn on attn1 (self-attn), leave attn2 (IP-Adapter) intact
    anchor_idx = views.index(args.anchor_view) if args.xattn_mode == "anchor" else 0
    xattn_state = install_mv_attn(pipe.unet, n_views=n_views, only_self=True,
                                  mode=args.xattn_mode, anchor_idx=anchor_idx,
                                  end_frac=args.xattn_end_frac)
    xattn_state['total'] = args.steps
    print(f"  installed MultiViewAttnProcessor (mode={args.xattn_mode}, "
          f"anchor_idx={anchor_idx}, end_frac={args.xattn_end_frac})")

    pipe.to("cuda")
    # cpu_offload sends the text_encoder back to CPU via a forward hook, so it is
    # disabled in compel (long_prompt) mode. Full SDXL+IP-Adapter load is ~15GB.
    if not args.long_prompt:
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass

    prompt = args.prompt + args.prompt_suffix
    in_dir = Path(args.input_dir); mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ref_img = Image.open(args.ref_image).convert("RGB")
    print(f"  ref: {args.ref_image} size={ref_img.size}")

    imgs, masks, controls = [], [], []
    for v in views:
        src = in_dir / f"{v}.png"
        m = mask_dir / f"{v}_stylable.png"
        assert src.exists() and m.exists(), f"missing {v}"
        img = Image.open(src).convert("RGB").resize((args.width, args.height))
        mask = Image.open(m).convert("L").resize((args.width, args.height))
        if args.feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(args.feather))
        imgs.append(img); masks.append(mask)
        if args.control == "depth":
            dp = Path(args.depth_dir) / f"{v}.png"
            assert dp.exists(), f"missing depth {dp}"
            controls.append(Image.open(dp).convert("RGB").resize((args.width, args.height)))
        elif args.control == "canny":
            import numpy as np, cv2
            edges = cv2.Canny(np.asarray(img.convert("L")), 80, 200)
            controls.append(Image.fromarray(np.stack([edges]*3, -1)))

    print(f"\nbatching {n_views} views in one pipe call with xattn + IP-Adapter ref...")
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    extra = {}
    if use_cn:
        extra = dict(control_image=controls,
                     controlnet_conditioning_scale=args.cn_scale)
    # step-gated xattn: increment the shared counter after each denoising step
    if args.xattn_end_frac < 1.0:
        def _step_cb(pipe_, step, ts, kw):
            xattn_state['step'] = step + 1
            return kw
        extra['callback_on_step_end'] = _step_cb
    if args.long_prompt:
        # Compel: long prompt (>77 CLIP tokens) embedding for SDXL (both encoders).
        from compel import Compel, ReturnedEmbeddingsType
        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )
        pos_embeds, pos_pooled = compel([prompt] * n_views)
        neg_embeds, neg_pooled = compel([args.negative] * n_views)
        try:
            pos_embeds, neg_embeds = compel.pad_conditioning_tensors_to_same_length([pos_embeds, neg_embeds])
        except AttributeError:
            # compel >= 2.4: manual pad with zeros to the longer length
            max_len = max(pos_embeds.shape[1], neg_embeds.shape[1])
            def _pad(t, target):
                if t.shape[1] == target: return t
                pad = torch.zeros(t.shape[0], target - t.shape[1], t.shape[2],
                                  device=t.device, dtype=t.dtype)
                return torch.cat([t, pad], dim=1)
            pos_embeds = _pad(pos_embeds, max_len)
            neg_embeds = _pad(neg_embeds, max_len)
        print(f"  [compel] long-prompt enabled: pos_tokens={pos_embeds.shape[1]}, neg_tokens={neg_embeds.shape[1]}")
        prompt_kwargs = dict(
            prompt_embeds=pos_embeds, pooled_prompt_embeds=pos_pooled,
            negative_prompt_embeds=neg_embeds, negative_pooled_prompt_embeds=neg_pooled,
        )
    else:
        prompt_kwargs = dict(
            prompt=[prompt] * n_views,
            negative_prompt=[args.negative] * n_views,
        )
    result = pipe(
        **prompt_kwargs,
        image=imgs, mask_image=masks,
        ip_adapter_image=ref_img,
        strength=args.strength,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        generator=gen,
        width=args.width, height=args.height,
        **extra,
    ).images

    # Post-composite: keep the input as-is where mask=0 (avoids the issue where
    # SDXL Inpainting does not preserve init noise at strength=1.0). The same
    # feather is applied so the boundary stays smooth.
    import numpy as np
    for v, gen_im, src_im, mk_im in zip(views, result, imgs, masks):
        gen = np.array(gen_im.convert('RGB'), dtype=np.float32)
        src = np.array(src_im.convert('RGB'), dtype=np.float32)
        m   = np.array(mk_im.convert('L'),   dtype=np.float32) / 255.0  # 0=keep, 1=replace
        alpha = m[:,:,None]
        out_arr = (alpha * gen + (1 - alpha) * src).clip(0, 255).astype(np.uint8)
        op = out_dir / f"{v}.png"
        Image.fromarray(out_arr).save(str(op))
        print(f"  {v} -> {op}")
    print(f"\nDONE -> {out_dir}")


if __name__ == "__main__":
    main()
