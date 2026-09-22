"""SDXL Inpainting 0.1 + cross-view self-attention for true MV consistency.

Strategy:
  - Batch all V views into a single pipe call (batch_size = V).
  - Replace UNet self-attention with MultiViewAttnProcessor that lets each
    view's queries attend to KV from ALL views concatenated.
  - Cross-attention to text encoder is unchanged.
  - Spatial mask preserves fix/load BC regions identically to inpaint.py.

Implementation note: only self-attention is patched (encoder_hidden_states is None).
Attention slicing recommended to fit 1024² × 5 views in ~16GB VRAM.
"""
import argparse
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
NEGATIVE = ("smooth blob, solid body, few holes, no holes, filled, dense tiny, blurry, "
            "deformed, text, watermark, photo, glossy, colorful")


class MultiViewAttnProcessor:
    """Self-attention that shares K/V across views in the batch dim.

    mode='full'  : each view's query attends to the concat of tokens from ALL
                   views (default behavior).
    mode='anchor': each view attends only to its own tokens + the anchor view's
                   tokens (anchor_idx), tying every view to a reference view for
                   higher consistency (consistency ablation).
    Cross-attention (encoder_hidden_states != None) is unchanged.

    state: {'step':int,'total':int} shared dict. If end_frac<1.0, K/V sharing is
           disabled (plain self-attn) on the later steps once progress exceeds
           end_frac. Low-frequency structure is set early, so consistency is
           enforced only early while later steps are free to add detail.
    """
    def __init__(self, n_views=5, mode='full', anchor_idx=0, end_frac=1.0, state=None):
        self.n_views = n_views
        self.mode = mode
        self.anchor_idx = anchor_idx
        self.end_frac = end_frac
        self.state = state if state is not None else {'step': 0, 'total': 1}

    def _share_active(self):
        if self.end_frac >= 1.0:
            return True
        frac = self.state['step'] / max(self.state['total'], 1)
        return frac <= self.end_frac

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape
        if encoder_hidden_states is None:
            encoder_hidden_states_local = hidden_states
            is_self_attn = True
        else:
            encoder_hidden_states_local = encoder_hidden_states
            is_self_attn = False
            if attn.norm_cross is not None:
                encoder_hidden_states_local = attn.norm_encoder_hidden_states(encoder_hidden_states_local)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states_local)
        value = attn.to_v(encoder_hidden_states_local)

        # For self-attention only: concatenate K, V across the view dim.
        # batch_size = effective_batch = (cfg_dup * n_views) where cfg_dup is 1 or 2.
        if (is_self_attn and batch_size % self.n_views == 0
                and batch_size >= self.n_views and self._share_active()):
            cfg_dup = batch_size // self.n_views
            d = key.shape[-1]
            if self.mode == 'anchor':
                # each view: K/V = concat(own tokens, anchor-view tokens) -> (batch, 2*seq, d)
                kv = key.view(cfg_dup, self.n_views, sequence_length, d)
                vv = value.view(cfg_dup, self.n_views, sequence_length, d)
                ka = kv[:, self.anchor_idx:self.anchor_idx + 1].expand(-1, self.n_views, -1, -1)
                va = vv[:, self.anchor_idx:self.anchor_idx + 1].expand(-1, self.n_views, -1, -1)
                key = torch.cat([kv, ka], dim=2).reshape(batch_size, 2 * sequence_length, d)
                value = torch.cat([vv, va], dim=2).reshape(batch_size, 2 * sequence_length, d)
            else:
                # Reshape: (cfg_dup, n_views, seq, d) → concat seq across views
                #   key: (cfg_dup, n_views * seq, d)
                key = key.view(cfg_dup, self.n_views, sequence_length, d).reshape(
                    cfg_dup, self.n_views * sequence_length, d
                )
                value = value.view(cfg_dup, self.n_views, sequence_length, d).reshape(
                    cfg_dup, self.n_views * sequence_length, d
                )
                # Tile to match query batch (one K/V copy per view query)
                key = key.repeat_interleave(self.n_views, dim=0)
                value = value.repeat_interleave(self.n_views, dim=0)
                # Now key/value shape: (cfg_dup * n_views, n_views * seq, d) = (batch_size, n_views*seq, d)

        head_dim = query.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(B, C, H, W)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def install_mv_attn(unet, n_views, only_self=True, mode='full', anchor_idx=0,
                    end_frac=1.0):
    """Replace self-attn (attn1) processors only. Leave cross-attn (attn2) — they
    may carry IP-Adapter processors that must stay installed.

    Returns the shared step-state dict (shared by all processors). When
    end_frac<1.0 is used, the pipeline's callback_on_step_end must increment
    state['step'].
    """
    state = {'step': 0, 'total': 1}
    procs = {}
    for name, proc in unet.attn_processors.items():
        # In SDXL UNet, self-attn names end with `attn1.processor`, cross-attn `attn2.processor`
        if only_self and not name.endswith("attn1.processor"):
            procs[name] = proc  # keep existing (e.g., IPAdapterAttnProcessor on attn2)
        else:
            procs[name] = MultiViewAttnProcessor(
                n_views=n_views, mode=mode, anchor_idx=anchor_idx,
                end_frac=end_frac, state=state)
    unet.set_attn_processor(procs)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt", default=None, help="positive prompt text")
    ap.add_argument("--negative", default=None, help="override negative prompt")
    ap.add_argument("--views", default="v00_front_lo,v02_right_lo,v04_back_lo,v06_left_lo,v_top")
    ap.add_argument("--strength", type=float, default=0.95)
    ap.add_argument("--feather", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-model", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    args = ap.parse_args()

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    n_views = len(views)
    print(f"loading SDXL Inpainting + cross-view attention (n_views={n_views})")
    from diffusers import StableDiffusionXLInpaintPipeline
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.float16, variant="fp16",
    )
    install_mv_attn(pipe.unet, n_views=n_views, only_self=True)
    print(f"installed MultiViewAttnProcessor on self-attn layers")
    pipe.to("cuda")
    try:
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()
    except Exception:
        pass

    prompt = args.prompt if args.prompt else ""
    neg_prompt = args.negative if args.negative else NEGATIVE
    in_dir = Path(args.input_dir); mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    imgs, masks = [], []
    for v in views:
        src = in_dir / f"{v}.png"
        m = mask_dir / f"{v}_stylable.png"
        assert src.exists() and m.exists(), f"missing {v}"
        img = Image.open(src).convert("RGB").resize((1024, 1024))
        mask = Image.open(m).convert("L").resize((1024, 1024))
        if args.feather > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(args.feather))
        imgs.append(img); masks.append(mask)

    print(f"\nbatching {n_views} views in one pipe call...")
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    result = pipe(
        prompt=[prompt] * n_views,
        negative_prompt=[neg_prompt] * n_views,
        image=imgs, mask_image=masks,
        strength=args.strength,
        guidance_scale=args.guidance,
        num_inference_steps=args.steps,
        generator=gen,
        width=1024, height=1024,
    ).images

    for v, im in zip(views, result):
        op = out_dir / f"{v}.png"
        im.save(str(op))
        print(f"  {v} -> {op}")
    print(f"\nDONE -> {out_dir}")


if __name__ == "__main__":
    main()
