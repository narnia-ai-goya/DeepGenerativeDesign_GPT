#!/usr/bin/env python
"""Conditioning-image generation: prompt -> multi-view conditioning images.

This reproduces the upstream stage that turns a style prompt plus the design
domain into the multi-view conditioning images consumed by the generator
(`run_from_image.py --stage gen`, which reads `--target-dir <conditioning>`).

Chain (per style):
  1. render      : cond_render_pv.py            design body -> multi-view renders
  2. masks       : cond_render_masks.py  occlusion-aware stylable / BC masks
  3. reference   : text2img_style_ref.py  SDXL text-to-image, then centre-crop + upscale +
                   binarize -> a two-tone reference tile (black structure / white holes)
  4. inpaint     : cond_stylize_inpaint_refip_xattn.py  masked inpaint (+ IP-Adapter,
                   cross-view attention), boundary-condition regions frozen
  5. normalize   : nonblack_to_white.py  luminance normalization (BC regions protected) ->
                   a solid-silhouette + solid-BC image (intermediate, written to white/)
  6. consolidate : cond_img2img_consolidate.py  plain SDXL img2img over the six normalized
                   views (no mask, ip_scale 0) -> the final conditioning the generator reads.
                   This is the step the shipped images came from: they are byte-identical to the
                   original run's img2img output, and 2.26/255 away from the step-5 image.

Steps 1-2 depend only on the domain and are cached per domain. Steps 3-6 run per style.
The style prompt text is read from `configs/curated_prompts.json` (the curated 100/20/20
prompt sets per domain) and passed to the conditioning scripts directly.

The SDXL / IP-Adapter / ControlNet weights are public pretrained models, downloaded
from the Hugging Face Hub on first run (see README.md). They are NOT bundled.

Environments: steps 1-2 and 5 need pyvista/trimesh/PIL (the generation env, PY_GEO); steps 3, 4
and 6 need the SDXL stack (SD_PY, defaults to the same interpreter). Set SD_PY when the SDXL
pipelines should run in their own env.

Usage:
  # generation env active (diffusers + pyvista), then:
  python run_conditioning.py --domain bracket --style de_gyroid
  python run_conditioning.py --domain motor_mount --style de_swiss --seed 42
  # the final conditioning lands in data/<domain>/conditioning/<style>/, exactly where the
  # generator reads it, so run_from_image picks it up automatically:
  #   python run_from_image.py --domain bracket --style de_gyroid --stage all
"""
import argparse, json, os, sys, time
from run_from_image import DOMAINS, PY_D3D, ROOT, DATA_ROOT, EXP, CODE, run, find_nvml_preload
from run_prep import DATA_DIR          # domain-geometry root the prepared STLs live in

COND = CODE / 'conditioning'
# Two interpreters, because the two halves of this stage need different dependency sets:
#   PY_GEO — steps 1-2 (+5): pyvista + trimesh for the multi-view renders and occlusion masks,
#            numpy + PIL for the luminance normalization. This is the generation env.
#   SD_PY  — steps 3, 4, 6: the SDXL pipelines (diffusers, transformers, optional compel).
#            Defaults to PY_GEO, so a single env holding everything works unchanged; point it at
#            a dedicated SDXL env (the reported runs used one) when the versions differ.
PY_GEO = PY_D3D
SD_PY = os.environ.get('SD_PY', PY_GEO)
# Fallback only. The six views are a per-domain choice — cond_render_pv.py renders 26 of them
# and which six the chain uses is `views` in configs/<domain>.json. That key existed but
# nothing read it, so every domain silently got this bracket-derived set; on the caliper two
# of the six (v_top, v02_right_lo) show zero boundary-condition pixels and no through-holes,
# i.e. they carry no information about the part at all.
VIEWS_DEFAULT = 'v00_front_lo,v02_right_lo,v04_back_lo,v06_left_lo,v_top,v_bottom'
CURATED = json.load(open(ROOT / 'configs/curated_prompts.json'))


def _style_entry(dom, style):
    """The {'prompt', 'negative'} config entry for a (domain, style)."""
    try:
        return CURATED[dom]['styles'][style]
    except KeyError:
        # Geometry-only caliper variants share the same language/style bank.
        # Their rendered masks differ, so this does not reuse a conditioning image.
        if dom.startswith('caliper_'):
            try:
                return CURATED['caliper']['styles'][style]
            except KeyError:
                pass
        raise SystemExit(f'no prompt for {dom}/{style} in configs/curated_prompts.json')


def prompt_of(dom, style):
    """Positive prompt text (configs/curated_prompts.json: styles.<style>.prompt)."""
    return _style_entry(dom, style)['prompt']


def negative_of(dom, style):
    """Per-prompt negative text (configs/curated_prompts.json: styles.<style>.negative)."""
    return _style_entry(dom, style)['negative']


def descriptor_of(dom, style):
    """The bare form description. Goes into the ref prompt (step 3) and the img2img prompt (step 6);
    the inpaint step (4) uses the fuller `prompt` instead."""
    e = _style_entry(dom, style)
    return e.get('descriptor', e['prompt'])


def form_of(dom, style):
    """Bare form name for the img2img template ({form} openwork)."""
    return _style_entry(dom, style).get('form', style)


def render_cfg(c):
    """Per-domain render settings for steps 1-2 (`stages.cond_render`): the view list and the
    render resolution. Both were hardcoded here, so a domain could not choose either."""
    cfg = json.load(open(ROOT / c['config']))
    st = cfg.get('stages', {}).get('cond_render', {})
    return {'views': cfg.get('views', VIEWS_DEFAULT),
            'size': str(st.get('size', 1024)),
            'seed': int(cfg.get('seed', 42))}


def views_of(c):
    """The six view names the conditioning chain renders and stylizes, from the domain
    config (`views`), falling back to the shipped bracket set."""
    return json.load(open(ROOT / c['config'])).get('views', VIEWS_DEFAULT)


def _stage_cfg(cfg_path, stage):
    """One stages.<stage> block from a config, for text the orchestrator has to assemble."""
    return json.load(open(cfg_path)).get('stages', {}).get(stage, {})


def _env():
    # conditioning scripts import run_config from code/ (their sys.path only adds conditioning/),
    # so put code/ on PYTHONPATH.
    env = dict(os.environ, PYTHONPATH=str(CODE) + (os.pathsep + os.environ['PYTHONPATH']
                                                   if os.environ.get('PYTHONPATH') else ''))
    pl = find_nvml_preload()
    if pl:
        env['LD_PRELOAD'] = pl + (':' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else '')
    return env


def _step(label, cmd, log, env):
    """Run one conditioning step and stop the chain if it fails — a step that dies silently used to
    leave the previous artifact in place, which then looks like a bad result rather than a crash."""
    rc = run(cmd, log, env=env, cwd=str(DATA_ROOT))
    if rc != 0:
        raise SystemExit(f'[{label}] failed (rc={rc}) — see {log}')
    return rc


def render_domain(dom, wd, force, rcfg):
    """Steps 1-2: domain-only renders + occlusion masks. Geometry-only and deterministic, so they
    are cached per domain and NOT regenerated under --force (avoids a write race when several styles
    of the same domain build conditioning in parallel; --force still re-does the style-specific steps)."""
    # Domain geometry lives in the prepared domain root (see run_prep.py); the raw STLs are what
    # the shipped conditioning renders were produced from.
    data = DATA_ROOT / DATA_DIR / dom
    ds, fix, load = data / 'original_DesignSpace.stl', data / 'fixed.stl', data / 'load.stl'
    renders, masks = wd / 'renders', wd / 'masks'
    env = _env()
    vs = rcfg['views'].split(',')
    if not all((renders / f'{v}.png').exists() for v in vs):
        renders.mkdir(parents=True, exist_ok=True)
        _step('render', [PY_GEO, str(COND / 'cond_render_pv.py'), '--bracket', str(ds), '--fix', str(fix),
             '--load', str(load), '--out-dir', str(renders), '--size', rcfg['size']],
            wd / 'render.log', env)
    if not all((masks / f'{v}_bc.png').exists() for v in vs):
        masks.mkdir(parents=True, exist_ok=True)
        _step('masks', [PY_GEO, str(COND / 'cond_render_masks.py'), '--bracket', str(ds), '--fix', str(fix),
             '--load', str(load), '--out-dir', str(masks), '--size', rcfg['size']],
            wd / 'masks.log', env)
    return renders, masks


def gen_conditioning(dom, c, style, seed, force):
    wd = EXP / f'{dom}/cond'      # intermediate artifacts (renders, masks, ref, ...)
    sd = wd / style
    # Final 6-view conditioning lands in data/, exactly where run_from_image reads it.
    imgs = DATA_ROOT / f'data/{dom}/conditioning/{style}'
    rcfg = render_cfg(c)          # views / render size / seed (per-domain)
    # Check EVERY selected view, not just the first. A run killed mid-img2img leaves one view on
    # disk, and a first-view-only test then reports the conditioning as already done and skips it,
    # so the generator silently reads an incomplete set. The view names are per-domain now, so a
    # hardcoded name would also mis-fire whenever `views` differs from the shipped default.
    if all((imgs / f'{v}.png').exists() for v in rcfg['views'].split(',')) and not force:
        print(f'[{dom}/{style}] SKIP (conditioning exists) -> {imgs}  (use --force to regenerate)')
        return imgs
    sd.mkdir(parents=True, exist_ok=True)
    env = _env()
    config = str(ROOT / c['config'])
    prompt = prompt_of(dom, style)   # positive prompt text from configs/curated_prompts.json
    neg = negative_of(dom, style)    # per-style negative (reference-matched)

    renders, masks = render_domain(dom, wd, force, rcfg)

    # 3. SDXL text-to-image style reference, then normalized to a two-tone tile (see the script).
    #    Positive = the style's descriptor + stages.cond_ref.ref_common; negative = ref_negative.
    #    NOT the inpaint prompt: the reference has to read as "black structure / white holes".
    ref = sd / 'ref.png'
    rc = _stage_cfg(config, 'cond_ref')
    ref_prompt = descriptor_of(dom, style) + ', ' + rc.get('ref_common', '')
    _step('ref', [SD_PY, str(COND / 'text2img_style_ref.py'), '--config', config,
         '--prompt', ref_prompt, '--out', str(ref), '--seed', str(seed),
         '--raw-out', str(sd / 'ref_raw.png')],
        sd / 'ref.log', env)

    # 4. masked inpaint with external reference + cross-view attention
    remask = sd / 'remask'
    _step('inpaint', [SD_PY, str(COND / 'cond_stylize_inpaint_refip_xattn.py'),
         '--config', config, '--seed', str(seed), '--views', rcfg['views'],
         '--input-dir', str(renders), '--mask-dir', str(masks),
         '--out-dir', str(remask), '--prompt', prompt, '--negative', neg, '--ref-image', str(ref)],
        sd / 'inpaint.log', env)

    # 5. luminance normalization (remask -> white/): protect the boundary-condition regions
    #    (fill BC black = structure). Intermediate — step 6 consumes it.
    white = sd / 'white'
    nc = _stage_cfg(config, 'cond_normalize')
    _step('normalize', [PY_GEO, str(COND / 'nonblack_to_white.py'),
         '--in-dir', str(remask), '--out-dir', str(white),
         '--mode', nc.get('mode', 'keep'), '--protect-mask-dir', str(masks),
         '--protect-fill', nc.get('protect_fill', 'black')],
        sd / 'normalize.log', env)

    # 6. SDXL img2img consolidation (white/ -> final conditioning). Unifies tone across the six
    #    views and sharpens the carved openings. ip_scale 0 in every shipped config, so the
    #    reference image is inert here; strength also sets the step count (steps x strength).
    #    Its prompt is the recorded template, not the inpaint prompt:
    #      "metal bracket, {form} openwork, {descriptor}, monochrome metal, white background"
    ic = _stage_cfg(config, 'cond_img2img')
    i2i_prompt = ic.get('template', '{descriptor}').format(
        form=form_of(dom, style), descriptor=descriptor_of(dom, style))
    _step('img2img', [SD_PY, str(COND / 'cond_img2img_consolidate.py'), '--config', config,
         '--input-dir', str(white), '--out-dir', str(imgs), '--ref-image', str(ref),
         '--prompt', i2i_prompt, '--seed', str(seed)],
        sd / 'img2img.log', env)

    ok = all((imgs / f'{v}.png').exists() for v in rcfg['views'].split(','))
    print(f'[{dom}/{style}] {"OK" if ok else "FAIL"} -> {imgs}')
    return imgs if ok else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--domain', required=True, help='bracket|motor_mount|link')
    ap.add_argument('--style', required=True, help='style key (see configs/curated_prompts.json)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.domain not in DOMAINS:
        sys.exit(f'unknown domain {args.domain}; choose from {list(DOMAINS)}')
    os.chdir(DATA_ROOT)
    t0 = time.time()
    out = gen_conditioning(args.domain, DOMAINS[args.domain], args.style, args.seed, args.force)
    print(f'  ({time.time()-t0:.0f}s)')
    if out:
        print(f'\nConditioning ready: {out}\n'
              f'run_from_image picks it up automatically (it reads data/<domain>/conditioning/<style>/):\n'
              f'  python run_from_image.py --domain {args.domain} --style {args.style} --stage all')


if __name__ == '__main__':
    main()
