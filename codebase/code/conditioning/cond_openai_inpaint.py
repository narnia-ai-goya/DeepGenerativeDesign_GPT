"""OpenAI Images API 로 하는 마스크 인페인팅 — SDXL 4단계의 대체.

SDXL 파이프라인에서 이미지 모델이 하는 일은 세 가지다:
  3. ref 타일 (text2img)   : 모든 배포 config 가 ip_scale=0 이라 최종 결과에 기여하지 않는다
                             (run_conditioning.py:209 주석). 따라서 대체 대상이 아니다.
  4. 마스크 인페인팅       : renders/ + masks/<view>_stylable.png + prompt/negative
                             -> images.edit(image, mask, prompt) 로 대응된다. 이 스크립트가 담당.
  6. img2img consolidate   : strength 0.82 로 6뷰 톤을 통일. Images API 에 strength 가 없어
                             그대로 옮길 수 없다 (--skip-stage6 참고).

Images API 에 없는 것과 그 처리:
  negative prompt : 없음. --negative 를 "Do not include: ..." 로 접어 positive 에 붙인다.
  strength        : 없음. 인페인팅은 마스크 영역을 전량 재생성한다.
  seed            : 없음. 재현성이 사라진다. 이 파이프라인은 같은 설정 재실행이 compliance 를
                    최대 93배 흔든 전례가 있다 (bionic_bone seed 42/7/13 = 178.845/99.047/1.923 mJ).
  cross-view attn : 없음. SDXL 쪽은 6뷰를 교차 어텐션으로 묶어 일관성을 유지하는데
                    (cond_stylize_inpaint_refip_xattn.py), 여기서는 뷰마다 독립 호출이라
                    뷰 간 무늬가 어긋날 수 있다. dense 단계가 6뷰를 함께 쓰므로 실제 위험이다.

마스크 규약: Images API 는 알파가 투명한 곳을 편집한다. 이 리포의 <view>_stylable.png 는
L 모드에 흰색(>128)이 스타일 적용 영역이므로, 흰색을 투명으로 바꾼 RGBA 를 만들어 보낸다.

usage:
  python cond_openai_inpaint.py --input-dir <renders/> --mask-dir <masks/> --out-dir <remask/> \
      --views v00_front_lo,... --prompt "..." [--negative "..."] \
      [--model gpt-image-1.5] [--mask-kind stylable] [--dry-run]
"""
import argparse
import base64
import io
import json
import pathlib
import sys
import time

import numpy as np
from PIL import Image


def load_cfg():
    p = pathlib.Path.home() / '.config/openai/config.json'
    if not p.exists():
        sys.exit(f'설정 없음: {p}')
    return json.load(open(p))


def build_prompt(positive: str, negative: str | None) -> str:
    """Images API 에 negative 가 없으므로 positive 에 접어 넣는다."""
    if not negative:
        return positive
    neg = ', '.join(t.strip() for t in negative.split(',') if t.strip())
    return f'{positive}. Do not include: {neg}.'


def to_edit_mask(mask_png: pathlib.Path, size: tuple[int, int]) -> bytes:
    """L 모드 마스크(흰색=스타일 영역) -> RGBA PNG (스타일 영역이 투명).

    Images API 는 알파=0 인 픽셀을 재생성한다.
    """
    m = np.array(Image.open(mask_png).convert('L').resize(size, Image.NEAREST))
    rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = np.where(m > 128, 0, 255)      # 흰색 -> 투명(편집 대상)
    buf = io.BytesIO()
    Image.fromarray(rgba, 'RGBA').save(buf, format='PNG')
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=True)
    ap.add_argument('--mask-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--views', required=True, help='쉼표로 구분된 뷰 이름')
    ap.add_argument('--prompt', required=True)
    ap.add_argument('--negative', default=None)
    ap.add_argument('--model', default=None, help='없으면 config 의 image.model')
    ap.add_argument('--size', default=None)
    ap.add_argument('--quality', default=None)
    ap.add_argument('--mask-kind', default='stylable',
                    help='masks/<view>_<kind>.png 에서 고를 마스크 (stylable | bracket)')
    ap.add_argument('--max-calls', type=int, default=None)
    ap.add_argument('--dry-run', action='store_true',
                    help='API 를 호출하지 않고 프롬프트 조립과 마스크 변환만 검증한다')
    a = ap.parse_args()

    cfg = load_cfg()
    ic = cfg.get('image', {})
    model = a.model or ic.get('model')
    size = a.size or ic.get('size', '1024x1024')
    quality = a.quality or ic.get('quality')
    budget = a.max_calls or cfg.get('budget', {}).get('max_calls', 10**9)

    ind, mkd, outd = (pathlib.Path(x) for x in (a.input_dir, a.mask_dir, a.out_dir))
    outd.mkdir(parents=True, exist_ok=True)
    views = [v.strip() for v in a.views.split(',') if v.strip()]
    prompt = build_prompt(a.prompt, a.negative)
    W, H = (int(x) for x in size.split('x'))

    print(f'model={model} size={size} quality={quality} mask={a.mask_kind} '
          f'views={len(views)} dry_run={a.dry_run}')
    print(f'prompt ({len(prompt)}자): {prompt[:200]}{"..." if len(prompt) > 200 else ""}')

    cl = None
    if not a.dry_run:
        from openai import OpenAI
        kw = {'api_key': cfg['api_key']}
        for k in ('base_url', 'organization', 'project'):
            if cfg.get(k):
                kw[k] = cfg[k]
        cl = OpenAI(**kw)

    calls = ok = 0
    for v in views:
        src = ind / f'{v}.png'
        msk = mkd / f'{v}_{a.mask_kind}.png'
        if not src.exists() or not msk.exists():
            print(f'  {v:22s} SKIP (입력 {src.exists()} / 마스크 {msk.exists()})')
            continue
        img = Image.open(src).convert('RGB').resize((W, H), Image.LANCZOS)
        ibuf = io.BytesIO(); img.save(ibuf, format='PNG')
        mbytes = to_edit_mask(msk, (W, H))
        edit_frac = (np.array(Image.open(io.BytesIO(mbytes)))[..., 3] == 0).mean() * 100

        if a.dry_run:
            print(f'  {v:22s} OK(dry)  입력 {img.size}  편집영역 {edit_frac:5.1f}%  '
                  f'마스크 {len(mbytes)/1024:.0f}KB')
            ok += 1
            continue

        if calls >= budget:
            print(f'  예산 상한 {budget} 도달 — 중단'); break
        kw = dict(model=model, image=('image.png', ibuf.getvalue(), 'image/png'),
                  mask=('mask.png', mbytes, 'image/png'), prompt=prompt, size=size, n=1)
        if quality:
            kw['quality'] = quality
        try:
            t = time.time(); r = cl.images.edit(**kw); calls += 1
            d = r.data[0]
            if getattr(d, 'b64_json', None):
                (outd / f'{v}.png').write_bytes(base64.b64decode(d.b64_json))
            elif getattr(d, 'url', None):
                import urllib.request
                urllib.request.urlretrieve(d.url, outd / f'{v}.png')
            else:
                print(f'  {v:22s} FAIL 응답에 이미지 없음'); continue
            ok += 1
            print(f'  {v:22s} OK {time.time()-t:5.1f}s  편집영역 {edit_frac:5.1f}%')
        except Exception as e:
            print(f'  {v:22s} FAIL {type(e).__name__}: {str(e)[:160]}')
            if cfg.get('budget', {}).get('stop_on_error', True):
                break

    print(f'\n{ok}/{len(views)} 완료  (API 호출 {calls}회)')
    sys.exit(0 if ok == len(views) else 1)


if __name__ == '__main__':
    main()
