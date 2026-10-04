"""Build readable contact sheets from existing chair image/mesh experiments."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path('/home/goya/SDL/3d_qd/experiments/chair/sofa_style_2026-09-28/text_reasoned_front_axes_2026-10-03/long_qd_fea_2026-10-03')
OUT = ROOT / 'multi_example_gallery_2026-10-03'
CASES = ('qd_01', 'qd_02', 'qd_03', 'qd_04', 'random_01', 'random_02', 'random_03', 'random_04')
EXPANDED = ('qd_01', 'qd_02', 'qd_04', 'random_02', 'random_03')
BG = '#f1f4f6'
INK = '#172d3c'
MUTED = '#516474'


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype(f'/usr/share/fonts/truetype/dejavu/{name}', size)


def fit(im: Image.Image, w: int, h: int) -> Image.Image:
    im = im.convert('RGB')
    scale = min(w / im.width, h / im.height)
    return im.resize((round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS)


def paste_center(page: Image.Image, src: Path, box: tuple[int, int, int, int]) -> None:
    x, y, w, h = box
    im = fit(Image.open(src), w, h)
    page.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))


def source_mesh_pages() -> list[Path]:
    paths = []
    for page_idx in range(2):
        names = CASES[page_idx * 4:(page_idx + 1) * 4]
        page = Image.new('RGB', (1700, 1690), BG)
        d = ImageDraw.Draw(page)
        d.text((35, 28), f'Chair image -> 3D mesh | examples {page_idx * 4 + 1}-{page_idx * 4 + 4}', fill=INK, font=font(34, True))
        d.text((36, 78), 'Source images and existing generation outputs; same camera layout per row.', fill=MUTED, font=font(20))
        d.text((105, 125), 'Input image', fill=INK, font=font(22, True))
        d.text((700, 125), '3D mesh: front + side', fill=INK, font=font(22, True))
        for j, name in enumerate(names):
            y = 160 + j * 375
            d.rounded_rectangle((25, y, 1675, y + 355), radius=16, fill='white')
            d.text((42, y + 13), name, fill=INK, font=font(21, True))
            paste_center(page, ROOT / f'{name}.png', (60, y + 48, 335, 290))
            paste_center(page, ROOT / 'mesh_cases' / name / 'combined_preview.png', (470, y + 50, 1150, 280))
        dest = OUT / f'input_to_3d_examples_{page_idx + 1:02d}.png'
        page.save(dest)
        paths.append(dest)
    return paths


def envelope_page() -> Path:
    page = Image.new('RGB', (1700, 2190), BG)
    d = ImageDraw.Draw(page)
    d.text((35, 28), 'Envelope expansion | five image examples', fill=INK, font=font(34, True))
    d.text((36, 79), 'Same generated mesh clipped to each envelope. This is a geometric pilot, not a new generation run.', fill=MUTED, font=font(19))
    for j, name in enumerate(EXPANDED):
        y = 125 + j * 405
        d.rounded_rectangle((25, y, 1675, y + 385), radius=16, fill='white')
        d.text((45, y + 13), name, fill=INK, font=font(21, True))
        paste_center(page, ROOT / f'{name}.png', (42, y + 58, 270, 300))
        d.text((90, y + 352), 'Input image', fill=MUTED, font=font(18))
        base = ROOT / 'envelope_expansion_2026-10-03' / name
        for i, n in enumerate((0, 1, 2)):
            x = 345 + 445 * i
            d.text((x + 20, y + 26), ('Original', '+15.8 mm', '+31.6 mm')[i], fill=INK, font=font(20, True))
            paste_center(page, base / f'plus_{n}' / 'preview.png', (x, y + 75, 430, 270))
    dest = OUT / 'envelope_five_examples.png'
    page.save(dest)
    return dest


def html_page(paths: list[Path]) -> Path:
    items = ''.join(f'<figure><img src="{p.name}"><figcaption>{p.stem.replace("_", " ")}</figcaption></figure>' for p in paths)
    dest = OUT / 'index.html'
    dest.write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>Chair examples</title>'
                    '<style>body{font:17px/1.5 system-ui;background:#f1f4f6;color:#172d3c;max-width:1800px;margin:25px auto;padding:0 20px}'
                    'figure{margin:25px 0;background:white;padding:15px;border-radius:12px}img{max-width:100%;display:block;margin:auto}'
                    'figcaption{text-align:center;margin-top:12px}</style><h1>의자 입력 이미지와 3D 결과</h1>'
                    '<p>Envelope 비교는 기존 생성 메쉬를 확장된 허용 영역으로 다시 자른 기하학적 예비 실험입니다. 새 envelope으로 3D 생성을 다시 실행한 결과는 아닙니다.</p>'
                    + items + '</html>')
    return dest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = source_mesh_pages()
    paths.append(envelope_page())
    paths.append(html_page(paths))
    for path in paths:
        print(path)


if __name__ == '__main__':
    main()
