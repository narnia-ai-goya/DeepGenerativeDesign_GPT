#!/usr/bin/env python3
"""Convert gray GPT conditioning renders to geometry-preserving black silhouettes."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

SRC=Path('/home/goya/SDL/3d_qd/data/bracket/conditioning/gpt_bridge_arch_v2')
DST=Path('/home/goya/SDL/3d_qd/data/bracket/conditioning/gpt_bridge_arch_v2_black')
DST.mkdir(parents=True,exist_ok=True)
rows=[]
for src in sorted(SRC.glob('*.png')):
    rgb=np.asarray(Image.open(src).convert('RGB'))
    candidate=rgb.mean(axis=2)<245
    labels,n=ndi.label(candidate)
    sizes=np.bincount(labels.ravel()); sizes[0]=0
    component=int(sizes.argmax())
    hard=labels==component
    # Subpixel edge only; the component interior remains true RGB black.
    alpha=ndi.gaussian_filter(hard.astype(np.float32),sigma=0.45)
    alpha=np.clip((alpha-0.05)/0.90,0,1)
    out=np.rint(255*(1-alpha))[...,None].repeat(3,axis=2).astype(np.uint8)
    dst=DST/src.name; Image.fromarray(out).save(dst)
    yy,xx=np.where(hard)
    rows.append({'file':src.name,'source':str(src.resolve()),'output':str(dst.resolve()),
                 'threshold_rgb_mean_lt':245,'component_pixels':int(hard.sum()),
                 'bbox_xyxy':[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],
                 'core_black_fraction':float((out.max(2)<8).mean()),
                 'white_fraction':float((out.min(2)>247).mean())})
# Contact sheet
thumbs=[]
for row in rows:
    im=Image.open(row['output']).convert('RGB').resize((384,384),Image.Resampling.LANCZOS)
    tile=Image.new('RGB',(384,420),'white'); tile.paste(im,(0,0)); ImageDraw.Draw(tile).text((10,394),row['file'],fill='black'); thumbs.append(tile)
sheet=Image.new('RGB',(384*3,420*2),(235,235,235))
for i,im in enumerate(thumbs): sheet.paste(im,((i%3)*384,(i//3)*420))
sheet.save(DST/'contact_sheet.png')
manifest={'method':'largest connected component of mean RGB <245, black fill, 0.45 px antialias','source_dir':str(SRC.resolve()),'output_dir':str(DST.resolve()),'contact_sheet':str((DST/'contact_sheet.png').resolve()),'images':rows}
(DST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(DST.resolve()); print((DST/'contact_sheet.png').resolve()); print((DST/'manifest.json').resolve())
