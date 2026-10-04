#!/usr/bin/env python3
"""Register complex-armchair orthographic images to the established chair BC frame."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw

from make_chair_domain import ROOT

CASE=ROOT/'experiments/chair/image_concepts_2026-09-27/complex_truss_armchair/multiview_registered'
REFERENCE=ROOT/'experiments/chair/aesthetic_reference_2026-09-27/input'
VIEWS=('v00_front_lo','v02_right_lo','v_top')


def bbox(image):
    pixels=np.asarray(image.convert('RGB'))
    mask=pixels.min(axis=2)<215
    yy,xx=np.where(mask)
    return int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)


def main():
    (CASE/'input').mkdir(parents=True,exist_ok=True)
    (CASE/'input_lr162').mkdir(exist_ok=True)
    sheet=Image.new('RGB',(512*3,548),'white')
    draw=ImageDraw.Draw(sheet)
    records={}
    for i,stem in enumerate(VIEWS):
        source=Image.open(CASE/'generated_orthographic'/f'{stem}.png').convert('RGB')
        target=Image.open(REFERENCE/f'{stem}.png').convert('RGB')
        source_box,target_box=bbox(source),bbox(target)
        cut=source.crop(source_box).resize((target_box[2]-target_box[0],
            target_box[3]-target_box[1]),Image.Resampling.LANCZOS)
        aligned=Image.new('RGB',(512,512),'white')
        aligned.paste(cut,target_box[:2])
        aligned.save(CASE/'input'/f'{stem}.png')
        aligned.resize((162,162),Image.Resampling.LANCZOS).save(CASE/'input_lr162'/f'{stem}.png')
        sheet.paste(aligned,(i*512,36))
        draw.text((i*512+12,10),stem,fill='#20252b')
        records[stem]={'source_bbox':source_box,'target_bbox':target_box,
            'registered_image':str(CASE/'input'/f'{stem}.png'),
            'lowres_image':str(CASE/'input_lr162'/f'{stem}.png')}
    sheet.save(CASE/'input_contact.png')
    (CASE/'registration.json').write_text(json.dumps(records,indent=2)+'\n')
    print(CASE/'input_contact.png')


if __name__=='__main__':main()
