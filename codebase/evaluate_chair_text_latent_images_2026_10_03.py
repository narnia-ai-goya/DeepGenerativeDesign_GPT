#!/usr/bin/env python3
"""Measure image phenotype, projected interfaces, and front envelope after each round."""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from build_chair_image_qd_2026_10_03 import normalized,metrics,image_cell
from chair_text_latent_qd_pilot_2026_10_03 import OUT,BASE


def projected_envelope()->list[int]:
    spec=json.loads((BASE/'single_view_spec_2026-10-03/specification.json').read_text())
    reference=spec['reference_mesh_bounds_m']
    envelope=spec['envelope_bounds_m']
    baseline=cv2.resize(cv2.imread(str(BASE/'open_arm/input/v00_front_lo.png')),(512,512))
    gray=cv2.cvtColor(baseline,cv2.COLOR_BGR2GRAY)
    yy,xx=np.where(gray<190)
    px_per_x=(xx.max()-xx.min())/(reference[1][0]-reference[0][0])
    px_per_z=(yy.max()-yy.min())/(reference[1][2]-reference[0][2])
    left=int(np.floor(xx.min()+(envelope[0][0]-reference[0][0])*px_per_x))
    right=int(np.ceil(xx.max()+(envelope[1][0]-reference[1][0])*px_per_x))
    top=int(np.floor(yy.min()-(envelope[1][2]-reference[1][2])*px_per_z))
    bottom=int(np.ceil(yy.max()+(reference[0][2]-envelope[0][2])*px_per_z))
    return [left,right,top,bottom]


def main(round_index:int)->None:
    selection=json.loads((OUT/f'selection_round_{round_index:02d}.json').read_text())
    old_path=OUT/'observations.json'
    old=json.loads(old_path.read_text()) if old_path.exists() else []
    prior_ids={r['id'] for r in old}
    box=projected_envelope()
    _,reference=normalized(BASE/'open_arm/input/v00_front_lo.png')
    rows=[]
    for selected in selection['selected']:
        name=selected['id']
        if name in prior_ids:raise ValueError(f'Already observed: {name}')
        source=OUT/'images'/f'{name}.png'
        norm,mask=normalized(source)
        cv2.imwrite(str(OUT/'images'/f'{name}_normalized.png'),norm)
        raw=cv2.resize(cv2.imread(str(source)),(512,512),interpolation=cv2.INTER_AREA)
        foreground=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)<190
        outside=foreground.copy();outside[box[2]:box[3]+1,box[0]:box[1]+1]=False
        outside_fraction=float(outside.sum()/foreground.sum())
        result={**selected,'round':round_index,'source_image':str(source),
                'normalized_image':str(OUT/'images'/f'{name}_normalized.png'),
                **metrics(mask,reference),
                'image_cell':image_cell(metrics(mask,reference)),
                'projected_envelope_box_px':box,
                'projected_envelope_outside_fraction':outside_fraction}
        result['image_gate']=bool(result['image_cell'] is not None and
                                  result['projected_interface_retention']>=.90 and
                                  result['projected_back_load_retention']>=.80 and
                                  outside_fraction<=.01)
        rows.append(result)
    all_rows=old+rows
    old_path.write_text(json.dumps(all_rows,indent=2)+'\n')
    (OUT/f'image_results_round_{round_index:02d}.json').write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:
        print(row['method'],row['id'],'text cell',row['text_cell'],
              'image cell',row['image_cell'],'image gate',row['image_gate'],
              'outside',round(row['projected_envelope_outside_fraction'],4))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--round',type=int,required=True)
    args=parser.parse_args();main(args.round)
