#!/usr/bin/env python3
"""Frozen-CLIP text-latent QD selector for image-conditioned chair design.

The external CLIP embedding is a prompt search descriptor, not the hidden
conditioning state of GPT image.  The bank and niche centers are fixed before
new images are observed.  Random and QD get equal one-image-per-round budgets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from make_chair_domain import ROOT

BASE=ROOT/'experiments/chair/sofa_style_2026-09-28'
OUT=BASE/'text_latent_qd_2026-10-03'
MODEL='openai/clip-vit-large-patch14'
BACKS={
    'round':'a solid gently rounded rectangular backrest panel with a continuous central load-bearing area',
    'faceted':'a solid subtly faceted backrest panel with a continuous central load-bearing area',
    'tapered':'a solid modestly tapered backrest panel, narrower at the top, with a continuous central load-bearing area',
    'flared':'a solid modestly flared backrest panel, wider at the top but within the original outer width, with a continuous central load-bearing area',
}
ARMS={
    'curved':'slender continuous curved armrests with visible open side gaps',
    'straight':'slender continuous straight armrests with visible open side gaps',
    'diagonal':'slender continuous diagonal armrests with visible open side gaps',
}
GAPS={
    'standard':'keep the original distance between the backrest bottom and seat',
    'raised':'raise the backrest bottom only slightly to enlarge the white gap above the seat, while keeping its load-bearing center solid',
}
COMMON=("Edit target: the supplied reference front-view black-metal chair rendering. "
        "Produce one single-chair square front-view studio image on pure white. "
        "Preserve the exact four floor-contact pads and leg positions, seat top height and width, "
        "camera, physical scale, black metal rendering, and overall outer footprint. "
        "Keep the central backrest solid where the seated person's back pushes. "
        "Only vary the upper backrest and arm geometry as specified. "
        "Do not add or remove legs, holes at the central back contact, or extra furniture. No text.")


def bank()->list[dict]:
    rows=[]
    for back,btext in BACKS.items():
        for arm,atext in ARMS.items():
            for gap,gtext in GAPS.items():
                identifier=f'{back}__{arm}__{gap}'
                shape=f'{btext}; {atext}; {gtext}.'
                rows.append({'id':identifier,'back':back,'arm':arm,'gap':gap,
                             'shape_text':shape,'full_prompt':COMMON+' Shape change: '+shape})
    return rows


def embed(texts:list[str])->np.ndarray:
    import torch
    from transformers import CLIPTextModelWithProjection,CLIPTokenizer
    tokenizer=CLIPTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=CLIPTextModelWithProjection.from_pretrained(MODEL,local_files_only=True).eval()
    outputs=[]
    with torch.inference_mode():
        for start in range(0,len(texts),16):
            tokens=tokenizer(texts[start:start+16],padding=True,truncation=True,
                             max_length=77,return_tensors='pt')
            vectors=model(**tokens).text_embeds.float().cpu().numpy()
            vectors/=np.linalg.norm(vectors,axis=1,keepdims=True).clip(1e-12)
            outputs.append(vectors)
    return np.vstack(outputs)


def prepare()->None:
    OUT.mkdir(parents=True,exist_ok=True)
    entries=bank()
    vectors=embed([r['shape_text'] for r in entries])
    mean=vectors.mean(0)
    _,_,vt=np.linalg.svd(vectors-mean,full_matrices=False)
    z=(vectors-mean)@vt[:2].T
    # Nine fixed farthest-point prototypes in full CLIP space.  PCA is only
    # for a human-readable plot; selection/cell assignment use all 768 dims.
    selected=[int(np.argmax(np.linalg.norm(vectors-vectors.mean(0),axis=1)))]
    while len(selected)<9:
        distances=np.min(((vectors[:,None,:]-vectors[selected][None,:,:])**2).sum(2),axis=1)
        selected.append(int(np.argmax(distances)))
    centers=vectors[selected]
    for i,row in enumerate(entries):
        row['text_cell']=int(np.argmin(((centers-vectors[i])**2).sum(1)))
        row['pca2']=z[i].tolist()
        row['embedding_index']=i
    np.save(OUT/'text_embeddings.npy',vectors)
    payload={'encoder':MODEL,'embedding':'frozen L2-normalized CLIP text embedding of shape-only clause',
             'niche_centers':'nine fixed farthest-point prototypes in full 768D embedding',
             'pca_for_display_only':True,'center_ids':[entries[i]['id'] for i in selected],
             'candidates':entries,'common_prompt':COMMON}
    (OUT/'bank.json').write_text(json.dumps(payload,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(9,7),layout='constrained')
    sc=ax.scatter(z[:,0],z[:,1],c=[r['text_cell'] for r in entries],cmap='tab10',s=90,
                  edgecolor='#24343c')
    for i,row in enumerate(entries):ax.annotate(row['id'],z[i],xytext=(4,4),textcoords='offset points',fontsize=7)
    ax.set_title('Frozen CLIP prompt space (PCA display only)')
    ax.set_xlabel('PCA 1');ax.set_ylabel('PCA 2');ax.grid(alpha=.2)
    fig.colorbar(sc,ax=ax,label='full-embedding CVT-like niche')
    fig.savefig(OUT/'text_latent_bank.png',dpi=160);plt.close(fig)
    print('bank',len(entries),'cells',len(set(r['text_cell'] for r in entries)))


def choose(round_index:int)->None:
    data=json.loads((OUT/'bank.json').read_text())
    entries=data['candidates'];vectors=np.load(OUT/'text_embeddings.npy')
    observed_path=OUT/'observations.json'
    observations=json.loads(observed_path.read_text()) if observed_path.exists() else []
    used={r['id'] for r in observations}
    used|={r['id'] for p in OUT.glob('selection_round_*.json') for r in json.loads(p.read_text())['selected']}
    count={i:0 for i in range(9)}
    for r in observations:
        if r['method']=='latent_qd':count[r['text_cell']]+=1
    candidates=[r for r in entries if r['id'] not in used]
    chosen=[]
    for row in candidates:
        i=row['embedding_index']
        if observations:
            prior=np.array([vectors[entries.index(next(x for x in entries if x['id']==o['id']))] for o in observations])
            novelty=float(np.min(1-prior@vectors[i]))
            same=[o for o in observations if o['text_cell']==row['text_cell']]
            # A measured failed 2D gate in this niche is a real penalty; no
            # untrained surrogate or claimed Bayesian posterior is hidden here.
            feasibility=float(np.mean([o['image_gate'] for o in same])) if same else .5
        else:
            novelty=1.
            feasibility=.5
        score=1.5/(1+count[row['text_cell']])+2*novelty+.4*feasibility
        chosen.append((score,row,novelty,feasibility))
    qd=max(chosen,key=lambda x:(x[0],x[1]['id']))
    pool=[r for r in candidates if r['id']!=qd[1]['id']]
    rng=np.random.default_rng(20261003+round_index)
    random_row=pool[int(rng.integers(len(pool)))]
    selected=[]
    for method,row in [('latent_qd',qd[1]),('random',random_row)]:
        selected.append({'method':method,**row})
    result={'round':round_index,'selection_rule':'empty/rare text niche + full-CLIP novelty + observed 2D feasibility; matched uniform random control',
            'selected':selected,'observations_available':len(observations)}
    path=OUT/f'selection_round_{round_index:02d}.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(path)
    for r in selected:print(r['method'],r['id'],'cell',r['text_cell'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','select'])
    parser.add_argument('--round',type=int,default=0)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:choose(args.round)
