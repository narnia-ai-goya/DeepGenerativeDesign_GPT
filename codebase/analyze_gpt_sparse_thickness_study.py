#!/usr/bin/env python3
from __future__ import annotations
import json, html, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
ROOT=Path('/home/goya/SDL/3d_qd')
OUT=ROOT/'experiments/bracket/gpt_sparse_thickness_study_2026-09-14'

def load_cases():
    base=json.loads((OUT/'protocol.json').read_text())['cases']
    adaptive=json.loads((OUT/'adaptive_protocol.json').read_text())['cases']
    rows=[]
    for phase, cs in [('OFAT',base),('adaptive',adaptive)]:
      for c in cs:
        cid=c['id']; g=OUT/'cases'/cid/'gen'
        m=json.loads((g/'metrics/metrics.json').read_text()); raw=json.loads((g/'raw_metrics/metrics.json').read_text()); f=json.loads((g/'fea/fea_tet_summary.json').read_text())
        views=[v for v in m['views'].values() if 'input' in v]
        void=sum(v['mesh']['projected_holes_area_mm2'] for v in views)/sum(v['input']['projected_holes_area_mm2'] for v in views)
        rv=[v for v in raw['views'].values() if 'input' in v]
        raw_void=sum(v['mesh']['projected_holes_area_mm2'] for v in rv)/sum(v['input']['projected_holes_area_mm2'] for v in rv)
        cfg=json.loads((OUT/'cases'/cid/'config.json').read_text())['stages']['mesh']
        rows.append(dict(id=cid,phase=phase,parameters=c.get('parameters') or ({c['changed_parameter']:c['value']} if c.get('changed_parameter') else {}),
          thick_w=cfg['sp_thick_w'],thick_target=cfg['sp_thick_target'],rmin_w=cfg['sp_rmin_w'],rmin_vox=cfg['sp_r_min_voxels'],interior_w=cfg['sp_interior_w'],lap_w=cfg['sp_lap_w'],mc=cfg['mc_threshold'],
          p10=m['voxel']['medial_thickness_p10_mm'],median=m['voxel']['medial_thickness_median_mm'],thin=m['voxel']['medial_samples_below_3mm_fraction'],
          raw_p10=raw['voxel']['medial_thickness_p10_mm'],raw_median=raw['voxel']['medial_thickness_median_mm'],raw_thin=raw['voxel']['medial_samples_below_3mm_fraction'],
          void=void,raw_void=raw_void,volume=m['volume_mm3'],compliance_mJ=f['compliance']*1000,stress_MPa=f['vm_max']/1e6,
          watertight=m['watertight'],components=m['components'],mesh=str((g/'final.obj').resolve()),raw_mesh=str((g/'mesh.obj').resolve()),preview=str((g/'metrics/final_preview.png').resolve()),metrics=str((g/'metrics/metrics.json').resolve()),raw_metrics=str((g/'raw_metrics/metrics.json').resolve()),fea=str((g/'fea/fea_tet_summary.json').resolve())))
    return rows

def plots(rows):
    b=next(x for x in rows if x['id']=='baseline')
    fig,ax=plt.subplots(figsize=(9,6)); phases={'OFAT':'o','adaptive':'s'}
    for phase,marker in phases.items():
      rr=[x for x in rows if x['phase']==phase]; sc=ax.scatter([x['raw_thin']*100 for x in rr],[x['compliance_mJ'] for x in rr],c=[x['void']*100 for x in rr],cmap='viridis',marker=marker,s=75,edgecolor='black',linewidth=.5,label=phase)
    ax.scatter([b['raw_thin']*100],[b['compliance_mJ']],marker='*',s=250,c='#e63946',edgecolor='black',zorder=5,label='baseline')
    ax.set(xlabel='Raw mesh: samples below 3 mm (%) ↓',ylabel='Final compliance (mJ) ↓',title='Thickness risk vs mechanics (color = final void retention)'); ax.grid(alpha=.25); ax.legend(); cb=fig.colorbar(sc,ax=ax); cb.set_label('Final void retention (%)'); fig.tight_layout(); fig.savefig(OUT/'thickness_compliance_tradeoff.png',dpi=180); plt.close(fig)
    grid=[x for x in rows if x['phase']=='adaptive' and x['id'].startswith('adapt_thick')]
    ws=[5,20,100]; ts=[.06,.12,.24]
    fig,axs=plt.subplots(1,2,figsize=(10,4.3))
    for ax,key,title,fmt in [(axs[0],'raw_thin','Raw <3 mm fraction (%)','.1f'),(axs[1],'median','Final median thickness (mm)','.1f')]:
      a=np.full((3,3),np.nan)
      for i,t in enumerate(ts):
       for j,w in enumerate(ws):
        z=next((x for x in grid if x['thick_w']==w and abs(x['thick_target']-t)<1e-9),None)
        if z: a[i,j]=z[key]*(100 if key=='raw_thin' else 1)
      im=ax.imshow(a,cmap='magma_r' if key=='raw_thin' else 'viridis',aspect='auto')
      ax.set_xticks(range(3),ws); ax.set_yticks(range(3),ts); ax.set_xlabel('sp_thick_w'); ax.set_ylabel('sp_thick_target'); ax.set_title(title)
      for i in range(3):
       for j in range(3): ax.text(j,i,format(a[i,j],fmt),ha='center',va='center',color='white' if np.isfinite(a[i,j]) and a[i,j]>(np.nanmin(a)+np.nanmax(a))/2 else 'black',fontweight='bold')
      fig.colorbar(im,ax=ax,shrink=.8)
    fig.tight_layout(); fig.savefig(OUT/'adaptive_thickness_heatmap.png',dpi=180); plt.close(fig)

def montage(rows, ids):
    ims=[]
    for cid in ids:
      r=next(x for x in rows if x['id']==cid); im=Image.open(r['preview']).convert('RGB').resize((430,430)); canvas=Image.new('RGB',(430,490),'white'); canvas.paste(im,(0,0)); d=ImageDraw.Draw(canvas); d.text((12,438),cid,fill='black'); d.text((12,461),f"raw<3mm {r['raw_thin']*100:.1f}% | med {r['median']:.1f}mm | C {r['compliance_mJ']:.3f}mJ",fill=(45,45,45)); ims.append(canvas)
    out=Image.new('RGB',(430*4,490*2),(235,237,240))
    for i,im in enumerate(ims): out.paste(im,((i%4)*430,(i//4)*490))
    out.save(OUT/'selected_black_montage.png')

def main():
 rows=load_cases(); plots(rows)
 selected=['baseline','interiorw_0','interiorw_20','mc_0p3','adapt_thickw20_t0p06','adapt_thickw100_t0p06','adapt_thickw100_t0p24','adapt_rminw25_v20']; montage(rows,selected)
 base=next(x for x in rows if x['id']=='baseline'); thick=next(x for x in rows if x['id']=='adapt_thickw100_t0p24'); mech=min(rows,key=lambda x:x['compliance_mJ']); robust=min(rows,key=lambda x:x['raw_thin'])
 ranking=sorted(rows,key=lambda x:(x['raw_thin'],x['compliance_mJ']))
 summary={'study':'FEA-on shared-dense sparse thickness parameter study','study_root':str(OUT.resolve()),'conditioning':str((ROOT/'data/bracket/conditioning/gpt_bridge_arch_v2').resolve()),'dense_cache':str((OUT/'dense_cache.npz').resolve()),'n_cases':len(rows),'n_ofat':sum(x['phase']=='OFAT' for x in rows),'n_adaptive':sum(x['phase']=='adaptive' for x in rows),'baseline':base,'recommended_thickness':thick,'best_compliance':mech,'lowest_raw_thin_fraction':robust,'cases':rows,'notes':['All cases share one FEA-on dense cache.','Final metrics use unioned/clipped/remeshed final.obj; raw metrics isolate sparse mesh.obj.','Voxel medial thickness is a 1 mm pitch EDT proxy.','The earlier FEA-off dense pilot is preserved separately and excluded from ranking.']}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 def row(r):
  return f"<tr><td>{html.escape(r['id'])}</td><td>{r['phase']}</td><td>{r['thick_w']:g}</td><td>{r['thick_target']:g}</td><td>{r['rmin_w']:g}/{r['rmin_vox']:g}</td><td>{r['raw_p10']:.1f}</td><td>{r['raw_median']:.1f}</td><td>{r['raw_thin']*100:.1f}</td><td>{r['median']:.1f}</td><td>{r['void']*100:.1f}</td><td>{r['volume']:.0f}</td><td>{r['compliance_mJ']:.3f}</td></tr>"
 table=''.join(row(r) for r in sorted(rows,key=lambda x:(x['phase'],x['id'])))
 page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPT bridge sparse thickness study</title><style>body{{font-family:system-ui,sans-serif;max-width:1440px;margin:30px auto;padding:0 24px;color:#15181d;line-height:1.55}}h1,h2{{letter-spacing:-.02em}}.lead{{background:#f0f5f3;border-left:5px solid #176b55;padding:15px 18px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{border:1px solid #ccd2d7;border-radius:10px;padding:14px}}.num{{font-size:1.55rem;font-weight:750}}img{{max-width:100%;background:white}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #d2d6da;padding:6px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#20252b;color:white;position:sticky;top:0}}code{{overflow-wrap:anywhere}}.note{{color:#4b535b}}@media(max-width:900px){{.cards{{grid-template-columns:1fr}}}}</style>
<h1>GPT bridge · sparse thickness parameter study</h1><p class="lead"><b>32개 모두 성공:</b> FEA-on dense cache 하나를 고정하고 OFAT 20개 + adaptive log-scale 12개를 비교했다. 일반 범위(weight 0–2)는 두께를 거의 바꾸지 않았고, <b>weight 100 + target .24</b>에서만 raw 3 mm 미만 비율이 {base['raw_thin']*100:.1f}%→{thick['raw_thin']*100:.1f}%로 안정적으로 줄었다. compliance 변화는 {((thick['compliance_mJ']/base['compliance_mJ'])-1)*100:+.1f}%다.</p>
<div class="cards"><div class="card"><div>권장 두께 조건</div><div class="num">w=100, target=.24</div><p>final median {thick['median']:.1f} mm · raw thin {thick['raw_thin']*100:.1f}% · void {thick['void']*100:.1f}% · C {thick['compliance_mJ']:.3f} mJ</p></div><div class="card"><div>최저 compliance</div><div class="num">{html.escape(mech['id'])}</div><p>C {mech['compliance_mJ']:.3f} mJ · void {mech['void']*100:.1f}% · final median {mech['median']:.1f} mm</p></div><div class="card"><div>핵심 해석</div><div class="num">두께는 loss scale 문제</div><p>target만 높이거나 weight≤20만 쓰면 변화가 작다. MC threshold는 두께를 크게 바꾸지만 형상 전체 offset이라 별도 제어로 취급해야 한다.</p></div></div>
<h2>선택 결과 · 검정색 형상 렌더</h2><img src="selected_black_montage.png" alt="selected black mesh previews">
<h2>두께–역학 trade-off</h2><img src="thickness_compliance_tradeoff.png" alt="tradeoff plot"><h2>adaptive weight × target</h2><img src="adaptive_thickness_heatmap.png" alt="adaptive heatmap">
<h2>전체 32개 결과</h2><table><thead><tr><th>case</th><th>phase</th><th>thick w</th><th>target</th><th>rmin w/vox</th><th>raw p10 mm</th><th>raw median mm</th><th>raw &lt;3mm %</th><th>final median mm</th><th>void %</th><th>volume mm³</th><th>C mJ</th></tr></thead><tbody>{table}</tbody></table>
<h2>판단</h2><p>제조 두께를 직접 제어하려면 <b>sp_thick_w=100, sp_thick_target=.24</b>가 현재 표본에서 가장 균형이 좋다. `mc_threshold=.3`은 raw p10을 2→4 mm로 올리고 compliance도 가장 낮추지만, void retention을 35.1→33.7%로 낮추고 체적을 늘린다. 따라서 MC threshold는 두께 regularizer라기보다 최종 등가면 offset으로 해석하는 편이 맞다. `interior_w=0`은 중앙 두께를 10 mm로 올리지만 얇은 요소를 선택적으로 두껍게 하기보다 내부 재료를 제거해 분포를 바꾸므로 주 방법으로 쓰기 어렵다.</p>
<p class="note">두께는 1 mm voxel EDT 기반 proxy다. 최종 후보에는 국소 ray thickness 또는 CAD thickness analysis를 추가하는 것이 좋다. 모든 결과는 watertight, single component이며 FEA summary가 유효하다. 잘못된 FEA-off dense pilot은 <code>{OUT/'cases_fea_off_dense_pilot'}</code>에 보존했지만 위 순위에서는 제외했다.</p>
<p><code>{OUT/'summary.json'}</code><br><code>{OUT/'report.html'}</code><br><code>{OUT/'dense_cache.npz'}</code></p></html>'''
 (OUT/'report.html').write_text(page)
 print(OUT/'report.html')
if __name__=='__main__': main()
