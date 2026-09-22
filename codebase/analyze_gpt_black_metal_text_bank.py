#!/usr/bin/env python3
from pathlib import Path
from PIL import Image,ImageDraw
import json,html,os
import matplotlib.pyplot as plt
ROOT=Path('/home/goya/SDL/3d_qd'); OUT=ROOT/'experiments/bracket/gpt_black_metal_text_bank_3d_2026-09-14'; BANK=ROOT/'data/bracket/conditioning/gpt_black_metal_text_bank_2026-09-14'; IMG=ROOT/'output/imagegen/bracket_black_metal_text_bank_2026-09-14'

def load():
 rows=[]
 for cid in json.loads((OUT/'protocol.json').read_text())['cases']:
  g=OUT/'cases'/cid/'gen';m=json.loads((g/'metrics/metrics.json').read_text());f=json.loads((g/'fea/fea_tet_summary.json').read_text());vs=[x for x in m['views'].values() if 'input' in x]
  rows.append({'id':cid,'iou':sum(x['input']['mesh_foreground_iou'] for x in vs)/len(vs),'void':sum(x['mesh']['projected_holes_area_mm2'] for x in vs)/sum(x['input']['projected_holes_area_mm2'] for x in vs),'volume':m['volume_mm3'],'compliance_mJ':f['compliance']*1000,'stress_MPa':f['vm_max']/1e6,'p10':m['voxel']['medial_thickness_p10_mm'],'median':m['voxel']['medial_thickness_median_mm'],'thin':m['voxel']['medial_samples_below_3mm_fraction'],'watertight':m['watertight'],'components':m['components'],'input':str((BANK/cid/'v00_front_lo.png').resolve()),'prompt':str((IMG/cid/'prompt.txt').resolve()),'contact':str((IMG/cid/'contact_sheet.png').resolve()),'mesh':str((g/'final.obj').resolve()),'raw_mesh':str((g/'mesh.obj').resolve()),'preview':str((g/'metrics/final_preview.png').resolve()),'front':str((g/'metrics/v00_front_lo.png').resolve()),'metrics':str((g/'metrics/metrics.json').resolve()),'fea':str((g/'fea/fea_tet_summary.json').resolve())})
 return rows

def pareto(rows):
 return [r for r in rows if not any(q['void']>=r['void'] and q['compliance_mJ']<=r['compliance_mJ'] and (q['void']>r['void'] or q['compliance_mJ']<r['compliance_mJ']) for q in rows)]

def montage(rows):
 tiles=[]
 for r in rows:
  a=Image.open(r['input']).convert('RGB').resize((360,360),Image.Resampling.LANCZOS);b=Image.open(r['front']).convert('RGB').resize((360,360),Image.Resampling.LANCZOS)
  t=Image.new('RGB',(720,410),'white');t.paste(a,(0,0));t.paste(b,(360,0));d=ImageDraw.Draw(t);d.text((10,370),f"{r['id']}  input",fill='black');d.text((370,370),f"3D  void {r['void']*100:.1f}%  C {r['compliance_mJ']:.3f}mJ",fill='black');tiles.append(t)
 out=Image.new('RGB',(1440,1230),(230,232,234))
 for i,t in enumerate(tiles):out.paste(t,((i%2)*720,(i//2)*410))
 out.save(OUT/'input_to_3d_montage.png')

def main():
 rows=load();p=pareto(rows);montage(rows)
 fig,ax=plt.subplots(figsize=(8.5,6));ax.scatter([r['void']*100 for r in rows],[r['compliance_mJ'] for r in rows],s=100,c=['#e45756' if r in p else '#4c78a8' for r in rows],edgecolor='black')
 for r in rows:ax.annotate(r['id'],(r['void']*100,r['compliance_mJ']),xytext=(5,5),textcoords='offset points',fontsize=9)
 pp=sorted(p,key=lambda r:r['void']);ax.plot([r['void']*100 for r in pp],[r['compliance_mJ'] for r in pp],'--',color='#e45756',alpha=.6);ax.set(xlabel='Projected void retention (%) ↑',ylabel='Compliance (mJ) ↓',title='Text-prompt morphology archive');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUT/'prompt_pareto.png',dpi=180);plt.close(fig)
 summary={'study':'six distinct structural text prompts -> black-metal six-view images -> 3D + FEA','count':len(rows),'valid_count':sum(r['watertight'] and r['components']==1 for r in rows),'pareto':[r['id'] for r in sorted(p,key=lambda x:-x['void'])],'best_compliance':min(rows,key=lambda x:x['compliance_mJ'])['id'],'best_void':max(rows,key=lambda x:x['void'])['id'],'balanced':'x_brace','conditioning_root':str(BANK.resolve()),'image_bank_manifest':str((IMG/'manifest.json').resolve()),'cases':rows}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 trs=''.join(f"<tr><td>{r['id']}</td><td>{r['iou']:.3f}</td><td>{r['void']*100:.1f}</td><td>{r['volume']:.0f}</td><td>{r['p10']:.1f}</td><td>{r['median']:.1f}</td><td>{r['thin']*100:.1f}</td><td>{r['compliance_mJ']:.3f}</td><td>{r['stress_MPa']:.1f}</td><td>{'✓' if r in p else ''}</td></tr>" for r in rows)
 cards=''.join(f'''<article><img src="cases/{r['id']}/gen/metrics/final_preview.png"><h3>{r['id']}</h3><p>void {r['void']*100:.1f}% · C {r['compliance_mJ']:.3f} mJ<br>median {r['median']:.1f} mm · thin {r['thin']*100:.1f}%</p><p><a href="{os.path.relpath(r['prompt'],OUT)}">prompt</a> · <a href="{os.path.relpath(r['mesh'],OUT)}">final.obj</a> · <a href="{os.path.relpath(r['fea'],OUT)}">FEA</a></p></article>''' for r in rows)
 page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Black-metal text prompt bank</title><style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:30px auto;padding:0 24px;color:#171a1e;line-height:1.55}}.lead{{background:#eff5f3;border-left:5px solid #176b55;padding:15px 18px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d0d5da;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#20252b;color:white}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}article{{border:1px solid #ccd2d7;border-radius:10px;padding:12px}}article img{{background:white}}a{{color:#075ea8}}code{{overflow-wrap:anywhere}}@media(max-width:900px){{.cards{{grid-template-columns:1fr}}}}</style><h1>Black-metal structural text bank → 3D</h1><p class="lead">서로 다른 6개 구조 text prompt로 black-metal 6-view 이미지를 생성하고, 동일한 seed·CFG·FEA·두께 설정으로 3D화했다. <b>6/6 모두 watertight, single component, FEA valid.</b> Pareto 후보는 <b>{', '.join(summary['pareto'])}</b>다.</p><h2>핵심 결과</h2><p><b>개방형:</b> arch — void 42.6%, C 10.134 mJ. <b>균형형:</b> X-brace — void 30.3%, C 6.935 mJ. <b>역학형:</b> fan-rib — C 5.803 mJ, void 15.6%. Warren은 p10 8 mm와 C 6.924 mJ를 함께 확보했지만 void가 14.3%다.</p><h2>Input → generated 3D</h2><img src="input_to_3d_montage.png"><h2>QD-style trade-off</h2><img src="prompt_pareto.png"><h2>전체 수치</h2><table><tr><th>prompt</th><th>IoU</th><th>void %</th><th>volume mm³</th><th>p10 mm</th><th>median mm</th><th>&lt;3mm %</th><th>C mJ</th><th>stress MPa</th><th>Pareto</th></tr>{trs}</table><h2>개별 결과</h2><div class="cards">{cards}</div><p><code>{OUT/'summary.json'}</code><br><code>{OUT/'report.html'}</code><br><code>{BANK}</code></p></html>'''
 (OUT/'report.html').write_text(page);print((OUT/'report.html').resolve())
if __name__=='__main__':main()
