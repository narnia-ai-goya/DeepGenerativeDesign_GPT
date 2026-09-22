"""Execute prepared RA-QD target probes and archive only verified final structures."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from analyze_raqd_descriptors import features_for_mesh
from qd_archive import PARAMETERS
from raqd_core import RealizationArchive

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'codebase'
PILOT = ROOT / 'experiments/bracket/qd_pilot_2026-09-13'


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def execute(job, gpu, out, descriptor_names):
    case = out / 'cases' / job['id']; case.mkdir(parents=True, exist_ok=True)
    result_path = case / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    config = json.loads((PILOT / 'base_config.json').read_text())
    config['seed'] = job['seed']
    config['stages']['mesh'].update({name: job['genome'][name] for name in PARAMETERS})
    config['stages']['mesh'].update(job['volume_control'])
    save(case / 'config.json', config)
    conditioning = PILOT / 'image_bank' / job['genome']['style']
    gen = case / 'gen'
    env = {**os.environ, 'DATA_ROOT': str(ROOT), 'D3DS2_ROOT': str(ROOT),
           'EXP_ROOT': str(ROOT / 'experiments'), 'D3DS2_PY': sys.executable,
           'FENICS_PY': '/home/goya/miniconda3/envs/fenics/bin/python',
           'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED': '1',
           'FEA_WORK_DIR': str(case / 'fea_work'), 'PYVISTA_OFF_SCREEN': 'true'}
    start = time.time()
    result = {**job, 'gpu': gpu, 'case_dir': str(case), 'valid': False,
              'constraints_satisfied': False, 'invalid_reasons': []}
    print(f'START {job["id"]} target={job["target_cell"]} GPU={gpu}', flush=True)
    try:
        with (case / 'run.log').open('a') as log:
            subprocess.run([sys.executable, str(CODE / 'run_conditioning_case.py'),
                            '--config', str(case / 'config.json'), '--conditioning', str(conditioning),
                            '--out', str(gen)], cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, str(CODE / 'evaluate_conditioning_case.py'),
                            '--mesh', str(gen / 'final.obj'), '--conditioning', str(conditioning),
                            '--out', str(gen / 'metrics'),
                            '--domain-dir', str(ROOT / 'data_real/bracket'),
                            '--size', '384', '--pitch-mm', '1.0'], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        with np.load(PILOT / 'descriptor_reference.npz') as reference:
            xyz = np.ascontiguousarray(reference['xyz'], dtype=np.float32)
            pitch = float(reference['pitch_m'])
        features = features_for_mesh(gen / 'final.obj', xyz, pitch)
        metrics = json.loads((gen / 'metrics/metrics.json').read_text())
        fea = json.loads((gen / 'fea/fea_tet_summary.json').read_text())
        containment = metrics['voxel']['containment_fraction_half_voxel_tolerance']
        if not metrics['watertight']: result['invalid_reasons'].append('not_watertight')
        if metrics['components'] != 1: result['invalid_reasons'].append('disconnected')
        if containment is None or containment < .99: result['invalid_reasons'].append('outside_domain')
        if not math.isfinite(fea['compliance']) or not 0 < fea['compliance'] < 1:
            result['invalid_reasons'].append('invalid_compliance')
        volume_error = abs(features['material_volume_fraction'] - .5) / .5
        result.update(valid=not result['invalid_reasons'],
                      constraints_satisfied=not result['invalid_reasons'] and volume_error <= .05,
                      realized_descriptors=[features[name] for name in descriptor_names],
                      measured_design_volume_fraction=features['material_volume_fraction'],
                      relative_volume_error=volume_error, compliance_J=fea['compliance'],
                      volume_mm3=metrics['volume_mm3'], features=features,
                      containment_fraction=containment,
                      mesh=str(gen / 'final.obj'), preview=str(gen / 'metrics/final_preview.png'))
    except Exception as exc:
        result['invalid_reasons'].append(f'{type(exc).__name__}: {exc}')
    result['seconds'] = time.time() - start
    save(result_path, result)
    print(f'DONE {job["id"]} valid={result["valid"]} constraints={result["constraints_satisfied"]} '
          f'd={result.get("realized_descriptors")}', flush=True)
    return result


def make_report(out, proposal, attempts, archive_summary):
    development = {tuple(x) for x in proposal['occupied_development_cells']}
    cards = []
    for result in attempts:
        hit = result['realized_cell'] == result['target_cell']
        new = tuple(result['realized_cell']) not in development if result['realized_cell'] else False
        preview = os.path.relpath(result['preview'], out) if result.get('preview') else ''
        image = f'<img src="{preview}" alt="{result["id"]}">' if preview else ''
        cards.append(f'''<article><h3>{result['id']}</h3>{image}<p>목표 셀: <b>{result['target_cell']}</b><br>예측 descriptor: {[round(x,4) for x in result['predicted_descriptors']]}<br>실제 descriptor: {[round(x,4) for x in result.get('realized_descriptors',[])]}<br>실제 셀: <b>{result['realized_cell']}</b><br>목표 적중: <b>{hit}</b><br>개발 자료 기준 신규 셀: <b>{new}</b><br>체적비: {result.get('measured_design_volume_fraction',float('nan')):.4f} · 체적 제약: {result['constraints_satisfied']}<br>Compliance: {1000*result.get('compliance_J',float('nan')):.4f} mJ</p><p class="path">{result.get('mesh','')}</p></article>''')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RA-QD v0 목표 셀 probe</title><style>body{{font-family:system-ui,sans-serif;max-width:1180px;margin:36px auto;padding:0 22px;color:#18212b;line-height:1.55}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:18px}}article{{border:1px solid #d5dce2;border-radius:9px;padding:14px}}img{{width:100%}}.path{{font-size:11px;overflow-wrap:anywhere;color:#58606b}}.note{{background:#eef7f4;border-left:4px solid #147d64;padding:12px 16px}}</style><h1>RA-QD v0 목표 셀 probe</h1><p class="note">Surrogate는 평가 순서만 결정했다. 목표 적중, coverage와 품질은 전체 파이프라인을 통과한 최종 mesh에서 계산했다.</p><p>Descriptor: <b>{proposal['descriptor_names'][0]}</b> × <b>{proposal['descriptor_names'][1]}</b><br>목표 적중률: <b>{archive_summary['target_hit_rate']}</b><br>검증 archive: <b>{archive_summary['verified_elites']}/{archive_summary['total_cells']}</b> 셀</p><div class="cards">{''.join(cards)}</div><p>절대경로: {out / 'summary.json'}</p></html>'''
    (out / 'report.html').write_text(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--proposal', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gpus', default='0,1,2')
    args = ap.parse_args()
    proposal = json.loads(args.proposal.resolve().read_text())
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    save(out / 'proposal.json', proposal)
    gpus = [int(x) for x in args.gpus.split(',')]
    jobs = proposal['jobs']
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(execute, job, gpus[i], out, proposal['descriptor_names'])
                   for i, job in enumerate(jobs)]
        results = [future.result() for future in futures]
    archive = RealizationArchive(proposal['dims'], proposal['descriptor_ranges'])
    attempts = []
    for result in results:
        observed, inserted = archive.record(result)
        attempts.append({**observed, 'inserted': inserted})
    archive_summary = archive.summary()
    summary = {'status': 'complete', 'proposal': str(args.proposal.resolve()),
               'descriptor_names': proposal['descriptor_names'],
               'descriptor_ranges': proposal['descriptor_ranges'],
               'archive': archive_summary, 'attempts': attempts,
               'elites': [{'cell': list(cell), **elite} for cell, elite in sorted(archive.elites.items())]}
    save(out / 'summary.json', summary)
    make_report(out, proposal, attempts, archive_summary)
    print(out / 'report.html')


if __name__ == '__main__':
    main()
