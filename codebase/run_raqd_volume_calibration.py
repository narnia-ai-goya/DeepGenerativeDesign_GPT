"""Run a three-way dense volume-control calibration on the bracket pipeline."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from analyze_raqd_descriptors import features_for_mesh

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'codebase'
PILOT = ROOT / 'experiments/bracket/qd_pilot_2026-09-13'
DEFAULT_OUT = ROOT / 'experiments/bracket/raqd_volume_calibration_2026-09-13'

VARIANT_SETS = {
    'mechanisms': [
        ('dense_soft_vw20', {'vw': 20.0, 'vol_target': .50}),
        ('dense_projection_w5', {'vol_projection': True, 'vol_proj_w': 5.0, 'vol_target': .50}),
        ('dense_aug_lag_mu50', {'aug_lag': True, 'lambda_init': 0.0,
                                'mu_aug_lag': 50.0, 'lambda_alpha': 1.0, 'vol_target': .50}),
    ],
    'soft_target_sweep': [
        ('dense_soft_vw20_target045', {'vw': 20.0, 'vol_target': .45}),
        ('dense_soft_vw20_target047', {'vw': 20.0, 'vol_target': .47}),
        ('dense_soft_vw20_target049', {'vw': 20.0, 'vol_target': .49}),
    ],
    'sparse_target_sweep': [
        ('sparse_soft_target030', {'vw': 20.0, 'vol_target': .50,
                                   'sp_vw': 10.0, 'sp_vol_target': .30}),
        ('sparse_soft_target040', {'vw': 20.0, 'vol_target': .50,
                                   'sp_vw': 10.0, 'sp_vol_target': .40}),
        ('sparse_soft_target050', {'vw': 20.0, 'vol_target': .50,
                                   'sp_vw': 10.0, 'sp_vol_target': .50}),
    ],
}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare_case(out, name, overrides):
    case = out / 'cases' / name
    case.mkdir(parents=True, exist_ok=True)
    config = json.loads((PILOT / 'base_config.json').read_text())
    config['seed'] = 42000
    config['stages']['mesh'].update({'cfg': 7., 'sp_cfg': 5., 'sp_guide_w_peak': 80., **overrides})
    save(case / 'config.json', config)
    return case


def run_one(out, name, overrides, gpu):
    case = prepare_case(out, name, overrides)
    result_path = case / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    gen = case / 'gen'
    env = {**os.environ, 'DATA_ROOT': str(ROOT), 'D3DS2_ROOT': str(ROOT),
           'EXP_ROOT': str(ROOT / 'experiments'), 'D3DS2_PY': sys.executable,
           'FENICS_PY': '/home/goya/miniconda3/envs/fenics/bin/python',
           'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED': '1',
           'FEA_WORK_DIR': str(case / 'fea_work'), 'PYVISTA_OFF_SCREEN': 'true'}
    start = time.time()
    result = {'id': name, 'gpu': gpu, 'seed': 42000, 'style': 'builtin_kagome_trial',
              'target_design_volume_fraction': .50, 'overrides': overrides,
              'case_dir': str(case), 'valid': False, 'error': None}
    print(f'START {name} GPU={gpu} {overrides}', flush=True)
    try:
        with (case / 'run.log').open('a') as log:
            subprocess.run([sys.executable, str(CODE / 'run_conditioning_case.py'),
                            '--config', str(case / 'config.json'),
                            '--conditioning', str(PILOT / 'image_bank/builtin_kagome_trial'),
                            '--out', str(gen)], cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, str(CODE / 'evaluate_conditioning_case.py'),
                            '--mesh', str(gen / 'final.obj'),
                            '--conditioning', str(PILOT / 'image_bank/builtin_kagome_trial'),
                            '--out', str(gen / 'metrics'),
                            '--domain-dir', str(ROOT / 'data_real/bracket'),
                            '--size', '384', '--pitch-mm', '1.0'], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        with __import__('numpy').load(PILOT / 'descriptor_reference.npz') as reference:
            xyz = __import__('numpy').ascontiguousarray(reference['xyz'], dtype=__import__('numpy').float32)
            pitch = float(reference['pitch_m'])
        features = features_for_mesh(gen / 'final.obj', xyz, pitch)
        metrics = json.loads((gen / 'metrics/metrics.json').read_text())
        fea = json.loads((gen / 'fea/fea_tet_summary.json').read_text())
        measured = features['material_volume_fraction']
        result.update(valid=bool(metrics['watertight'] and metrics['components'] == 1 and
                                 metrics['voxel']['containment_fraction_half_voxel_tolerance'] >= .99),
                      measured_design_volume_fraction=measured,
                      absolute_volume_error=abs(measured - .50),
                      relative_volume_error=abs(measured - .50) / .50,
                      compliance_J=fea['compliance'], volume_mm3=metrics['volume_mm3'],
                      features=features, mesh=str(gen / 'final.obj'),
                      preview=str(gen / 'metrics/final_preview.png'))
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['seconds'] = time.time() - start
    save(result_path, result)
    print(f'DONE {name} valid={result["valid"]} volume={result.get("measured_design_volume_fraction")}', flush=True)
    return result


def report(out, results):
    baseline_summary = json.loads((PILOT / 'summary.json').read_text())
    baseline = next(r for r in baseline_summary['results'] if r['id'] == 'initial_00')
    rows = [{'id': 'baseline_disabled', 'control': '없음',
             'measured_design_volume_fraction': baseline['descriptors'][0],
             'relative_volume_error': abs(baseline['descriptors'][0]-.5)/.5,
             'compliance_J': baseline['compliance_J'], 'valid': baseline['valid'],
             'seconds': baseline['seconds'], 'preview': baseline['preview'],
             'mesh': str(Path(baseline['case_dir']) / 'gen/final.obj')}, *results]
    valid_controls = [r for r in results if r['valid']]
    selected = min(valid_controls, key=lambda r: r['relative_volume_error']) if valid_controls else None
    summary = {'status': 'complete', 'target_design_volume_fraction': .5,
               'same_seed_style_and_generation_settings': True, 'baseline_reused': True,
               'results': rows, 'selected_control': selected['id'] if selected else None,
               'acceptance_relative_volume_error': .05,
               'selected_passes_acceptance': bool(selected and selected['relative_volume_error'] <= .05)}
    save(out / 'summary.json', summary)
    body = []
    for r in rows:
        preview = Path(r['preview'])
        try: src = os.path.relpath(preview, out)
        except ValueError: src = str(preview)
        body.append(f'''<article><h3>{r['id']}</h3><img src="{src}" alt="{r['id']}"><p>최종 설계 체적비: <b>{r['measured_design_volume_fraction']:.4f}</b><br>목표 대비 상대 오차: <b>{100*r['relative_volume_error']:.1f}%</b><br>Compliance: {1000*r['compliance_J']:.4f} mJ<br>유효: {r['valid']}</p><p class="path">{r['mesh']}</p></article>''')
    selected_text = selected['id'] if selected else '없음'
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RA-QD 체적 제어 보정</title><style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 22px;color:#18212b;line-height:1.55}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:18px}}article{{border:1px solid #d5dce2;border-radius:9px;padding:14px}}img{{width:100%;height:auto}}.path{{font-size:11px;overflow-wrap:anywhere;color:#58606b}}.note{{background:#fff7df;border-left:4px solid #d29a16;padding:12px 16px}}</style><h1>RA-QD 체적 제어 보정</h1><p class="note">동일한 이미지, seed와 생성 설정에서 목표 설계 체적비 0.50을 시험했다. 기준 결과는 재사용했으며 세 제어 방식만 새로 실행했다.</p><p>목표에 가장 가까운 제어: <b>{selected_text}</b>. 5% 상대 오차 통과: <b>{summary['selected_passes_acceptance']}</b>.</p><div class="cards">{''.join(body)}</div><p>절대경로: {out / 'summary.json'}</p></html>'''
    (out / 'report.html').write_text(page)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--gpus', default='0,1,2')
    ap.add_argument('--variant-set', choices=VARIANT_SETS, default='mechanisms')
    args = ap.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    gpus = [int(x) for x in args.gpus.split(',')]
    variants = VARIANT_SETS[args.variant_set]
    if len(gpus) < len(variants):
        raise ValueError('Three GPUs are required for the concurrent calibration')
    save(out / 'protocol.json', {'purpose': 'RA-QD volume-control calibration',
         'source_baseline': str(PILOT), 'target_design_volume_fraction': .5,
         'seed': 42000, 'style': 'builtin_kagome_trial',
         'variant_set': args.variant_set,
         'variants': [{'id': n, 'overrides': v, 'gpu': gpus[i]} for i, (n, v) in enumerate(variants)]})
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_one, out, name, overrides, gpus[i])
                   for i, (name, overrides) in enumerate(variants)]
        results = [future.result() for future in futures]
    report(out, results)
    print(out / 'report.html')


if __name__ == '__main__':
    main()
