"""Fair, resumable constrained online RA-QD versus Sobol random search."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from analyze_raqd_descriptors import features_for_mesh
from qd_archive import PARAMETERS
from raqd_core import RealizationArchive
from raqd_posterior import STYLES, propose_batch, sobol_genomes

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'codebase'
PILOT = ROOT / 'experiments/bracket/qd_pilot_2026-09-13'
STUDY = ROOT / 'experiments/bracket/raqd_descriptor_study_2026-09-13/analysis.json'
DEFAULT_OUT = ROOT / 'experiments/bracket/raqd_online_2026-09-13'
VOLUME_CONTROL = {'vw': 20., 'vol_target': .5, 'sp_vw': 10., 'sp_vol_target': .5}
MICRO_BATCHES = (3, 1, 3, 3, 2, 3, 3, 2)
CHECKPOINTS = (12, 16, 24, 32)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = out / 'protocol.json'
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        for path, digest in protocol['hashes'].items():
            if sha(path) != digest:
                raise RuntimeError(f'Frozen input changed: {path}')
        return protocol
    shutil.copy2(PILOT / 'base_config.json', out / 'base_config.json')
    shutil.copy2(PILOT / 'descriptor_reference.npz', out / 'descriptor_reference.npz')
    for style in STYLES:
        target = out / 'image_bank' / style; target.mkdir(parents=True, exist_ok=True)
        for source in sorted((PILOT / 'image_bank' / style).glob('*.png')):
            shutil.copy2(source, target / source.name)
    selected = json.loads(STUDY.read_text())['selected_pair']
    frozen_files = [out / 'base_config.json', out / 'descriptor_reference.npz',
                    CODE / 'run_raqd_online.py', CODE / 'raqd_posterior.py',
                    CODE / 'raqd_core.py', CODE / 'analyze_raqd_descriptors.py',
                    CODE / 'run_conditioning_case.py', CODE / 'run_from_image.py',
                    CODE / 'evaluate_conditioning_case.py',
                    CODE / 'code/generate_with_physics_guidance.py']
    frozen_files += sorted((out / 'image_bank').glob('*/*.png'))
    protocol = {
        'created_at': time.time(), 'method': 'online realization-aware QD v1',
        'comparison': 'RA-QD versus pre-generated scrambled Sobol random search',
        'descriptor_names': selected['descriptors'], 'descriptor_ranges': selected['ranges'],
        'dims': [4, 4], 'objective': 'minimize final-mesh compliance_J',
        'constraint': 'valid final mesh and abs(material_volume_fraction-0.5)<=0.025',
        'volume_control': VOLUME_CONTROL, 'styles': list(STYLES),
        'parameters': PARAMETERS, 'shared_initial_evaluations': 12,
        'adaptive_evaluations_per_method': 20, 'budget_per_method': 32,
        'unique_full_pipeline_evaluations': 52, 'checkpoints_per_method': list(CHECKPOINTS),
        'micro_batches': list(MICRO_BATCHES), 'master_seed': 20260913,
        'posterior': {
            'model': 'Matheron-rule RFF approximate GP posterior',
            'kernel': 'equicorrelated categorical style times continuous RBF',
            'outputs': ['descriptor_0', 'descriptor_1', 'volume_fraction', 'log_compliance'],
            'noise': 'estimated from shared replicated seeds; seed excluded from covariates',
            'acquisition': 'constrained Thompson archive improvement with greedy maximin batch diversity',
            'candidate_pool': 8192, 'epsilon_audit_indices_zero_based': [9, 19]},
        'fairness': [
            'Both methods reuse and count the same 12 initial full evaluations.',
            'Four Sobol recipes each receive three distinct seeds to estimate stochastic noise.',
            'The 20 random proposals are frozen before any result is observed.',
            'RA-QD and random receive matched evaluation seeds at every post-initial position.',
            'Every attempted full evaluation counts, including failures and constraint violations.',
            'Only final-mesh measurements determine feasibility, cell membership, and quality.',
            'Descriptor ranges, volume tolerance, and checkpoints remain fixed.'
        ],
        'hashes': {str(path.resolve()): sha(path) for path in frozen_files}}
    save(protocol_path, protocol)
    return protocol


def initial_jobs():
    recipes = sobol_genomes(4, seed=20260913)
    return [{'id': f'shared_recipe{recipe:02d}_seed{repeat:02d}', 'method': 'shared',
             'round': 0, 'replicate_group': f'recipe_{recipe:02d}', 'seed': 45000+repeat,
             'target_cell': None, 'genome': genome, 'volume_control': VOLUME_CONTROL}
            for recipe, genome in enumerate(recipes) for repeat in range(3)]


def random_jobs():
    genomes = sobol_genomes(20, seed=20260914, skip=64)
    return [{'id': f'random_{i:02d}', 'method': 'random', 'round': i,
             'replicate_group': f'random_{i:02d}', 'seed': 46000+i,
             'target_cell': None, 'genome': genome, 'volume_control': VOLUME_CONTROL}
            for i, genome in enumerate(genomes)]


def evaluate(job, gpu, out, protocol):
    case = out / 'cases' / job['id']; case.mkdir(parents=True, exist_ok=True)
    result_path = case / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['genome'] != job['genome'] or result['seed'] != job['seed']:
            raise RuntimeError(f'Job changed on resume: {job["id"]}')
        return result
    config = json.loads((out / 'base_config.json').read_text())
    config['seed'] = job['seed']
    config['stages']['mesh'].update({name: job['genome'][name] for name in PARAMETERS})
    config['stages']['mesh'].update(VOLUME_CONTROL)
    save(case / 'config.json', config)
    conditioning = out / 'image_bank' / job['genome']['style']
    gen = case / 'gen'
    env = {**os.environ, 'DATA_ROOT': str(ROOT), 'D3DS2_ROOT': str(ROOT),
           'EXP_ROOT': str(ROOT / 'experiments'), 'D3DS2_PY': sys.executable,
           'FENICS_PY': '/home/goya/miniconda3/envs/fenics/bin/python',
           'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED': '1',
           'FEA_WORK_DIR': str(case / 'fea_work'), 'PYVISTA_OFF_SCREEN': 'true'}
    start = time.time()
    result = {**job, 'gpu': gpu, 'case_dir': str(case), 'started_at': start,
              'valid': False, 'constraints_satisfied': False, 'invalid_reasons': []}
    save(case / 'running.json', result)
    print(f'START {job["id"]} GPU={gpu} style={job["genome"]["style"]}', flush=True)
    try:
        with (case / 'run.log').open('a') as log:
            subprocess.run([sys.executable, str(CODE / 'run_conditioning_case.py'),
                            '--config', str(case / 'config.json'), '--conditioning', str(conditioning),
                            '--out', str(gen)], cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, str(CODE / 'evaluate_conditioning_case.py'),
                            '--mesh', str(gen / 'final.obj'), '--conditioning', str(conditioning),
                            '--out', str(gen / 'metrics'), '--domain-dir', str(ROOT / 'data_real/bracket'),
                            '--size', '384', '--pitch-mm', '1.0'], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        with np.load(out / 'descriptor_reference.npz') as reference:
            xyz = np.ascontiguousarray(reference['xyz'], dtype=np.float32)
            pitch = float(reference['pitch_m'])
        realization_path = {}
        for stage, filename in [('dense', 'mesh_dense.obj'), ('sparse', 'mesh.obj'), ('final', 'final.obj')]:
            path = gen / filename
            if path.exists():
                realization_path[stage] = features_for_mesh(path, xyz, pitch)
        features = realization_path['final']
        metrics = json.loads((gen / 'metrics/metrics.json').read_text())
        fea = json.loads((gen / 'fea/fea_tet_summary.json').read_text())
        containment = metrics['voxel']['containment_fraction_half_voxel_tolerance']
        if not metrics['watertight']: result['invalid_reasons'].append('not_watertight')
        if metrics['components'] != 1: result['invalid_reasons'].append('disconnected')
        if containment is None or containment < .99: result['invalid_reasons'].append('outside_domain')
        if not math.isfinite(fea['compliance']) or not 0 < fea['compliance'] < 1:
            result['invalid_reasons'].append('invalid_compliance')
        error = abs(features['material_volume_fraction'] - .5) / .5
        result.update(valid=not result['invalid_reasons'],
                      constraints_satisfied=not result['invalid_reasons'] and error <= .05,
                      realized_descriptors=[features[name] for name in protocol['descriptor_names']],
                      measured_design_volume_fraction=features['material_volume_fraction'],
                      relative_volume_error=error, compliance_J=fea['compliance'],
                      volume_mm3=metrics['volume_mm3'], features=features,
                      realization_path=realization_path, containment_fraction=containment,
                      mesh=str(gen / 'final.obj'), preview=str(gen / 'metrics/final_preview.png'))
    except Exception as exc:
        result['invalid_reasons'].append(f'{type(exc).__name__}: {exc}')
    result.update(seconds=time.time()-start, completed_at=time.time())
    save(result_path, result)
    print(f'DONE {job["id"]} valid={result["valid"]} feasible={result["constraints_satisfied"]} '
          f'C={result.get("compliance_J")} d={result.get("realized_descriptors")}', flush=True)
    return result


def build_method(method, shared, own, protocol):
    archive = RealizationArchive(protocol['dims'], protocol['descriptor_ranges'])
    attempts = []
    for result in [*shared, *own]:
        enriched, inserted = archive.record(result)
        attempts.append({**enriched, 'inserted': inserted})
    feasible = [a for a in attempts if a.get('constraints_satisfied')]
    return {'evaluations': len(attempts), 'post_initial_evaluations': len(own),
            'volume_feasible_rate': len(feasible)/len(attempts) if attempts else None,
            'archive': archive.summary(), 'attempts': attempts,
            'elites': [{'cell': list(cell), **row} for cell, row in sorted(archive.elites.items())]}


def render_report(out, summary):
    methods = summary.get('methods', {})
    rows = ''
    for name in ('raqd', 'random'):
        method = methods.get(name)
        if not method: continue
        archive = method['archive']
        rows += (f'<tr><td>{name}</td><td>{method["evaluations"]}</td>'
                 f'<td>{archive["verified_elites"]}/16 ({100*archive["verified_coverage"]:.1f}%)</td>'
                 f'<td>{100*method["volume_feasible_rate"]:.1f}%</td>'
                 f'<td>{archive["target_hit_rate"] if archive["target_hit_rate"] is not None else "–"}</td></tr>')
    recent = summary.get('results', [])[-12:]
    cards = ''
    for result in recent:
        preview = os.path.relpath(result['preview'], out) if result.get('preview') else ''
        image = f'<img src="{html.escape(preview)}">' if preview else ''
        cards += (f'<article>{image}<h3>{html.escape(result["id"])}</h3>'
                  f'<p>{html.escape(result["method"])} · feasible={result.get("constraints_satisfied")}<br>'
                  f'cell={result.get("realized_cell")} · C={1000*result.get("compliance_J", float("nan")):.3f} mJ</p></article>')
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Online RA-QD vs random</title><style>body{{font-family:system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 22px;color:#17212b;line-height:1.55}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd4da;padding:9px}}th{{background:#eef2f4}}.note{{background:#edf7f4;border-left:4px solid #187b65;padding:12px 16px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;margin-top:22px}}article{{border:1px solid #d4dce2;border-radius:9px;padding:10px}}img{{width:100%}}code{{overflow-wrap:anywhere}}</style><h1>Online RA-QD vs Sobol random</h1><p class="note">상태: <b>{html.escape(summary['phase'])}</b> · 완료된 고유 full pipeline 평가: <b>{summary.get('unique_completed',0)}/52</b>. 30초마다 자동 갱신됩니다.</p><table><tr><th>방법</th><th>논리 평가</th><th>검증 coverage</th><th>체적 제약 통과</th><th>목표 셀 적중률</th></tr>{rows}</table><div class="cards">{cards}</div><p>요약 절대경로: <code>{html.escape(str(out/'summary.json'))}</code></p><p>프로토콜 절대경로: <code>{html.escape(str(out/'protocol.json'))}</code></p></html>'''
    (out / 'report.html').write_text(page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--gpus', default='0,1,2')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = prepare(out)
    shared_plan, random_plan = initial_jobs(), random_jobs()
    if not (out / 'jobs_shared.json').exists(): save(out / 'jobs_shared.json', shared_plan)
    if not (out / 'jobs_random_frozen.json').exists(): save(out / 'jobs_random_frozen.json', random_plan)
    if args.prepare_only:
        render_report(out, {'phase': 'prepared', 'unique_completed': 0, 'results': []})
        print(out / 'protocol.json'); return
    gpus = [int(v) for v in args.gpus.split(',')]
    results = []

    def run_batch(jobs):
        queues = [jobs[i::len(gpus)] for i in range(len(gpus))]
        def worker(pair):
            gpu, queue = pair
            return [evaluate(job, gpu, out, protocol) for job in queue]
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            found = [row for group in pool.map(worker, zip(gpus, queues)) for row in group]
        by_id = {row['id']: row for row in found}
        return [by_id[job['id']] for job in jobs]

    shared, raqd, random_rows = [], [], []
    def publish(phase):
        methods = {}
        if shared:
            methods['raqd'] = build_method('raqd', shared, raqd, protocol)
            methods['random'] = build_method('random', shared, random_rows, protocol)
        payload = {'phase': phase, 'updated_at': time.time(), 'protocol': str(out/'protocol.json'),
                   'unique_completed': len(results), 'unique_gpu_pipeline_seconds': sum(r['seconds'] for r in results),
                   'methods': methods, 'results': results}
        save(out / 'summary.json', payload); render_report(out, payload)

    publish('shared_initial_running')
    shared = run_batch(shared_plan); results.extend(shared); publish('checkpoint_12')
    cursor = 0
    for batch_id, count in enumerate(MICRO_BATCHES):
        plan_path = out / f'jobs_raqd_batch_{batch_id:02d}.json'
        if plan_path.exists():
            adaptive = json.loads(plan_path.read_text())
        else:
            epsilon = tuple(i-cursor for i in (9, 19) if cursor <= i < cursor+count)
            proposed, diagnostics = propose_batch([*shared, *raqd], count, 20261000+batch_id,
                                                  protocol['descriptor_ranges'], protocol['dims'],
                                                  epsilon_indices=epsilon)
            adaptive = []
            for local, proposal in enumerate(proposed):
                index = cursor + local
                adaptive.append({'id': f'raqd_{index:02d}', 'method': 'raqd', 'round': batch_id+1,
                                 'replicate_group': f'raqd_{index:02d}', 'seed': 46000+index,
                                 'volume_control': VOLUME_CONTROL, **proposal})
            save(plan_path, adaptive); save(out / f'posterior_batch_{batch_id:02d}.json', diagnostics)
        paired = [item for pair in zip(adaptive, random_plan[cursor:cursor+count]) for item in pair]
        publish(f'online_batch_{batch_id:02d}_running')
        evaluated = run_batch(paired); results.extend(evaluated)
        raqd.extend([r for r in evaluated if r['method'] == 'raqd'])
        random_rows.extend([r for r in evaluated if r['method'] == 'random'])
        cursor += count
        budget = 12 + cursor
        publish(f'checkpoint_{budget}' if budget in CHECKPOINTS else f'online_batch_{batch_id:02d}_complete')
    publish('complete')
    print(out / 'report.html', flush=True)


if __name__ == '__main__':
    main()
