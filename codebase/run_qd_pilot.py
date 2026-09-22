"""Resumable bracket MAP-Elites vs random pilot with equal evaluation budgets.

Three shared seeds, then two feedback rounds of three offspring per method.
Failed physical evaluations count toward the budget. No adaptive retuning of bins.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from qd_archive import GridArchive, PARAMETERS, descriptors_from_occupancy, propose

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'codebase'
DEFAULT_OUT = ROOT / 'experiments/bracket/qd_pilot_2026-09-13'
BANK = ['builtin_kagome_trial', 'bionic_bone', 'de_trilattice']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / 'protocol.json'
    if frozen.exists():
        protocol = json.loads(frozen.read_text())
        for path, digest in protocol['hashes'].items():
            if sha(path) != digest:
                raise RuntimeError(f'Frozen input changed: {path}')
        return protocol
    baseline = ROOT / 'experiments/bracket/builtin_kagome_trial/config.json'
    shutil.copy2(baseline, out / 'base_config.json')
    hashes = {str(out / 'base_config.json'): sha(out / 'base_config.json')}
    for style in BANK:
        source = ROOT / 'data/bracket/conditioning' / style
        images = sorted(source.glob('*.png'))
        if len(images) != 6:
            raise ValueError(f'Expected six images: {source}')
        target = out / 'image_bank' / style
        target.mkdir(parents=True, exist_ok=True)
        for src in images:
            dst = target / src.name
            shutil.copy2(src, dst)
            hashes[str(dst)] = sha(dst)
    for src in [CODE / 'qd_archive.py', CODE / 'run_qd_pilot.py',
                CODE / 'run_conditioning_case.py', CODE / 'run_from_image.py',
                CODE / 'evaluate_conditioning_case.py',
                CODE / 'code/generate_with_physics_guidance.py']:
        hashes[str(src)] = sha(src)
    for name in ['original_DesignSpace.stl', 'original_DesignSpace_remesh.stl',
                 'fixed.stl', 'fixed_remesh.stl', 'load.stl', 'load_remesh.stl', 'voxel.npz']:
        src = ROOT / 'data_real/bracket' / name
        hashes[str(src)] = sha(src)
    protocol = {'created_at': time.time(), 'bank': BANK, 'parameters': PARAMETERS,
                'dims': [4, 4], 'ranges': [[0, 1], [0, 1]], 'pitch_mm': 1.,
                'descriptors': ['design_volume_fraction', 'design_upper_z_material_share'],
                'design_region': 'original_DesignSpace minus fixed and load solids',
                'upper_z': 'CAD envelope midpoint; fixed physical coordinate frame',
                'objective': 'minimize final-mesh FEA compliance_J',
                'quality_for_qd_score': '1/(1+compliance_J/0.01)',
                'budget_per_method': 9, 'shared_initial': 3, 'rounds': 2,
                'offspring_per_round_per_method': 3, 'unique_jobs': 15,
                'random_seed': 20260913, 'validity': {
                    'watertight': True, 'components': 1,
                    'min_containment_fraction_half_voxel_tolerance': .99,
                    'compliance_J': [0, 1]},
                'notes': ['Existing image bank is reused; builtin Kagome and legacy style sets have different provenance.',
                          'No image generation or paid API call occurs during this search.',
                          'Same three evaluated initial candidates are counted in each method budget.',
                          'Failures count as evaluations. No automatic replacement of failed designs.',
                          'Fixed seeds do not guarantee deterministic CUDA geometry.',
                          'Single small run is not a statistical comparison or manufacturing certification.',
                          'Archive ranges are fixed before evaluations and never clipped or retuned.',
                          'vol_target is not searched: vw=0 and aug_lag=false in this baseline.'],
                'hashes': hashes}
    save(frozen, protocol)
    return protocol


def reference_samples(out, protocol):
    import numpy as np
    import trimesh
    from pysdf import SDF
    path = out / 'descriptor_reference.npz'
    if path.exists():
        return
    meshes = [trimesh.load(ROOT / 'data_real/bracket' / n, force='mesh')
              for n in ['original_DesignSpace.stl', 'fixed.stl', 'load.stl']]
    lo, hi = meshes[0].bounds
    pitch = protocol['pitch_mm'] / 1000
    axes = [np.arange(a + pitch/2, b, pitch) for a, b in zip(lo, hi)]
    xyz = np.ascontiguousarray(np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3), dtype=np.float32)
    masks = []
    for mesh in meshes:
        field = SDF(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
        masks.append(field(xyz, n_threads=4) > 0)
    xyz = xyz[masks[0] & ~masks[1] & ~masks[2]]
    np.savez_compressed(path, xyz=xyz, upper=xyz[:, 2] >= (lo[2] + hi[2])/2,
                        pitch_m=pitch, envelope_bounds=meshes[0].bounds)
    save(out / 'descriptor_reference.json', {'path': str(path), 'sha256': sha(path),
         'samples': len(xyz), 'pitch_mm': protocol['pitch_mm'],
         'design_reference_volume_mm3': len(xyz) * protocol['pitch_mm']**3})


def evaluate(job, gpu, out, protocol):
    import numpy as np
    import trimesh
    from pysdf import SDF
    case = out / 'cases' / job['id']
    case.mkdir(parents=True, exist_ok=True)
    result_path = case / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result['genome'] != job['genome'] or result['seed'] != job['seed']:
            raise RuntimeError('Job changed on resume')
        return result
    cfg = json.loads((out / 'base_config.json').read_text())
    cfg['seed'] = job['seed']
    cfg['stages']['mesh'].update({k: job['genome'][k] for k in PARAMETERS})
    save(case / 'config.json', cfg)
    cond = out / 'image_bank' / job['genome']['style']
    gen = case / 'gen'
    env = {**os.environ, 'DATA_ROOT': str(ROOT), 'D3DS2_ROOT': str(ROOT),
           'EXP_ROOT': str(ROOT / 'experiments'), 'D3DS2_PY': sys.executable,
           'FENICS_PY': '/home/goya/miniconda3/envs/fenics/bin/python',
           'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED': '1',
           'FEA_WORK_DIR': str(case / 'fea_work'), 'PYVISTA_OFF_SCREEN': 'true'}
    start = time.time()
    result = {**job, 'case_dir': str(case), 'gpu': gpu, 'started_at': start,
              'valid': False, 'invalid_reasons': []}
    save(case / 'running.json', result)
    print(f'START {job["id"]} GPU={gpu} {job["genome"]}', flush=True)
    try:
        with (case / 'run.log').open('a') as log:
            subprocess.run([sys.executable, str(CODE / 'run_conditioning_case.py'),
                            '--config', str(case / 'config.json'), '--conditioning', str(cond),
                            '--out', str(gen)], cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, str(CODE / 'evaluate_conditioning_case.py'),
                            '--mesh', str(gen / 'final.obj'), '--conditioning', str(cond),
                            '--out', str(gen / 'metrics'), '--domain-dir', str(ROOT / 'data_real/bracket'),
                            '--size', '384', '--pitch-mm', '1.0'],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        mesh = trimesh.load(gen / 'final.obj', force='mesh')
        mm = mesh.copy(); mm.apply_scale(1000); mm.export(gen / 'final_mm.stl')
        metrics = json.loads((gen / 'metrics/metrics.json').read_text())
        fea = json.loads((gen / 'fea/fea_tet_summary.json').read_text())
        with np.load(out / 'descriptor_reference.npz') as ref:
            field = SDF(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
            occupied = field(ref['xyz'], n_threads=4) > 0
            descriptors = descriptors_from_occupancy(occupied, ref['upper'])
        if not metrics['watertight']: result['invalid_reasons'].append('not_watertight')
        if metrics['components'] != 1: result['invalid_reasons'].append('disconnected')
        containment = metrics['voxel']['containment_fraction_half_voxel_tolerance']
        if containment is None or containment < .99: result['invalid_reasons'].append('outside_domain')
        if not np.isfinite(fea['compliance']) or not 0 < fea['compliance'] < 1:
            result['invalid_reasons'].append('invalid_compliance')
        result.update(descriptors=descriptors, compliance_J=fea['compliance'],
                      volume_mm3=metrics['volume_mm3'], stress_max_MPa=fea['vm_max']/1e6,
                      displacement_max_mm=fea['u_max']*1000,
                      containment_fraction=containment,
                      thin_ridge_fraction=metrics['voxel']['medial_samples_below_3mm_fraction'],
                      mesh_mm=str(gen / 'final_mm.stl'), preview=str(gen / 'metrics/final_preview.png'),
                      metrics=str(gen / 'metrics/metrics.json'), fea=str(gen / 'fea/fea_tet_summary.json'))
        result['valid'] = not result['invalid_reasons']
    except Exception as exc:
        result['invalid_reasons'].append(f'{type(exc).__name__}: {exc}')
    result.update(seconds=time.time() - start, completed_at=time.time())
    save(result_path, result)
    print(f'DONE {job["id"]} valid={result["valid"]} C={result.get("compliance_J")} '
          f'd={result.get("descriptors")} seconds={result["seconds"]:.1f}', flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--gpus', default='0,1,2')
    ap.add_argument('--prepare-only', action='store_true')
    a = ap.parse_args()
    out = a.out.resolve()
    # Prevent concurrent coordinators writing the same archive.
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = prepare(out)
    reference_samples(out, protocol)
    if a.prepare_only:
        print(out / 'protocol.json'); return
    gpus = [int(x) for x in a.gpus.split(',')]
    archives = {k: GridArchive(protocol['dims'], protocol['ranges']) for k in ['map_elites', 'random']}
    histories = {k: [] for k in archives}
    results = []
    def publish(phase):
        save(out / 'summary.json', {'phase': phase, 'updated_at': time.time(),
             'protocol': str(out / 'protocol.json'), 'unique_completed': len(results),
             'unique_gpu_pipeline_seconds': sum(r['seconds'] for r in results),
             'methods': {k: {'evaluations': len(histories[k]), 'history': histories[k],
                             **v.summary()} for k, v in archives.items()}, 'results': results})
    def batch(jobs):
        # Each GPU processes its own fixed queue: no two subprocesses share a GPU.
        queues = [jobs[i::len(gpus)] for i in range(len(gpus))]
        def worker(pair):
            gpu, queue = pair
            return [evaluate(j, gpu, out, protocol) for j in queue]
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            found = [r for group in pool.map(worker, zip(gpus, queues)) for r in group]
        by_id = {r['id']: r for r in found}
        ordered = [by_id[j['id']] for j in jobs]
        results.extend(ordered)
        return ordered
    def ingest(method, evaluated):
        for result in evaluated:
            inserted = archives[method].add(result)
            histories[method].append({'id': result['id'], 'valid': result['valid'], 'inserted': inserted,
                                      **{k: v for k, v in archives[method].summary().items() if k != 'elites'}})
    shared = []
    for i, style in enumerate(BANK):
        shared.append({'id': f'initial_{i:02d}', 'method': 'shared', 'round': 0,
                       'seed': 42000+i, 'parent': None,
                       'genome': {'style': style, 'cfg': 7., 'sp_cfg': 5., 'sp_guide_w_peak': 80.}})
    save(out / 'jobs_initial.json', shared)
    publish('initial_running')
    initial = batch(shared)
    for method in archives: ingest(method, initial)
    publish('initial_complete')
    for round_id in range(1, protocol['rounds']+1):
        plan = out / f'jobs_round_{round_id}.json'
        if plan.exists():
            jobs = json.loads(plan.read_text())
        else:
            jobs = []
            for m, method in enumerate(archives):
                rng = random.Random(protocol['random_seed'] + round_id*100 + m)
                for i in range(3):
                    genome, parent = propose(rng, BANK, archives[method] if method == 'map_elites' else None)
                    jobs.append({'id': f'{method}_r{round_id}_{i:02d}', 'method': method,
                                 'round': round_id, 'seed': 43000+round_id*100+i,
                                 'parent': parent, 'genome': genome})
            # Interleave methods to balance GPU and wall-time effects.
            jobs = [j for pair in zip(jobs[:3], jobs[3:]) for j in pair]
            save(plan, jobs)
        publish(f'round_{round_id}_running')
        evaluated = batch(jobs)
        for method in archives:
            ingest(method, [r for r in evaluated if r['method'] == method])
        publish(f'round_{round_id}_complete')
    publish('complete')
    print(out / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
