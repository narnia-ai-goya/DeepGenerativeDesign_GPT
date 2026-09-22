"""Audit frozen inputs, equal budgets, feedback lineage and final result provenance."""
import argparse
import hashlib
import json
from pathlib import Path

from qd_archive import GridArchive, PARAMETERS


def audit(out):
    state = json.loads((out / 'summary.json').read_text())
    protocol = json.loads((out / 'protocol.json').read_text())
    base = json.loads((out / 'base_config.json').read_text())
    checks = []
    def check(name, ok, detail=None):
        checks.append({'name': name, 'ok': bool(ok), 'detail': detail})
    check('run_complete', state['phase'] == 'complete', state['phase'])
    check('unique_budget', len(state['results']) == protocol['unique_jobs'], len(state['results']))
    check('unique_ids', len({r['id'] for r in state['results']}) == len(state['results']))
    changed = [path for path, digest in protocol['hashes'].items()
               if not Path(path).exists() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest]
    check('frozen_hashes', not changed, changed)
    ref = json.loads((out / 'descriptor_reference.json').read_text())
    check('reference_hash', hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() == ref['sha256'])
    by_id = {r['id']: r for r in state['results']}
    plans = [j for p in sorted(out.glob('jobs_*.json')) for j in json.loads(p.read_text())]
    check('planned_ids_match_results', {j['id'] for j in plans} == set(by_id))
    for job in plans:
        r = by_id.get(job['id'])
        if r is None: continue
        check(f'{job["id"]}:plan', all(r[k] == v for k, v in job.items()))
        case = Path(r['case_dir'])
        config = json.loads((case / 'config.json').read_text())
        expected = json.loads(json.dumps(base))
        expected['seed'] = job['seed']
        expected['stages']['mesh'].update({k: job['genome'][k] for k in PARAMETERS})
        check(f'{job["id"]}:only_authorized_config_changes', config == expected)
        if not r['valid']: continue
        status = json.loads((case / 'gen/case_status.json').read_text())
        check(f'{job["id"]}:stages', all(status['stages'].get(k, {}).get('ok') for k in ['gen', 'post', 'fea']))
        check(f'{job["id"]}:config_hash', status['inputs']['config_sha256'] == hashlib.sha256((case / 'config.json').read_bytes()).hexdigest())
        expected_images = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((out / 'image_bank' / r['genome']['style']).glob('*.png'))}
        check(f'{job["id"]}:image_hashes', status['inputs']['images'] == expected_images)
        fea = json.loads(Path(r['fea']).read_text())
        check(f'{job["id"]}:fea', fea['compliance'] == r['compliance_J'] and
              fea['force_N'] == 1000 and fea['force_dir'] == '0,0,-1')
        metrics = json.loads(Path(r['metrics']).read_text())
        check(f'{job["id"]}:geometry', metrics['watertight'] and metrics['components'] == 1 and
              metrics['voxel']['containment_fraction_half_voxel_tolerance'] >= .99)
        check(f'{job["id"]}:artifacts', all(Path(r[k]).exists() for k in ['mesh_mm', 'preview', 'metrics', 'fea']))
        import numpy as np
        import trimesh
        source_mesh = trimesh.load(case / 'gen/final.obj', force='mesh')
        exported_mesh = trimesh.load(r['mesh_mm'], force='mesh')
        check(f'{job["id"]}:stl_millimetre_scale',
              np.allclose(exported_mesh.bounds, source_mesh.bounds*1000, rtol=1e-5, atol=1e-4))
    for method, observed in state['methods'].items():
        check(f'{method}:budget', observed['evaluations'] == protocol['budget_per_method'])
        replay = GridArchive(protocol['dims'], protocol['ranges'])
        initial = [r for r in state['results'] if r['method'] == 'shared']
        for r in initial: replay.add(r)
        check(f'{method}:common_initial', [h['id'] for h in observed['history'][:3]] == [r['id'] for r in initial])
        for round_id in range(1, protocol['rounds']+1):
            parents = {r['id'] for r in replay.elites.values()}
            children = [r for r in state['results'] if r['method'] == method and r['round'] == round_id]
            check(f'{method}:round_{round_id}_size', len(children) == 3)
            for r in children:
                parent_ok = r['parent'] in parents if method == 'map_elites' and parents else r['parent'] is None
                check(f'{r["id"]}:parent_from_previous_archive', parent_ok)
            for r in children: replay.add(r)
        check(f'{method}:archive_replay', all(observed[k] == v for k, v in replay.summary().items()))
    for round_id in range(1, protocol['rounds']+1):
        seeds = {m: sorted(r['seed'] for r in state['results'] if r['method'] == m and r['round'] == round_id)
                 for m in ['map_elites', 'random']}
        check(f'round_{round_id}:paired_seeds', seeds['map_elites'] == seeds['random'])
    result = {'passed': all(c['ok'] for c in checks), 'checks': len(checks),
              'failed': [c for c in checks if not c['ok']], 'details': checks}
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'details'}, indent=2))
    return result['passed']


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out', type=Path)
    raise SystemExit(0 if audit(ap.parse_args().out.resolve()) else 1)
