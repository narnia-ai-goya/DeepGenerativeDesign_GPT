"""Run one isolated conditioning comparison with a frozen config and stage status."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from run_from_image import DATA_ROOT, STAGES, compliance_of, sane_c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--conditioning', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    config, conditioning, out = a.config.resolve(), a.conditioning.resolve(), a.out.resolve()
    cfg = json.loads(config.read_text())
    paths = sorted(conditioning.glob('*.png'))
    expected = cfg['stages']['mesh']['n_views']
    if len(paths) != expected:
        raise SystemExit(f'Expected exactly {expected} images, found {len(paths)}')
    inputs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    signature = {'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(), 'images': inputs}
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / 'case_status.json'
    state = json.loads(status_path.read_text()) if status_path.exists() else {}
    if state and state['inputs'] != signature:
        raise SystemExit('Inputs changed; choose a new output directory to avoid stale results')
    if not state and any((out / name).exists() for name in ('mesh.obj', 'final.obj')):
        raise SystemExit('Output has untracked artifacts; choose a fresh output directory')
    state = state or {'inputs': signature, 'config': str(config), 'conditioning': str(conditioning),
                      'output': str(out), 'stages': {}, 'started_at': time.time()}
    def save():
        tmp = status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(status_path)
    save()
    os.chdir(DATA_ROOT)
    c = {'config': str(config)}
    for name in ('gen', 'post', 'fea'):
        if state['stages'].get(name, {}).get('ok'):
            continue
        # Retry failed/unrecorded stages explicitly; do not trust orphaned files.
        start = time.time()
        extra = {'extra': ['--target-dir', str(conditioning)]} if name == 'gen' else {}
        try:
            ok = STAGES[name](cfg['domain'], c, conditioning.name, True, wd=out, **extra)
        except Exception as exc:
            state['stages'][name] = {'ok': False, 'seconds': time.time()-start,
                                     'error': f'{type(exc).__name__}: {exc}'}
            save()
            raise
        state['stages'][name] = {'ok': ok, 'seconds': time.time()-start}
        save()
        if not ok:
            raise SystemExit(f'Stage {name} failed; see {out}')
    state['compliance_J'] = compliance_of(out, sane_c(c))
    state['completed_at'] = time.time()
    save()


if __name__ == '__main__':
    main()
