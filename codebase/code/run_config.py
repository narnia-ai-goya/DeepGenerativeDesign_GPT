"""Run-config loader (CLI > config > script default).

Usage in a script:

    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    # ... add_argument for everything else with normal defaults ...
    apply_config(ap, stage='mesh')   # before parse_args
    args = ap.parse_args()

`apply_config` reads --config from argv, loads the JSON, and calls
`ap.set_defaults(**stage_dict)`. argparse already gives CLI > set_defaults,
so explicit CLI flags still win.

Config layout: see configs/bracket.json. Each `stages.<name>` block is a
flat dict whose keys match the script's argparse dest names (underscores,
not dashes). Top-level `seed`, `views` apply to every stage that has those
dests. Unknown keys are silently ignored, so it is safe to include extra
descriptive fields (name, domain) at the top level.

set_defaults does NOT override `required=True` args — keep required args on
the chain-shell CLI (e.g. --in/--out per style); put tunables in config.
"""
import argparse
import json
import sys
from pathlib import Path


def _find_config_arg(argv=None):
    argv = list(sys.argv if argv is None else argv)
    for i, a in enumerate(argv):
        if a == '--config' and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith('--config='):
            return a.split('=', 1)[1]
    return None


def load_config(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'config not found: {path}')
    with open(p) as f:
        return json.load(f)


def apply_config(parser: argparse.ArgumentParser, stage: str, argv=None):
    """If --config is in argv, set parser defaults from cfg['stages'][stage].
    Top-level seed/views also propagate. Returns the loaded cfg dict (or None)."""
    path = _find_config_arg(argv)
    if path is None:
        return None
    cfg = load_config(path)

    flat = {}
    for k in ('seed', 'views'):
        if cfg.get(k) is not None:
            flat[k] = cfg[k]
    for k, v in (cfg.get('stages', {}).get(stage) or {}).items():
        if v is not None:
            flat[k.replace('-', '_')] = v

    known = {a.dest for a in parser._actions}
    parser.set_defaults(**{k: v for k, v in flat.items() if k in known})
    return cfg
