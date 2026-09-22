#!/usr/bin/env python3
"""One-command image-first run for the supplied disk-brake caliper domain.

This is a small, explicit entry point over ``run_from_image.py``.  It intentionally
contains no tuning parameters: geometry, FEM, post-processing and conditioning paths
remain in ``configs/caliper.json``, the single source of truth.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Run the disk-brake caliper pipeline.')
    parser.add_argument('--style', default='builtin_kagome_consistent',
                        help='one or comma-separated conditioning-image directories')
    parser.add_argument('--stage', default='all',
                        help='all, or comma-separated gen,post,fea stages')
    parser.add_argument('--force', action='store_true', help='recompute completed stage outputs')
    parser.add_argument('--grid', action='store_true', help='render the standard result grid afterwards')
    args = parser.parse_args()
    cmd = [sys.executable, str(ROOT / 'run_from_image.py'), '--domain', 'caliper',
           '--style', args.style, '--stage', args.stage]
    if args.force:
        cmd.append('--force')
    if args.grid:
        cmd.append('--grid')
    raise SystemExit(subprocess.call(cmd, cwd=ROOT, env=os.environ.copy()))


if __name__ == '__main__':
    main()
