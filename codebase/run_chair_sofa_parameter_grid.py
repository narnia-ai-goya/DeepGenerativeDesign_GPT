#!/usr/bin/env python3
"""Controlled image/volume/path sweep for the sofa-style open-arm chair."""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess

from run_connectivity_qd_sampling import GENERATOR, PYTHON, ROOT, generation_env


BASE = ROOT / "experiments/chair/sofa_style_2026-09-28"
SOURCE = BASE / "open_arm/config_dense.json"
OUT = BASE / "tuning_open_arm"
IMAGE = BASE / "open_arm/input_lr162"


def cases():
    for projection, volume, path in itertools.product((50, 150), (.20, .35), (0.0, 3.5)):
        name = f"p{projection}_v{int(volume * 100):02d}_pw{str(path).replace('.', '')}"
        yield name, projection, volume, path


def prepare() -> None:
    source = json.loads(SOURCE.read_text())
    manifest = {}
    for name, projection, volume, path in cases():
        case = OUT / name
        case.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(json.dumps(source))
        cfg["name"] = f"chair_sofa_open_arm_{name}"
        mesh = cfg["stages"]["mesh"]
        mesh["image_proj_w"] = float(projection)
        mesh["vw"] = 50.0
        mesh["vol_target"] = float(volume)
        mesh["pw"] = float(path)
        if path == 0:
            mesh["load_path_mask"] = None
        mesh["save_dense_cache"] = str(case / "dense_cache.npz")
        mesh["load_dense_cache"] = None
        mesh["skip_sparse"] = True
        config = case / "config_dense.json"
        config.write_text(json.dumps(cfg, indent=2) + "\n")
        manifest[name] = {
            "config": str(config),
            "image_projection_weight": projection,
            "volume_weight": 50.0,
            "volume_target": volume,
            "load_path_weight": path,
            "input_dir": str(IMAGE),
            "dense_mesh": str(case / "dense/mesh_dense.obj"),
            "dense_cache": str(case / "dense_cache.npz"),
        }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(OUT / "manifest.json")


def run(name: str, gpu: int) -> None:
    case = OUT / name
    config = case / "config_dense.json"
    if not config.exists():
        raise FileNotFoundError(config)
    (case / "dense").mkdir(exist_ok=True)
    command = [str(PYTHON), str(GENERATOR), "--config", str(config),
               "--target-dir", str(IMAGE), "--out", str(case / "dense")]
    with (case / "dense.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=generation_env(gpu),
                                stdout=log, stderr=subprocess.STDOUT)
    (case / "run.json").write_text(json.dumps({
        "case": name, "gpu": gpu, "exit_code": result.returncode,
        "command": command}, indent=2) + "\n")
    print(f"{name}: exit {result.returncode}; log {case / 'dense.log'}", flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    one = sub.add_parser("run")
    one.add_argument("name", choices=[name for name, *_ in cases()])
    one.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        run(args.name, args.gpu)
