#!/usr/bin/env python3
"""Controlled sparse pre/post-refiner comparison for one chair case."""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_connectivity_qd_sampling import GENERATOR, PYTHON, ROOT, generation_env

SOURCE = ROOT / "experiments/chair/sofa_style_2026-09-28/backrest_bc_multi_examples_2026-09-29/diagonal_braced"
OUT = ROOT / "experiments/chair/sofa_style_2026-09-28/mmc_dense_pilot_2026-09-29"


def run(name: str, gpu: int, thick_w: float) -> dict:
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    config = json.loads((SOURCE / "sparse_pw2_d13/config_sparse.json").read_text())
    config["name"] = f"chair_mmc_diagnostic_{name}"
    config["stages"]["mesh"]["sp_thick_w"] = thick_w
    config_path = folder / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    command = [str(PYTHON), str(GENERATOR), "--config", str(config_path),
               "--target-dir", str(SOURCE / "input_lr162"), "--out", str(folder / "generation")]
    env = generation_env(gpu)
    env["D3DS2_SAVE_PRE_REFINER"] = "1"
    with (folder / "run.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    record = {"name": name, "gpu": gpu, "thick_w": thick_w,
              "exit_code": result.returncode, "command": command,
              "pre_refiner": str(folder / "generation/mesh_pre_refiner.obj"),
              "final": str(folder / "generation/mesh.obj")}
    (folder / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, "baseline", 0, 10.0),
                   pool.submit(run, "no_thickness", 1, 0.0)]
        records = [future.result() for future in futures]
    (OUT / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))
