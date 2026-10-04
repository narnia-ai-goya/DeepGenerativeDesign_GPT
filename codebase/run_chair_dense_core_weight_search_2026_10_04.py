"""Sparse core-loss weight search with fixed Dense cache and fixed 6 mm core."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from make_chair_domain import ROOT
from run_chair_dual_load_fea_2026_10_04 import INPUT
from run_chair_existing_fea_on_015_2026_10_04 import OUT
from run_chair_sparse_fea_loop_2026_10_03 import command
from run_connectivity_qd_sampling import generation_env


BASE = OUT / "diagnostics/dense_core_pilot/weight_search"
TEMPLATE = OUT / "diagnostics/dense_core_pilot/core_6mm_w30_fea_off/config.json"
CASES = ((5, 3), (10, 4), (15, 5), (60, 6), (120, 7))


def run_case(effective_weight: int, gpu: int) -> dict:
    case = BASE / f"w{effective_weight:03d}"
    case.mkdir(parents=True, exist_ok=True)
    config = json.loads(TEMPLATE.read_text())
    config["name"] = f"chair_core6_weight_{effective_weight}"
    mesh = config["stages"]["mesh"]
    # All other sparse guidance losses are zero.  Keep guide_w=30 and change
    # only the core coefficient; effective weight = guide_w * core_w.
    mesh["sp_dense_core_w"] = effective_weight / 30.0
    config_path = case / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    env = generation_env(gpu)
    env.update({
        "VANILLA": "0", "D3DS2_SPARSE_GRAD_DEBUG": "1",
        "D3DS2_SAVE_PRE_REFINER": "1", "FEA_LOAD_MAGNITUDE": "800",
        "FEA_MAX_NODE_MAP_DISTANCE": ".04", "BC_SURFACE_DIST": "1",
        "BC_DIST": ".025", "FEA_WORK_DIR": str(case / "fea_work"),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    for key in ("FEA_SECOND_LOAD_STL", "FEA_SECOND_LOAD_MAGNITUDE", "FEA_SECOND_LOAD_MODE"):
        env.pop(key, None)
    log = case / "generation.log"
    output = case / "generation"
    with log.open("w") as stream:
        process = subprocess.run(command(config_path, INPUT, output), cwd=ROOT,
                                 env=env, stdout=stream, stderr=subprocess.STDOUT)
    record = {"effective_weight": effective_weight, "gpu": gpu,
              "exit_code": process.returncode, "mesh": str(output / "mesh.obj"),
              "config": str(config_path), "log": str(log)}
    (case / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=len(CASES)) as executor:
        for result in executor.map(lambda args: run_case(*args), CASES):
            print(json.dumps(result), flush=True)
