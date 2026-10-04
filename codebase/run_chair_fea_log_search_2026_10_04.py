"""Stage-wise logarithmic FEA-weight sweep for the fixed chair image and BCs.

The displayed factor is dimensionless: dense fea_w = factor * 1e-7 and
sparse sp_fea_w = factor * 8e-5.  This preserves each stage's existing
normalization while testing 0, 10, 100, 1000, and 10000.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from make_chair_domain import ROOT
from run_chair_dual_load_fea_2026_10_04 import INPUT
from run_chair_existing_fea_on_015_2026_10_04 import OUT
from run_chair_sparse_fea_loop_2026_10_03 import command
from run_connectivity_qd_sampling import generation_env


BASE = OUT / "diagnostics/fea_log_search_2026-10-04"
FACTORS = (0, 10, 100, 1000, 10000)


def run(stage: str, factor: int, gpu: int, dense_factor: int | None) -> dict:
    if factor not in FACTORS:
        raise ValueError(f"factor must be one of {FACTORS}")
    if stage == "sparse" and dense_factor not in FACTORS:
        raise ValueError("sparse requires --dense-factor")
    case = BASE / (f"dense_f{factor:05d}" if stage == "dense" else
                   f"dense_f{dense_factor:05d}_sparse_f{factor:05d}")
    case.mkdir(parents=True, exist_ok=True)
    template = (OUT / "dense/config.json" if stage == "dense" else
                OUT / "diagnostics/dense_core_pilot/weight_search/w060_fea_on/config.json")
    cfg = json.loads(template.read_text())
    cfg["name"] = f"chair_fea_log_{stage}_f{factor:05d}"
    mesh = cfg["stages"]["mesh"]
    if stage == "dense":
        mesh["skip_sparse"] = True
        mesh["load_dense_cache"] = None
        mesh["save_dense_cache"] = str(case / "dense_cache.npz")
        mesh["fea_w"] = factor * 1e-7
        mesh["sp_fea_w"] = 0.0
    else:
        dense_case = BASE / f"dense_f{dense_factor:05d}"
        cache = dense_case / "dense_cache.npz"
        if not cache.exists():
            raise FileNotFoundError(cache)
        mesh["skip_sparse"] = False
        mesh["load_dense_cache"] = str(cache)
        mesh["save_dense_cache"] = None
        mesh["fea_w"] = 0.0
        mesh["sp_fea_w"] = factor * 8e-5
        mesh["sp_fea_mode"] = "manual"
        # Hold the geometric Dense-core setting fixed at the previous best.
        mesh["sp_guide_w"] = 30.0
        mesh["sp_guide_w_peak"] = 30.0
        mesh["sp_dense_core_mm"] = 6.0
        mesh["sp_dense_core_w"] = 2.0
        mesh["sp_sdf_inside_low"] = True
        mesh["sp_sdf_guidance_threshold"] = 0.6
    config = case / "config.json"
    config.write_text(json.dumps(cfg, indent=2) + "\n")
    env = generation_env(gpu)
    env.update({"VANILLA": "0", "FEA_LOAD_MAGNITUDE": "800",
                "FEA_MAX_NODE_MAP_DISTANCE": ".04", "BC_SURFACE_DIST": "1",
                "BC_DIST": ".025", "FEA_WORK_DIR": str(case / "fea_work"),
                "FEA_KSP_TYPE": "gmres", "FEA_PC_TYPE": "gamg",
                "FEA_KSP_RTOL": "1e-7", "FEA_KSP_MAX_IT": "1500",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "D3DS2_SAVE_PRE_REFINER": "1"})
    for key in ("FEA_SECOND_LOAD_STL", "FEA_SECOND_LOAD_MAGNITUDE", "FEA_SECOND_LOAD_MODE"):
        env.pop(key, None)
    started = time.monotonic()
    output = case / "generation"
    log = case / "generation.log"
    with log.open("w") as stream:
        proc = subprocess.run(command(config, INPUT, output), cwd=ROOT,
                              env=env, stdout=stream, stderr=subprocess.STDOUT)
    trace = log.read_text(errors="replace")
    tag = "[dense simultaneous FEA step " if stage == "dense" else "[sp simultaneous FEA step "
    record = {"stage": stage, "factor": factor, "dense_factor": dense_factor,
              "fea_weight": mesh["fea_w"] if stage == "dense" else mesh["sp_fea_w"],
              "gpu": gpu, "exit_code": proc.returncode,
              "elapsed_seconds": time.monotonic() - started,
              "fea_hits": trace.count(tag), "mesh": str(output / "mesh.obj"),
              "dense_cache": str(case / "dense_cache.npz") if stage == "dense" else mesh["load_dense_cache"],
              "config": str(config), "log": str(log)}
    (case / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    if proc.returncode:
        raise RuntimeError(f"generation failed; inspect {log}")
    if factor > 0 and record["fea_hits"] == 0:
        raise RuntimeError(f"FEA did not run; inspect {log}")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("dense", "sparse"), required=True)
    parser.add_argument("--factor", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--dense-factor", type=int)
    args = parser.parse_args()
    run(args.stage, args.factor, args.gpu, args.dense_factor)
