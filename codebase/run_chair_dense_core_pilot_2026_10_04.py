"""Paired sparse dense-core guidance pilots from one fixed chair Dense cache."""
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


BASE = OUT / "diagnostics/dense_core_pilot"


def run_case(margin_mm: int, gpu: int) -> dict:
    folder = BASE / f"core_{margin_mm}mm_w30_fea_off"
    folder.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((OUT / "sparse/config.json").read_text())
    cfg["name"] = f"chair_dense_core_{margin_mm}mm_fea_off"
    mesh = cfg["stages"]["mesh"]
    mesh.update({
        "fea_w": 0.0,
        "sp_fea_w": 0.0,
        "sp_guide_w": 30.0,
        "sp_guide_w_peak": 30.0,
        "sp_param_pool_schedule": "",
        "sp_sdf_inside_low": True,
        "sp_sdf_guidance_threshold": 0.6,
        "sp_dense_core_mm": float(margin_mm),
        "sp_dense_core_w": 1.0,
        "sp_dense_core_project": False,
    })
    config = folder / "config.json"
    config.write_text(json.dumps(cfg, indent=2) + "\n")
    env = generation_env(gpu)
    env.update({
        "VANILLA": "0", "FEA_LOAD_MAGNITUDE": "800",
        "FEA_MAX_NODE_MAP_DISTANCE": ".04", "BC_SURFACE_DIST": "1",
        "BC_DIST": ".025", "FEA_WORK_DIR": str(folder / "fea_work"),
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "D3DS2_SAVE_PRE_REFINER": "1",
        "D3DS2_SPARSE_GRAD_DEBUG": "1",
    })
    for key in ("FEA_SECOND_LOAD_STL", "FEA_SECOND_LOAD_MAGNITUDE", "FEA_SECOND_LOAD_MODE"):
        env.pop(key, None)
    output = folder / "generation"
    log = folder / "generation.log"
    with log.open("w") as stream:
        process = subprocess.run(command(config, INPUT, output), cwd=ROOT, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
    row = {"margin_mm": margin_mm, "gpu": gpu, "exit_code": process.returncode,
           "mesh": str(output / "mesh.obj"), "log": str(log), "config": str(config)}
    (folder / "run.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as executor:
        for result in executor.map(lambda pair: run_case(*pair), ((8, 2), (6, 3))):
            print(json.dumps(result), flush=True)
