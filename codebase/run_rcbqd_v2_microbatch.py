#!/usr/bin/env python3
"""Evaluate the four frozen RC-BQD preview proposals with SDF volume repair."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_raqd_repair_pilot import measurements, save


ROOT = Path(__file__).resolve().parents[1]
CODEBASE = ROOT / "codebase"
SOURCE = ROOT / "experiments/bracket/raqd_online_2026-09-13"
OUT = ROOT / "experiments/bracket/rcbqd_v2_microbatch_2026-09-13"
PROPOSALS = SOURCE / "rcbqd_v2_proposals_preview.json"
VOLUME_CONTROL = {"vw": 20.0, "vol_target": .5, "sp_vw": 10.0, "sp_vol_target": .5}


def command(args, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write("\n$ " + " ".join(map(str, args)) + "\n"); stream.flush()
        subprocess.run([str(v) for v in args], cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def evaluate(index, proposal):
    case = OUT / "cases" / f"v2_{index:02d}"
    result_path = case / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    original = case / "original" / "gen"; repaired = case / "repaired" / "gen"
    raw_sdf = original / "raw_final_refiner_sdf.npz"
    config_path = case / "config.json"
    config = json.loads((SOURCE / "base_config.json").read_text())
    config["seed"] = 47000 + index
    config["stages"]["mesh"].update(proposal["genome"])
    config["stages"]["mesh"].update(VOLUME_CONTROL)
    config["stages"]["mesh"]["save_raw_sdf"] = str(raw_sdf)
    save(config_path, config)
    style = proposal["genome"]["style"]
    conditioning = SOURCE / "image_bank" / style
    env = {**os.environ, "DATA_ROOT": str(ROOT), "D3DS2_ROOT": str(ROOT),
           "EXP_ROOT": str(ROOT / "experiments"), "D3DS2_PY": sys.executable,
           "FENICS_PY": "/home/goya/miniconda3/envs/fenics/bin/python",
           "CUDA_VISIBLE_DEVICES": str(index), "PYTHONUNBUFFERED": "1",
           "FEA_WORK_DIR": str(case / "fea_work"), "PYVISTA_OFF_SCREEN": "true"}
    started = time.time()
    result = {"id": f"v2_{index:02d}", "seed": 47000+index, "gpu": index,
              "proposal": proposal, "case": str(case), "status": "running"}
    save(result_path.with_name("running.json"), result)
    try:
        if not raw_sdf.exists() or not (original / "final.obj").exists():
            command([sys.executable, CODEBASE / "run_conditioning_case.py", "--config", config_path,
                     "--conditioning", conditioning, "--out", original], case / "original.log", env)
        if not (original / "metrics/metrics.json").exists():
            command([sys.executable, CODEBASE / "evaluate_conditioning_case.py", "--mesh", original / "final.obj",
                     "--conditioning", conditioning, "--out", original / "metrics", "--domain-dir",
                     ROOT / "data_real/bracket", "--size", "384", "--pitch-mm", "1.0"],
                    original / "metrics/evaluate.log", env)
        repaired.mkdir(parents=True, exist_ok=True)
        if not (repaired / "mesh.obj").exists():
            command([sys.executable, CODEBASE / "repair_raw_sdf_volume.py", "--raw-sdf", raw_sdf,
                     "--descriptor-reference", SOURCE / "descriptor_reference.npz", "--out", repaired / "mesh.obj",
                     "--report", repaired / "repair.json"], case / "repair.log", env)
        if not (repaired / "hybrid.obj").exists():
            command([sys.executable, CODEBASE / "code/post_hybrid_union_clip.py", "--config", config_path,
                     "--in", repaired / "mesh.obj", "--out", repaired / "hybrid.obj"], repaired / "post.log", env)
        if not (repaired / "final.obj").exists():
            command([sys.executable, CODEBASE / "code/surface_remesh_pre.py", "--config", config_path,
                     "--in", repaired / "hybrid.obj", "--out", repaired / "final.obj"], repaired / "remesh.log", env)
        if not (repaired / "fea/fea_tet_summary.json").exists():
            command([env["FENICS_PY"], CODEBASE / "code/fea_prep_and_run.py", "--config", config_path,
                     "--in", repaired / "final.obj", "--out-dir", repaired / "fea"], repaired / "fea/fea.log", env)
        if not (repaired / "metrics/metrics.json").exists():
            command([sys.executable, CODEBASE / "evaluate_conditioning_case.py", "--mesh", repaired / "final.obj",
                     "--conditioning", conditioning, "--out", repaired / "metrics", "--domain-dir",
                     ROOT / "data_real/bracket", "--size", "384", "--pitch-mm", "1.0"],
                    repaired / "metrics/evaluate.log", env)
        result.update(status="complete",
                      original=measurements(original, SOURCE / "descriptor_reference.npz"),
                      repaired=measurements(repaired, SOURCE / "descriptor_reference.npz"))
        result["repair_pass"] = (abs(result["repaired"]["material_volume_fraction"]-.5) <= .025
                                 and result["repaired"]["watertight"]
                                 and result["repaired"]["components"] == 1
                                 and result["repaired"]["containment_fraction"] >= .99)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    result["seconds"] = time.time()-started
    save(result_path, result)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = json.loads(PROPOSALS.read_text()); jobs = payload["jobs"][:4]
    save(OUT / "protocol.json", {"method": "RC-BQD v2 microbatch", "proposals": str(PROPOSALS),
         "count": len(jobs), "seeds": [47000+i for i in range(len(jobs))],
         "gpus": list(range(len(jobs))), "target_final_volume_fraction": .5,
         "volume_control": VOLUME_CONTROL,
         "repair": "CAD-reference SDF quantile followed by standard post and FEA"})
    results = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(evaluate, i, job): i for i, job in enumerate(jobs)}
        for future in as_completed(futures):
            result = future.result(); results.append(result)
            print(result["id"], result["status"], result.get("repair_pass"), flush=True)
    results.sort(key=lambda r: r["id"])
    complete = [r for r in results if r["status"] == "complete"]
    summary = {"status": "complete" if len(complete)==len(jobs) else "partial",
               "evaluations": len(results), "complete": len(complete),
               "repair_passes": sum(r.get("repair_pass",False) for r in results),
               "results": results}
    save(OUT / "summary.json", summary)
    print(OUT / "summary.json")


if __name__ == "__main__":
    main()
