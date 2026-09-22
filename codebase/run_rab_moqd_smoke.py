"""Run a three-candidate, no-volume-gate RAB-MOQD smoke loop."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from analyze_raqd_descriptors import features_for_mesh
from qd_archive import PARAMETERS
from rab_moqd_acquisition import (archive_fronts, cell_of, hypervolume_2d,
                                  objective_point, propose_rab_moqd)

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "codebase"
SOURCE = ROOT / "experiments/bracket/raqd_online_2026-09-13"
WARMSTART = ROOT / "experiments/bracket/rab_moqd_warmstart_2026-09-13.json"
DEFAULT_OUT = ROOT / "experiments/bracket/rab_moqd_smoke_2026-09-13"
DESCRIPTOR_NAMES = ["normalized_void_scale", "strain_energy_concentration"]
DESCRIPTOR_RANGES = [[.020, .041], [.58, .78]]
OBJECTIVE_BOUNDS = [[.0035, .014], [.35, .70]]
DIMS = [4, 4]


def save(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(job, gpu, out, characteristic_length_mm):
    case = out / "cases" / job["id"]
    result_path = case / "result.json"
    if result_path.exists(): return json.loads(result_path.read_text())
    case.mkdir(parents=True, exist_ok=True)
    config = json.loads((SOURCE / "base_config.json").read_text())
    config["seed"] = job["seed"]
    config["stages"]["mesh"].update({name: job["genome"][name] for name in PARAMETERS})
    config["stages"]["mesh"].update({"vw": 0.0, "sp_vw": 0.0,
                                      "aug_lag": False, "sp_aug_lag": False,
                                      "sp_fea_step_size": .03})
    save(case / "config.json", config)
    conditioning = SOURCE / "image_bank" / job["genome"]["style"]
    gen = case / "gen"
    env = {**os.environ, "DATA_ROOT": str(ROOT), "D3DS2_ROOT": str(ROOT),
           "EXP_ROOT": str(ROOT / "experiments"), "D3DS2_PY": sys.executable,
           "FENICS_PY": "/home/goya/miniconda3/envs/fenics/bin/python",
           "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
           "FEA_WORK_DIR": str(case / "fea_work"), "PYVISTA_OFF_SCREEN": "true"}
    result = {**job, "gpu": gpu, "case_dir": str(case), "started_at": time.time(),
              "valid": False, "constraints_satisfied": False, "invalid_reasons": []}
    save(case / "running.json", result)
    print(f"START {job['id']} GPU={gpu} target={job['target_cell']}", flush=True)
    try:
        with (case / "run.log").open("a") as log:
            subprocess.run([sys.executable, str(CODE / "run_conditioning_case.py"),
                            "--config", str(case / "config.json"), "--conditioning", str(conditioning),
                            "--out", str(gen)], cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, str(CODE / "evaluate_conditioning_case.py"),
                            "--mesh", str(gen / "final.obj"), "--conditioning", str(conditioning),
                            "--out", str(gen / "metrics"), "--domain-dir", str(ROOT / "data_real/bracket"),
                            "--size", "384", "--pitch-mm", "1.0"], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        with np.load(SOURCE / "descriptor_reference.npz") as reference:
            xyz = np.ascontiguousarray(reference["xyz"], dtype=np.float32)
            pitch = float(reference["pitch_m"])
        features = features_for_mesh(gen / "final.obj", xyz, pitch)
        metrics = json.loads((gen / "metrics/metrics.json").read_text())
        fea = json.loads((gen / "fea/fea_tet_summary.json").read_text())
        containment = metrics["voxel"]["containment_fraction_half_voxel_tolerance"]
        if not metrics["watertight"]: result["invalid_reasons"].append("not_watertight")
        if metrics["components"] != 1: result["invalid_reasons"].append("disconnected")
        if containment is None or containment < .99: result["invalid_reasons"].append("outside_domain")
        if not math.isfinite(fea["compliance"]) or not 0 < fea["compliance"] < 1:
            result["invalid_reasons"].append("invalid_compliance")
        descriptors = [features["void_clearance_mean_mm"]/characteristic_length_mm,
                       fea["strain_energy_concentration"]]
        valid = not result["invalid_reasons"]
        result.update(valid=valid, constraints_satisfied=valid,
                      realized_descriptors=descriptors,
                      measured_design_volume_fraction=features["material_volume_fraction"],
                      compliance_J=fea["compliance"], volume_mm3=metrics["volume_mm3"],
                      features=features, containment_fraction=containment,
                      mesh=str((gen / "final.obj").resolve()),
                      preview=str((gen / "metrics/final_preview.png").resolve()))
    except Exception as exc:
        result["invalid_reasons"].append(f"{type(exc).__name__}: {exc}")
    result.update(seconds=time.time()-result["started_at"], completed_at=time.time())
    save(result_path, result)
    print(f"DONE {job['id']} valid={result['valid']} d={result.get('realized_descriptors')} ", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    warm = json.loads(WARMSTART.read_text())
    observations = warm["results"]
    proposal_path = out / "proposal.json"
    diagnostics_path = out / "acquisition_diagnostics.json"
    if proposal_path.exists():
        jobs = json.loads(proposal_path.read_text())["jobs"]
    else:
        proposed, diagnostics = propose_rab_moqd(
            observations, args.count, 2026091302, DESCRIPTOR_RANGES, OBJECTIVE_BOUNDS, DIMS)
        jobs = [{"id": f"rab_moqd_{i:02d}", "method": "rab_moqd", "round": 1,
                 "replicate_group": f"rab_moqd_{i:02d}", "seed": 49000+i, **item}
                for i, item in enumerate(proposed)]
        save(proposal_path, {"jobs": jobs, "descriptor_names": DESCRIPTOR_NAMES,
                             "descriptor_ranges": DESCRIPTOR_RANGES,
                             "objective_bounds": OBJECTIVE_BOUNDS, "dims": DIMS})
        save(diagnostics_path, diagnostics)
    protocol = {
        "status": "method_smoke_test_not_benchmark", "created_at": time.time(),
        "warmstart": str(WARMSTART), "descriptor_names": DESCRIPTOR_NAMES,
        "descriptor_ranges": DESCRIPTOR_RANGES, "dims": DIMS,
        "objectives": ["minimize compliance_J", "minimize material_volume_fraction"],
        "objective_bounds": OBJECTIVE_BOUNDS,
        "hard_constraints": ["watertight", "single component", "containment >= 0.99", "valid FEA"],
        "volume_hard_constraint": False, "sp_fea_step_size": .03,
        "acquisition": "Monte Carlo expected cell-conditioned hypervolume improvement",
        "hashes": {str(path.resolve()): sha(path) for path in [
            CODE / "rab_moqd_acquisition.py", CODE / "run_rab_moqd_smoke.py",
            CODE / "code/behavior_descriptors.py", CODE / "code/fea_tet_from_mesh.py",
            SOURCE / "base_config.json", SOURCE / "descriptor_reference.npz"]},
    }
    save(out / "protocol.json", protocol)
    if args.prepare_only:
        print(proposal_path); return
    gpus = [int(value) for value in args.gpus.split(",")]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(evaluate, job, gpus[index % len(gpus)], out,
                               warm["characteristic_length_mm"])
                   for index, job in enumerate(jobs)]
        results = [future.result() for future in futures]

    before = archive_fronts(observations, DESCRIPTOR_RANGES, OBJECTIVE_BOUNDS, DIMS)
    after = archive_fronts([*observations, *results], DESCRIPTOR_RANGES, OBJECTIVE_BOUNDS, DIMS)
    before_hv = sum(hypervolume_2d(front) for front in before.values())
    after_hv = sum(hypervolume_2d(front) for front in after.values())
    attempts = []
    for row in results:
        cell = cell_of(row.get("realized_descriptors", []), DESCRIPTOR_RANGES, DIMS) if row.get("valid") else None
        realized_point = objective_point(row["compliance_J"], row["measured_design_volume_fraction"],
                                         OBJECTIVE_BOUNDS) if row.get("valid") else None
        attempts.append({**row, "realized_cell": list(cell) if cell else None,
                         "target_hit": cell == tuple(row["target_cell"]) if cell else False,
                         "normalized_objectives": realized_point})
    summary = {
        "status": "complete", "protocol": str((out / "protocol.json").resolve()),
        "valid": sum(row["valid"] for row in results), "evaluations": len(results),
        "target_hits": sum(row["target_hit"] for row in attempts),
        "occupied_cells_before": len(before), "occupied_cells_after": len(after),
        "qd_hypervolume_before": before_hv, "qd_hypervolume_after": after_hv,
        "qd_hypervolume_gain": after_hv-before_hv, "attempts": attempts,
    }
    save(out / "summary.json", summary)
    cards = "".join(
        f'<article><img src="{html.escape(os.path.relpath(row["preview"], out))}">'
        f'<h3>{row["id"]}</h3><p>target {row["target_cell"]} → realized {row["realized_cell"]}<br>'
        f'C={1000*row["compliance_J"]:.3f} mJ · V={row["measured_design_volume_fraction"]:.3f}<br>'
        f'd={list(map(lambda x: round(x,4), row["realized_descriptors"]))}</p>'
        f'<p class="path">{html.escape(row["mesh"])}</p></article>' for row in attempts if row["valid"])
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RAB-MOQD smoke loop</title><style>body{{font-family:system-ui,sans-serif;max-width:1180px;margin:34px auto;padding:0 22px;color:#17212b;line-height:1.55}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}article{{border:1px solid #d5dce2;border-radius:10px;padding:13px}}img{{width:100%}}.path{{font-size:11px;overflow-wrap:anywhere}}.note{{background:#eef7f4;border-left:4px solid #177b65;padding:12px 16px}}</style><h1>RAB-MOQD smoke loop</h1><p class="note">Volume hard gate 없이 normalized void scale × strain-energy concentration의 cell-wise compliance–volume Pareto improvement를 선택했다.</p><p>valid {summary['valid']}/{summary['evaluations']} · target hits {summary['target_hits']}/{summary['evaluations']} · occupied cells {len(before)} → {len(after)} · QD-HV gain {summary['qd_hypervolume_gain']:.5f}</p><div class="cards">{cards}</div><p class="path">{out / 'summary.json'}</p></html>'''
    (out / "report.html").write_text(page)
    print(out / "report.html")


if __name__ == "__main__":
    main()
