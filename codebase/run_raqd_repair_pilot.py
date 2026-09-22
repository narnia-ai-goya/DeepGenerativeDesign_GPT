#!/usr/bin/env python3
"""Re-run one frozen bracket candidate, save its raw SDF, and test volume repair."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from analyze_raqd_descriptors import features_for_mesh
from raqd_core import cell_index


ROOT = Path(__file__).resolve().parents[1]
CODEBASE = ROOT / "codebase"
SOURCE_STUDY = ROOT / "experiments/bracket/raqd_online_2026-09-13"
DEFAULT_OUT = ROOT / "experiments/bracket/raqd_v2_repair_pilot_2026-09-13"


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def run(command, log, env, python=None):
    command = [str(v) for v in command]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def measurements(gen, descriptor_reference):
    with np.load(descriptor_reference) as reference:
        xyz = np.ascontiguousarray(reference["xyz"], dtype=np.float32)
        pitch = float(reference["pitch_m"])
    features = features_for_mesh(gen / "final.obj", xyz, pitch)
    metrics = json.loads((gen / "metrics/metrics.json").read_text())
    fea = json.loads((gen / "fea/fea_tet_summary.json").read_text())
    return {
        "final_mesh": str((gen / "final.obj").resolve()),
        "preview": str((gen / "metrics/final_preview.png").resolve()),
        "material_volume_fraction": features["material_volume_fraction"],
        "void_clearance_mean_mm": features["void_clearance_mean_mm"],
        "material_anisotropy": features["material_anisotropy"],
        "compliance_J": fea["compliance"],
        "watertight": metrics["watertight"],
        "components": metrics["components"],
        "containment_fraction": metrics["voxel"]["containment_fraction_half_voxel_tolerance"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-id", default="raqd_14")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--sp-fea-step-size", type=float, default=.01)
    args = ap.parse_args()
    out = args.out.resolve()
    source = SOURCE_STUDY / "cases" / args.source_id
    source_result = json.loads((source / "result.json").read_text())
    style = source_result["genome"]["style"]
    conditioning = SOURCE_STUDY / "image_bank" / style
    descriptor_reference = SOURCE_STUDY / "descriptor_reference.npz"
    original = out / "original" / "gen"
    repaired = out / "repaired" / "gen"
    raw_sdf = original / "raw_final_refiner_sdf.npz"
    config_path = out / "config.json"
    if not config_path.exists():
        config = json.loads((source / "config.json").read_text())
        config["stages"]["mesh"]["save_raw_sdf"] = str(raw_sdf)
        config["stages"]["mesh"]["sp_fea_step_size"] = args.sp_fea_step_size
        save(config_path, config)
        shutil.copy2(descriptor_reference, out / "descriptor_reference.npz")
    protocol = {
        "purpose": "Test realization-calibrated iso-level repair on one frozen RA-QD candidate",
        "source_case": str(source.resolve()),
        "source_id": args.source_id,
        "style": style,
        "seed": source_result["seed"],
        "conditioning": str(conditioning.resolve()),
        "config": str(config_path),
        "raw_sdf": str(raw_sdf),
        "sp_fea_step_size": args.sp_fea_step_size,
        "target_final_volume_fraction": .5,
        "acceptance": {"absolute_volume_error_max": .025, "watertight": True,
                       "components": 1, "containment_fraction_min": .99},
    }
    save(out / "protocol.json", protocol)
    env = {**os.environ, "DATA_ROOT": str(ROOT), "D3DS2_ROOT": str(ROOT),
           "EXP_ROOT": str(ROOT / "experiments"), "D3DS2_PY": sys.executable,
           "FENICS_PY": "/home/goya/miniconda3/envs/fenics/bin/python",
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
           "PYTHONUNBUFFERED": "1", "FEA_WORK_DIR": str(out / "fea_work"),
           "PYVISTA_OFF_SCREEN": "true"}

    if not raw_sdf.exists() or not (original / "final.obj").exists():
        run([sys.executable, CODEBASE / "run_conditioning_case.py", "--config", config_path,
             "--conditioning", conditioning, "--out", original], out / "original.log", env)
    if not (original / "metrics/metrics.json").exists():
        run([sys.executable, CODEBASE / "evaluate_conditioning_case.py", "--mesh", original / "final.obj",
             "--conditioning", conditioning, "--out", original / "metrics",
             "--domain-dir", ROOT / "data_real/bracket", "--size", "384", "--pitch-mm", "1.0"],
            original / "metrics/evaluate.log", env)

    repaired.mkdir(parents=True, exist_ok=True)
    if not (repaired / "mesh.obj").exists():
        run([sys.executable, CODEBASE / "repair_raw_sdf_volume.py",
             "--raw-sdf", raw_sdf, "--descriptor-reference", out / "descriptor_reference.npz",
             "--out", repaired / "mesh.obj", "--report", repaired / "repair.json"],
            out / "repair.log", env)
    if not (repaired / "hybrid.obj").exists():
        run([sys.executable, CODEBASE / "code/post_hybrid_union_clip.py", "--config", config_path,
             "--in", repaired / "mesh.obj", "--out", repaired / "hybrid.obj"],
            repaired / "post.log", env)
    if not (repaired / "final.obj").exists():
        run([sys.executable, CODEBASE / "code/surface_remesh_pre.py", "--config", config_path,
             "--in", repaired / "hybrid.obj", "--out", repaired / "final.obj"],
            repaired / "remesh.log", env)
    if not (repaired / "fea/fea_tet_summary.json").exists():
        run([env["FENICS_PY"], CODEBASE / "code/fea_prep_and_run.py", "--config", config_path,
             "--in", repaired / "final.obj", "--out-dir", repaired / "fea"],
            repaired / "fea/fea.log", env)
    if not (repaired / "metrics/metrics.json").exists():
        run([sys.executable, CODEBASE / "evaluate_conditioning_case.py", "--mesh", repaired / "final.obj",
             "--conditioning", conditioning, "--out", repaired / "metrics",
             "--domain-dir", ROOT / "data_real/bracket", "--size", "384", "--pitch-mm", "1.0"],
            repaired / "metrics/evaluate.log", env)

    before = measurements(original, out / "descriptor_reference.npz")
    after = measurements(repaired, out / "descriptor_reference.npz")
    study_protocol = json.loads((SOURCE_STUDY / "protocol.json").read_text())
    for row in (before, after):
        row["descriptors"] = [row["void_clearance_mean_mm"], row["material_anisotropy"]]
        cell = cell_index(row["descriptors"], study_protocol["dims"],
                          study_protocol["descriptor_ranges"])
        row["archive_cell"] = list(cell) if cell is not None else None
    summary = {
        "status": "complete", "protocol": protocol,
        "original": before, "repaired": after,
        "previous_run_same_recipe": {
            "result": str((source / "result.json").resolve()),
            "material_volume_fraction": source_result["measured_design_volume_fraction"],
            "compliance_J": source_result["compliance_J"],
            "note": "The sparse attention path warns that some CUDA operations remain nondeterministic."
        },
        "changes": {
            "absolute_volume_error_before": abs(before["material_volume_fraction"] - .5),
            "absolute_volume_error_after": abs(after["material_volume_fraction"] - .5),
            "compliance_percent": 100 * (after["compliance_J"] / before["compliance_J"] - 1),
            "volume_constraint_pass_before": abs(before["material_volume_fraction"] - .5) <= .025,
            "volume_constraint_pass_after": abs(after["material_volume_fraction"] - .5) <= .025,
            "archive_cell_preserved": before["archive_cell"] == after["archive_cell"],
        },
    }
    save(out / "summary.json", summary)
    print(out / "summary.json")


if __name__ == "__main__":
    main()
