#!/usr/bin/env python3
"""Sequential geometry gates and independent FEA for one Semantic BO-QD round."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import trimesh


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent
D3D_PY = Path("/home/goya/miniconda3/envs/direct3ds2/bin/python")
FENICS_PY = Path("/home/goya/miniconda3/envs/fenics/bin/python")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path); parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()
    evaluation = args.experiment / "shared_evaluations" / f"round_{args.round:02d}"
    rows = json.loads((evaluation / "final_results.json").read_text())
    protocol = json.loads((args.experiment / "protocol_snapshot.json").read_text())
    envelope = trimesh.load(protocol["absolute_paths"]["design_domain"], force="mesh")
    envelope_mm3 = abs(float(envelope.volume)) * 1e9
    results = []
    for row in rows:
        candidate = Path(row["final_mesh"]).parents[1]
        geometry_dir = candidate / "geometry_evaluation"
        metrics_path = geometry_dir / "metrics.json"
        if not metrics_path.exists():
            command = [str(D3D_PY), str(ROOT / "evaluate_conditioning_case.py"),
                       "--mesh", row["final_mesh"], "--conditioning", str(Path(row["model_input"]).parent),
                       "--out", str(geometry_dir), "--domain-dir", str(DATA_ROOT / "data_real/bracket"),
                       "--pitch-mm", "0.75"]
            with (candidate / "geometry_evaluation.log").open("w") as log:
                rc = subprocess.run(command, cwd=DATA_ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
            print(f"{row['id']}: geometry rc={rc}", flush=True)
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None

        fea_dir = candidate / "fea"
        summary_path = fea_dir / "fea_tet_summary.json"
        if not summary_path.exists():
            fea_dir.mkdir(exist_ok=True)
            command = [str(FENICS_PY), str(ROOT / "code/fea_prep_and_run.py"),
                       "--config", str(ROOT / "configs/bracket.json"),
                       "--in", row["final_mesh"], "--out-dir", str(fea_dir)]
            environment = dict(os.environ, OMP_NUM_THREADS="9", FENICS_PY=str(FENICS_PY))
            with (fea_dir / "fea.log").open("w") as log:
                rc = subprocess.run(command, cwd=DATA_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env=environment).returncode
            print(f"{row['id']}: FEA rc={rc}", flush=True)
        fea = json.loads(summary_path.read_text()) if summary_path.exists() else None

        fix = metrics["boundary_surface_proximity"]["fix"]["within_1_5mm_fraction"] if metrics else 0
        load = metrics["boundary_surface_proximity"]["load"]["within_1_5mm_fraction"] if metrics else 0
        contain = metrics["voxel"]["containment_fraction_half_voxel_tolerance"] if metrics else 0
        thickness = metrics["voxel"]["medial_thickness_p10_mm"] if metrics else 0
        geometry_pass = bool(metrics and metrics["watertight"] and metrics["components"] == 1
                             and fix >= .70 and load >= .60 and contain >= .995
                             and thickness is not None and thickness >= 3.0)
        results.append({**row, "normalized_mass": row["final_volume_mm3"] / envelope_mm3,
                        "geometry_metrics": str(metrics_path.resolve()) if metrics else None,
                        "fix_coverage_1p5mm": fix, "load_coverage_1p5mm": load,
                        "inside_envelope_fraction": contain, "thickness_p10_mm": thickness,
                        "geometry_gate_pass": geometry_pass, "fea_complete": fea is not None,
                        "compliance_J": fea.get("compliance") if fea else None,
                        "vm_max_Pa": fea.get("vm_max") if fea else None,
                        "hard_gate_pass": bool(geometry_pass and fea is not None)})
    destination = evaluation / "verified_results.json"
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(destination.resolve())


if __name__ == "__main__": main()
