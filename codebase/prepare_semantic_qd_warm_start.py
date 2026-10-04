#!/usr/bin/env python3
"""Import existing bracket meshes as a shared Semantic BO-QD warm start and run FEA."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent
DEFAULT_SOURCE = DATA_ROOT / "experiments/bracket/image_qd_2026-09-22/image_qd_verified.json"
DEFAULT_EXPERIMENT = DATA_ROOT / "experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22"
FENICS_PY = Path("/home/goya/miniconda3/envs/fenics/bin/python")
NICHE = {
    "arch_tie": "arch_tie",
    "triangular_truss": "diagonal_truss",
    "fan_rib": "radial_fan",
    "organic_branching": "branching",
    "y_frame": "branching",
    "twin_spine": "longitudinal_spine",
}


def mass_bin(value: float, edges: list[float]) -> int | None:
    if value < edges[0] or value > edges[-1]:
        return None
    return min(len(edges) - 2, int(np.digitize(value, edges[1:-1])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-fea", action="store_true")
    parser.add_argument("--force-fea", action="store_true")
    args = parser.parse_args()

    protocol = json.loads((args.experiment / "protocol_snapshot.json").read_text())
    records = json.loads(args.source.read_text())
    warm = args.experiment / "shared_warm_start" / f"seed_{args.seed}"
    candidates = warm / "candidates"; candidates.mkdir(parents=True, exist_ok=True)
    envelope = trimesh.load(protocol["absolute_paths"]["design_domain"], force="mesh")
    envelope_mm3 = abs(float(envelope.volume)) * 1e9
    edges = protocol["archive"]["mass_edges"]
    output = []

    for source in records:
        ident = source["archetype"]
        candidate = candidates / ident
        candidate.mkdir(exist_ok=True)
        mesh = candidate / "final.obj"
        source_mesh = Path(source["final_mesh"])
        if not mesh.exists():
            try: mesh.symlink_to(source_mesh.resolve())
            except OSError: shutil.copy2(source_mesh, mesh)
        fea_dir = candidate / "fea"
        summary = fea_dir / "fea_tet_summary.json"
        if args.run_fea and (args.force_fea or not summary.exists()):
            fea_dir.mkdir(exist_ok=True)
            command = [str(FENICS_PY), str(ROOT / "code/fea_prep_and_run.py"),
                       "--config", str(ROOT / "configs/bracket.json"),
                       "--in", str(mesh), "--out-dir", str(fea_dir)]
            with (fea_dir / "fea.log").open("w") as log:
                environment = dict(os.environ, OMP_NUM_THREADS="9",
                                   FENICS_PY=str(FENICS_PY))
                result = subprocess.run(command, cwd=DATA_ROOT, stdout=log,
                                        stderr=subprocess.STDOUT,
                                        env=environment)
            print(f"{ident}: FEA rc={result.returncode}", flush=True)
        fea = json.loads(summary.read_text()) if summary.exists() else None
        normalized_mass = float(source["final_volume_mm3"]) / envelope_mm3
        valid_geometry = bool(source["final_watertight"] and source["final_components"] == 1
                              and source["fix_coverage_1p5mm"] >= .70
                              and source["load_coverage_1p5mm"] >= .60)
        output.append({
            "id": ident, "source_mesh": str(source_mesh.resolve()),
            "mesh": str(mesh.resolve()), "semantic_niche": NICHE[ident],
            "normalized_mass": normalized_mass,
            "mass_bin": mass_bin(normalized_mass, edges),
            "final_volume_mm3": source["final_volume_mm3"],
            "envelope_volume_mm3": envelope_mm3,
            "geometry_gate_pass": valid_geometry,
            "fix_coverage_1p5mm": source["fix_coverage_1p5mm"],
            "load_coverage_1p5mm": source["load_coverage_1p5mm"],
            "fea_complete": fea is not None,
            "compliance_J": fea.get("compliance") if fea else None,
            "vm_max_Pa": fea.get("vm_max") if fea else None,
            "hard_gate_pass": bool(valid_geometry and fea is not None),
        })
    payload = {
        "shared_across_conditions": True,
        "seed": args.seed,
        "mass_normalization": "final volume / design-envelope volume",
        "records": output,
    }
    destination = warm / "warm_start_records.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(destination.resolve())


if __name__ == "__main__":
    main()
