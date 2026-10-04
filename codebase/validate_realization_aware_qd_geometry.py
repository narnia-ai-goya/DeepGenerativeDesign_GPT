#!/usr/bin/env python3
"""Run the same geometry hard gate on every final mesh in the QD audit archive."""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22"
ARCHIVE = STUDY / "realization_aware_qd_archive_v2/archive.json"
OUT = STUDY / "realization_aware_qd_archive_v2/geometry_audit"
PY = "/home/goya/miniconda3/envs/direct3ds2/bin/python"


def input_dir(row: dict, candidates: dict[str, dict]) -> Path:
    known = candidates.get(row["id"])
    if known:
        return Path(known["model_input"]).parent
    if row["id"].startswith("r5_longitudinal_spine__angular"):
        return STUDY / "shape_language_angular_2026-09-23/input"
    if row["id"].startswith("r5_longitudinal_spine"):
        return STUDY / "shared_image_pool/round_05/novel_semantic/inputs/r5_longitudinal_spine__medium"
    if row["id"].endswith("__qdv2_fea_on"):
        source_id = row["id"].removesuffix("__qdv2_fea_on")
        if source_id in candidates:
            return Path(candidates[source_id]["model_input"]).parent
    raise ValueError(f"Cannot find conditioning images for {row['id']}")


def evaluate(row: dict, candidates: dict[str, dict], out: Path) -> dict:
    target = out / row["id"]
    target.mkdir(parents=True, exist_ok=True)
    metrics = target / "metrics.json"
    if metrics.exists():
        current = json.loads(metrics.read_text())
        mesh_path = Path(row["final_mesh"]).resolve()
        if (current.get("mesh") != str(mesh_path) or
                metrics.stat().st_mtime_ns < mesh_path.stat().st_mtime_ns):
            metrics.unlink()
    if not metrics.exists():
        command = ["xvfb-run", "-a", PY, str(ROOT / "codebase/evaluate_conditioning_case.py"),
                   "--mesh", row["final_mesh"], "--conditioning", str(input_dir(row, candidates)),
                   "--out", str(target), "--domain-dir", str(ROOT / "data_real/bracket"),
                   "--pitch-mm", "0.75"]
        with (target / "evaluation.log").open("w") as stream:
            rc = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
        if rc or not metrics.exists():
            return {"id": row["id"], "error": f"geometry evaluator exited {rc}",
                    "log": str(target / "evaluation.log")}
    result = json.loads(metrics.read_text())
    fix = result["boundary_surface_proximity"]["fix"]["within_1_5mm_fraction"]
    load = result["boundary_surface_proximity"]["load"]["within_1_5mm_fraction"]
    contain = result["voxel"]["containment_fraction_half_voxel_tolerance"]
    thick = result["voxel"]["medial_thickness_p10_mm"]
    passed = bool(result["watertight"] and result["components"] == 1 and
                  fix >= .70 and load >= .60 and contain >= .995 and
                  thick is not None and thick >= 3.0)
    return {"id": row["id"], "final_mesh": row["final_mesh"],
            "geometry_metrics": str(metrics), "geometry_gate_pass": passed,
            "watertight": result["watertight"], "components": result["components"],
            "fix_coverage_1p5mm": fix, "load_coverage_1p5mm": load,
            "inside_envelope_fraction": contain, "thickness_p10_mm": thick}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=ARCHIVE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    rows = json.loads(args.archive.read_text())["records"]
    candidates = {r["id"]: r for file in (STUDY / "shared_image_pool").rglob("candidate_images.json")
                  for r in json.loads(file.read_text())["records"]}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate, row, candidates, out): row["id"] for row in rows}
        for future in as_completed(futures):
            row = future.result()
            results[row["id"]] = row
            print(row["id"], "pass" if row.get("geometry_gate_pass") else "FAIL",
                  row.get("error", ""), flush=True)
            (out.parent / "geometry_gate_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(out.parent / "geometry_gate_results.json")


if __name__ == "__main__":
    main()
