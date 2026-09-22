"""Recover exact strain-energy concentration from stored tet displacement VTUs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import meshio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from behavior_descriptors import linear_tet_strain_energy, strain_energy_concentration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--E-GPa", type=float, default=110.0)
    parser.add_argument("--nu", type=float, default=.30)
    args = parser.parse_args()
    source, out = args.summary.resolve(), args.out.resolve()
    rows = [row for row in json.loads(source.read_text())["results"] if row.get("valid")]
    with np.load(source.parent / "descriptor_reference.npz") as reference:
        xyz = reference["xyz"]
        characteristic_length_mm = float(np.linalg.norm(xyz.max(0)-xyz.min(0))*1000)
    recovered = []
    for index, row in enumerate(rows, 1):
        path = Path(row["case_dir"]) / "gen/fea/tet_u.vtu"
        mesh = meshio.read(path)
        energy, volume = linear_tet_strain_energy(
            mesh.points, mesh.cells_dict["tetra"], mesh.point_data["u_xyz_m"],
            args.E_GPa*1e9, args.nu)
        recovered_compliance = float(energy.sum())
        recovered.append({
            **row,
            "realized_descriptors": [
                row["features"]["void_clearance_mean_mm"] / characteristic_length_mm,
                strain_energy_concentration(energy, volume),
            ],
            "descriptor_names": ["normalized_void_scale", "strain_energy_concentration"],
            "recovered_compliance_J": recovered_compliance,
            "recovered_to_reported_compliance_ratio": recovered_compliance/row["compliance_J"],
        })
        print(f"{index}/{len(rows)} {row['id']}", flush=True)
    payload = {
        "status": "development_warm_start_only",
        "source": str(source), "samples": len(recovered),
        "descriptor_names": ["normalized_void_scale", "strain_energy_concentration"],
        "characteristic_length_mm": characteristic_length_mm,
        "recovery_ratio_median": float(np.median([r["recovered_to_reported_compliance_ratio"] for r in recovered])),
        "recovery_ratio_max_abs_error": float(np.max(np.abs(np.array([r["recovered_to_reported_compliance_ratio"] for r in recovered])-1))),
        "results": recovered,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")
    print(out)


if __name__ == "__main__":
    main()
