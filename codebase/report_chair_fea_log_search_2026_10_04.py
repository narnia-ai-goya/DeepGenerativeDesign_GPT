"""Report the stage-wise 15 mm FEA-weight search with final-mesh FEA checks."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import trimesh

from make_chair_domain import ROOT
from run_chair_fea_log_search_2026_10_04 import BASE, FACTORS
from run_chair_existing_fea_on_015_2026_10_04 import OUT

import sys
sys.path.insert(0, str(ROOT / "codebase/code/conditioning"))
from cond_render_pv import camera_from_elev_azim, render_lit  # noqa: E402


def collect(case: Path) -> dict | None:
    run_file = case / "run.json"
    if not run_file.exists():
        return None
    run = json.loads(run_file.read_text())
    fea = case / "independent_fea_15mm/summary.json"
    if fea.exists():
        run["independent_fea"] = json.loads(fea.read_text())
    mesh = Path(run["mesh"])
    if mesh.exists():
        shape = trimesh.load(mesh, force="mesh", process=False)
        run["watertight"] = bool(shape.is_watertight)
        run["vertices"] = len(shape.vertices)
    return run


def montage(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    size, header = 420, 36
    canvas = Image.new("RGB", (size * 2, (size + header) * len(rows)), "#f3f6f8")
    draw = ImageDraw.Draw(canvas)
    center = np.array([0., .01, .46])
    spec = json.loads((OUT.parents[4] / "single_view_spec_2026-10-03/specification.json").read_text())
    reg = spec["native_frame_registration"]
    rotation = Rotation.from_euler("x", reg["rotation_x_degrees"], degrees=True).as_matrix()
    for idx, row in enumerate(rows):
        mesh = trimesh.load(row["mesh"], force="mesh", process=False)
        mesh.vertices = ((mesh.vertices - np.asarray(reg["source_center_m"])) @ rotation.T
                         * reg["uniform_scale"] + np.asarray(reg["physical_center_m"]))
        for col, (label, elev, azim) in enumerate((("front", 15, 0), ("iso", 25, 35))):
            eye, up = camera_from_elev_azim(center, 2.0, elev, azim)
            rendered = render_lit(mesh, eye, center, up, size=size,
                                  fit_extent=.56, color=(.56, .61, .66))
            canvas.paste(Image.fromarray(rendered).convert("RGB"),
                         (col * size, idx * (size + header) + header))
            draw.text((col * size + 10, idx * (size + header) + 10),
                      f"factor={row['factor']} | {label}", fill="#1d3441")
    canvas.save(path)


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    dense = [r for f in FACTORS if (r := collect(BASE / f"dense_f{f:05d}"))]
    sparse = []
    for case in sorted(BASE.glob("dense_f*_sparse_f*")):
        row = collect(case)
        if row:
            sparse.append(row)
    (BASE / "summary.json").write_text(json.dumps({"dense": dense, "sparse": sparse}, indent=2) + "\n")
    for stage, rows in (("dense", dense), ("sparse", sparse)):
        successful = [r for r in rows if r["exit_code"] == 0 and Path(r["mesh"]).exists()]
        montage(successful, BASE / f"{stage}_meshes.png")
        evaluated = [r for r in rows if "independent_fea" in r]
        if evaluated:
            fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
            ax.plot([r["factor"] for r in evaluated],
                    [r["independent_fea"]["compliance"] / 1e6 for r in evaluated],
                    "o-", color="#a65c42")
            ax.set_xscale("symlog", linthresh=10)
            ax.set_xticks(FACTORS)
            ax.set_xticklabels([str(x) for x in FACTORS])
            ax.set_xlabel("FEA weight multiplier")
            ax.set_ylabel("Independent 15 mm compliance (×10⁶) ↓")
            ax.grid(alpha=.25)
            fig.savefig(BASE / f"{stage}_trend.png", dpi=170, facecolor="white")
            plt.close(fig)
    def table(rows: list[dict]) -> str:
        cells = []
        for r in rows:
            fea = r.get("independent_fea", {})
            vol = f"{fea['solid_volume_liters']:.3f}" if fea else "—"
            comp = f"{fea['compliance']/1e6:.3f}" if fea else "—"
            valid = "yes" if r.get("watertight") else "no"
            obj = f"<a href='{html.escape(os.path.relpath(r['mesh'], BASE))}'>OBJ</a>" if Path(r["mesh"]).exists() else "—"
            cells.append(f"<tr><td>{r['factor']}</td><td>{r['fea_weight']:.3g}</td>"
                         f"<td>{r['fea_hits']}</td><td>{r['elapsed_seconds']/60:.1f}</td>"
                         f"<td>{valid}</td><td>{vol}</td><td>{comp}</td><td>{r['exit_code']}</td><td>{obj}</td></tr>")
        return "".join(cells)
    def block(stage: str, rows: list[dict]) -> str:
        if not rows:
            return f"<section><h2>{stage.title()}</h2><p>Runs in progress.</p></section>"
        grid = ("<table><tr><th>Multiplier</th><th>Raw FEA coefficient</th><th>In-loop FEA calls</th>"
                "<th>Generation min</th><th>Watertight</th><th>FEM volume L</th><th>C ×10⁶ ↓</th><th>Exit</th><th>OBJ</th></tr>"
                + table(rows) + "</table>")
        images = ""
        if (BASE / f"{stage}_trend.png").exists():
            images += f"<img src='{stage}_trend.png'>"
        if (BASE / f"{stage}_meshes.png").exists():
            images += f"<img src='{stage}_meshes.png'>"
        return f"<section><h2>{stage.title()}</h2>{grid}{images}</section>"
    source = "chair_015_msh_transparent.png"
    overlay = "chair_original_overlaid_on_015_msh_transparent.png"
    findings = []
    for stage, rows in (("Dense", dense), ("Sparse", sparse)):
        evaluated = [r for r in rows if r.get("watertight") and "independent_fea" in r]
        baseline = next((r for r in evaluated if r["factor"] == 0), None)
        if evaluated and baseline:
            best = min(evaluated, key=lambda r: r["independent_fea"]["compliance"])
            base_fea, best_fea = baseline["independent_fea"], best["independent_fea"]
            delta_c = (best_fea["compliance"] / base_fea["compliance"] - 1) * 100
            delta_v = (best_fea["solid_volume_liters"] / base_fea["solid_volume_liters"] - 1) * 100
            findings.append(f"<p>{stage}: best valid tested factor <b>{best['factor']}</b>; "
                            f"C {best_fea['compliance']/1e6:.3f}×10⁶ "
                            f"({delta_c:+.2f}% vs factor 0), FEM volume "
                            f"{best_fea['solid_volume_liters']:.3f} L "
                            f"({delta_v:+.2f}% vs factor 0).</p>")
        invalid = [r["factor"] for r in rows if r["exit_code"] == 0 and
                   Path(r["mesh"]).exists() and not r.get("watertight")]
        if invalid:
            findings.append(f"<p>{stage} invalid final OBJ (not watertight): "
                            f"{', '.join(map(str, invalid))}; no independent compliance assigned.</p>")
    page = f"""<!doctype html><html lang='ko'><meta charset='utf-8'>
<title>Chair FEA log-weight search</title><style>
body{{font:16px/1.5 system-ui;max-width:1080px;margin:25px auto;background:#f3f6f8;color:#1b3340}}
section{{background:white;padding:20px;margin:18px 0;border-radius:12px}}
table{{width:100%;border-collapse:collapse}}td,th{{border:1px solid #d2dfe5;padding:7px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}img{{max-width:100%}}
</style><h1>Chair FEA weight search</h1><section><p>Same image, seed, 15 mm FEM grid,
800 N −Z seat + 200 N +Y back loads. Multiplier ∈ {{0, 10, 100, 1000, 10000}};
Dense fea_w = multiplier × 10⁻⁷, Sparse sp_fea_w = multiplier × 8×10⁻⁵.
Sparse geometric Dense-core guidance is fixed at effective weight 60.</p>
<p>Dense and Sparse are searched stage-wise. C is independently recomputed from the final OBJ
on the same 15 mm FEM grid; lower C is better only at comparable solid volume.</p>
<p><a href='{source}'>Transparent FEM mesh</a> · <a href='{overlay}'>Original chair overlay</a> ·
<a href='summary.json'>Full numeric data</a></p></section>
<section><h2>Finding</h2>{''.join(findings)}</section>
{block('dense', dense)}{block('sparse', sparse)}</html>"""
    (BASE / "index.html").write_text(page)
    print(BASE / "index.html")


if __name__ == "__main__":
    main()
