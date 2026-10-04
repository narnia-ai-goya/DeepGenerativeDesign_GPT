#!/usr/bin/env python3
"""Training-free, connectivity-screened QD sampling for the proven lr162 bracket recipe.

The Direct3D-S2 sampler is never guided toward a QD target.  Diversity comes from ordinary
posterior samples (different seeds), dense meshes are screened cheaply, and only diverse feasible
representatives are promoted to the expensive sparse refiner.  This keeps every promoted sample on
the model's learned generation manifold.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage
from skimage.morphology import skeletonize


ROOT = Path(__file__).resolve().parents[1]
CODEBASE = ROOT / "codebase"
GENERATOR = CODEBASE / "code/generate_with_physics_guidance.py"
POST = CODEBASE / "code/post_hybrid_union_clip.py"
REMESH = CODEBASE / "code/surface_remesh_pre.py"
PYTHON = Path("/home/goya/miniconda3/envs/direct3ds2/bin/python")
BASE = ROOT / "experiments/bracket/direct3ds2_lowres_2026-09-15/lr162"
DEFAULT_OUT = ROOT / "experiments/bracket/connectivity_qd_sampling_2026-09-22"
SEEDS = (42, 7, 19, 73, 101, 137, 211, 307, 419, 557, 701, 887, 1051, 1291, 1601, 2027)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def make_cases(out: Path) -> list[dict]:
    base = json.loads((BASE / "config.json").read_text())
    rows = []
    for seed in SEEDS:
        case = out / "cases" / f"seed_{seed:04d}"
        gen = case / "gen"
        gen.mkdir(parents=True, exist_ok=True)
        cfg = json.loads(json.dumps(base))
        cfg["name"] = f"connectivity_qd_lr162_seed_{seed}"
        cfg["seed"] = seed
        mesh = cfg["stages"]["mesh"]
        mesh.update({
            "skip_sparse": True,
            "load_dense_cache": None,
            "save_dense_cache": str((case / "dense_cache.npz").resolve()),
            "shape_qd_w": 0.0,
            "shape_anchor_w": 0.0,
            "shape_scaffold_w": 0.0,
            "shape_residual_w": 0.0,
            "shape_transport_radius": 0.0,
            "oc_flow_w": 0.0,
            "sp_shape_qd_w": 0.0,
            "sp_shape_anchor_w": 0.0,
        })
        config = case / "config_dense.json"
        dump(config, cfg)
        rows.append({"seed": seed, "case": case, "gen": gen, "config": config})
    dump(out / "manifest.json", {
        "method": "unchanged lr162 sampler + connectivity screen + descriptor-space selection",
        "base_config": str((BASE / "config.json").resolve()),
        "conditioning": str((BASE / "input").resolve()),
        "seeds": list(SEEDS),
        "latent_qd_guidance": False,
    })
    return rows


def generation_env(gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "FEA_NORMALIZE": "0",
        "FEA_FENICS_BIN": "/home/goya/miniconda3/envs/fenics/bin/python",
        "FEA_FENICS_SCRIPT": str(CODEBASE / "code/fenics_fea_bracket.py"),
        "D3DS2_PATCH_FEATS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "DETERMINISTIC": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": str(ROOT / "external/Direct3D-S2") +
                      (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        "OMP_NUM_THREADS": "6",
        "OPENBLAS_NUM_THREADS": "6",
        "MKL_NUM_THREADS": "6",
        "LOAD_MODE": "-z",
    })
    return env


def generate_one(row: dict, gpu: int, sparse: bool, force: bool) -> dict:
    case, gen = row["case"], row["gen"]
    target = gen / ("mesh.obj" if sparse else "mesh_dense.obj")
    completion_marker = case / ("sparse_complete.json" if sparse else "dense_complete.json")
    if target.exists() and completion_marker.exists() and not force:
        return {"seed": row["seed"], "ok": True, "gpu": gpu, "skipped": True}
    cfg = json.loads(row["config"].read_text())
    if sparse:
        cfg["name"] += "_sparse"
        cfg["stages"]["mesh"].update({
            "skip_sparse": False,
            "load_dense_cache": str((case / "dense_cache.npz").resolve()),
            "save_dense_cache": None,
        })
        config = case / "config_sparse.json"
        dump(config, cfg)
    else:
        config = row["config"]
    log = case / ("sparse.log" if sparse else "dense.log")
    cmd = [str(PYTHON), str(GENERATOR), "--config", str(config),
           "--target-dir", str((BASE / "input").resolve()), "--out", str(gen)]
    with log.open("w") as stream:
        proc = subprocess.run(cmd, cwd=ROOT, env=generation_env(gpu), stdout=stream,
                              stderr=subprocess.STDOUT)
    ok = proc.returncode == 0 and target.exists()
    if ok:
        dump(completion_marker, {"seed": row["seed"], "stage": "sparse" if sparse else "dense",
                                 "mesh": str(target.resolve())})
    return {"seed": row["seed"], "ok": ok,
            "gpu": gpu, "returncode": proc.returncode, "log": str(log.resolve())}


def run_parallel(rows: list[dict], sparse: bool, workers: int, force: bool) -> list[dict]:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(generate_one, row, i % 8, sparse, force): row
                   for i, row in enumerate(rows)}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[{'sparse' if sparse else 'dense'}] seed={result['seed']} "
                  f"gpu={result['gpu']} ok={result['ok']}", flush=True)
    return sorted(results, key=lambda x: x["seed"])


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    return mesh


def raster(mesh: trimesh.Trimesh, lo: np.ndarray, hi: np.ndarray, resolution: int) -> np.ndarray:
    pitch = float(np.max(hi - lo) / (resolution - 1))
    points = mesh.voxelized(pitch).fill().points
    # trimesh voxelization is isotropic.  Keep that same pitch when mapping into the shared
    # domain frame; independently stretching each axis creates artificial one-voxel gaps.
    idx = np.rint((points - lo) / pitch).astype(int)
    good = np.all((idx >= 0) & (idx < resolution), axis=1)
    idx = idx[good]
    occ = np.zeros((resolution,) * 3, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def count_projection_holes(mask: np.ndarray) -> int:
    background = ~mask
    labels, n = ndimage.label(background, structure=np.ones((3, 3), dtype=np.uint8))
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    counts = np.bincount(labels.ravel())
    return int(sum(i not in border and counts[i] >= 3 for i in range(1, n + 1)))


def count_top_branches(mask: np.ndarray) -> int:
    clean = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=1)
    skel = skeletonize(clean)
    degree = ndimage.convolve(skel.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
                              mode="constant") - skel
    clusters, n = ndimage.label(skel & (degree >= 3), structure=np.ones((3, 3), dtype=np.uint8))
    if n == 0:
        return 0
    sizes = np.bincount(clusters.ravel())
    return int(sum(sizes[1:] >= 1))


def shape_feature(occ: np.ndarray) -> np.ndarray:
    maps = []
    for axis in (2, 1, 0):
        sil = occ.any(axis=axis).astype(np.float32)
        zoom = (24 / sil.shape[0], 24 / sil.shape[1])
        maps.append(ndimage.zoom(sil, zoom, order=0).reshape(-1))
    feature = np.concatenate(maps)
    norm = np.linalg.norm(feature)
    return feature / max(norm, 1e-12)


def analyze_dense(rows: list[dict], out: Path, resolution: int = 64) -> list[dict]:
    base_cfg = json.loads((BASE / "config.json").read_text())
    ms = base_cfg["stages"]["mesh"]
    envelope = load_mesh(Path(ms["fea_bracket_stl"]))
    fix = load_mesh(Path(ms["fix_stl"]))
    load = load_mesh(Path(ms["load_stl"]))
    lo, hi = envelope.bounds
    fix_occ = ndimage.binary_dilation(raster(fix, lo, hi, resolution), iterations=1)
    load_occ = ndimage.binary_dilation(raster(load, lo, hi, resolution), iterations=1)
    records = []
    for row in rows:
        path = row["gen"] / "mesh_dense.obj"
        if not path.exists():
            continue
        mesh = load_mesh(path)
        parts = mesh.split(only_watertight=False)
        face_sizes = sorted((len(p.faces) for p in parts), reverse=True)
        main_fraction = face_sizes[0] / max(1, sum(face_sizes))
        occ = raster(mesh, lo, hi, resolution)
        labels, ncomp = ndimage.label(occ, structure=np.ones((3, 3, 3), dtype=np.uint8))
        fixed_labels = set(np.unique(labels[fix_occ & occ])) - {0}
        load_labels = set(np.unique(labels[load_occ & occ])) - {0}
        bc_connected = bool(fixed_labels & load_labels)
        top = occ.any(axis=2)
        holes = count_projection_holes(top)
        branches = count_top_branches(top)
        feature = shape_feature(occ)
        records.append({
            "seed": row["seed"], "mesh": str(path.resolve()), "vertices": len(mesh.vertices),
            "faces": len(mesh.faces), "mesh_components": len(parts),
            "main_face_fraction": float(main_fraction), "voxel_components": int(ncomp),
            "bc_connected": bc_connected, "top_branches": branches,
            "top_holes": holes, "top_fill": float(top.mean()),
            "volume_mm3": float(abs(mesh.volume) * 1e9), "_feature": feature,
        })
    return records


def select_diverse(records: list[dict], count: int) -> list[dict]:
    feasible = [r for r in records if r["bc_connected"] and r["main_face_fraction"] >= 0.98]
    if not feasible:
        feasible = sorted(records, key=lambda r: (r["bc_connected"], r["main_face_fraction"]), reverse=True)
    # One best representative per interpretable QD cell first.
    cells: dict[tuple[int, int], list[dict]] = {}
    for r in feasible:
        cell = (min(r["top_branches"], 5), min(r["top_holes"], 6))
        cells.setdefault(cell, []).append(r)
    pool = [max(items, key=lambda r: r["main_face_fraction"]) for items in cells.values()]
    # Baseline seed 42 is the control anchor. Fill remaining slots by greedy max-min silhouette distance.
    anchor = next((r for r in feasible if r["seed"] == 42), feasible[0])
    selected = [anchor]
    candidates = [r for r in pool if r is not anchor] + [r for r in feasible if r not in pool and r is not anchor]
    while candidates and len(selected) < min(count, len(feasible)):
        def novelty(r: dict) -> float:
            return min(float(np.linalg.norm(r["_feature"] - s["_feature"])) for s in selected)
        best = max(candidates, key=lambda r: (novelty(r), r["main_face_fraction"]))
        selected.append(best)
        candidates.remove(best)
    return selected


def render_mesh(path: Path, image: Path, title: str) -> None:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv
    try:
        pv.start_xvfb()
    except Exception:
        pass
    mesh = load_mesh(path)
    pl = pv.Plotter(off_screen=True, window_size=(600, 430))
    pl.set_background("#f4f6f8")
    pl.add_mesh(pv.wrap(mesh), color="#8b98a5", smooth_shading=True, specular=0.35,
                specular_power=28)
    pl.add_text(title, font_size=11, color="#17212b")
    pl.enable_parallel_projection()
    pl.camera_position = "iso"
    pl.camera.zoom(1.15)
    image.parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(str(image))
    pl.close()


def write_report(records: list[dict], selected: list[dict], out: Path, stage: str) -> Path:
    assets = out / "report_assets"
    selected_seeds = {r["seed"] for r in selected}
    cards = []
    for r in records:
        mesh = Path(r.get("final_mesh") or r["mesh"])
        image = assets / f"seed_{r['seed']:04d}_{stage}.png"
        render_mesh(mesh, image, f"seed {r['seed']} | branches {r['top_branches']} | holes {r['top_holes']}")
        status = "SELECTED" if r["seed"] in selected_seeds else "screened"
        cards.append(f"<article class='{status.lower()}'><img src='{image.relative_to(out)}'>"
                     f"<h3>seed {r['seed']} · {status}</h3><p>BC path: {r['bc_connected']} · "
                     f"main: {100*r['main_face_fraction']:.2f}%<br>branches: {r['top_branches']} · "
                     f"holes: {r['top_holes']} · fill: {r['top_fill']:.3f}</p><code>{mesh}</code></article>")
    page = f"""<!doctype html><html lang='ko'><meta charset='utf-8'><title>Connectivity-screened QD</title>
<style>body{{font:15px system-ui;margin:30px;background:#f5f7f9;color:#17212b}}main{{max-width:1450px;margin:auto}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:15px}}article{{background:white;border:1px solid #d9e0e6;border-radius:10px;overflow:hidden;padding-bottom:12px}}article.selected{{border:3px solid #188568}}img{{width:100%;display:block}}h3,p,code{{margin:10px 13px}}code{{display:block;font-size:10px;overflow-wrap:anywhere}}</style>
<main><h1>Connectivity-screened QD — {stage}</h1><p>기존 lr162 sampler를 변경하지 않았다. QD latent guidance와 model training은 사용하지 않았다.</p><div class='grid'>{''.join(cards)}</div></main></html>"""
    report = out / f"index_{stage}.html"
    report.write_text(page)
    return report


def post_selected(selected: list[dict], rows_by_seed: dict[int, dict], force: bool) -> list[dict]:
    # lr162 generation and its legacy post assets use two calibrated physical frames.  Recover the
    # exact affine map from the reference mesh pair (same vertex order; residual < 10 nm) instead
    # of approximating it from bounding boxes.
    ref_raw = load_mesh(BASE / "generation/mesh.obj")
    ref_aligned = load_mesh(BASE / "generation/mesh_physical_aligned.obj")
    sample = np.arange(0, len(ref_raw.vertices), 64)
    transform = np.linalg.lstsq(
        np.c_[ref_raw.vertices[sample], np.ones(len(sample))],
        ref_aligned.vertices[sample], rcond=None)[0]
    done = []
    for record in selected:
        row = rows_by_seed[record["seed"]]
        gen = row["gen"]
        final = gen / "final.obj"
        if force or not final.exists():
            cfg = row["case"] / "config_sparse.json"
            raw = load_mesh(gen / "mesh.obj")
            aligned = trimesh.Trimesh(
                np.c_[raw.vertices, np.ones(len(raw.vertices))] @ transform,
                raw.faces.copy(), process=False)
            aligned_path = gen / "mesh_physical_aligned.obj"
            aligned.export(aligned_path)
            hybrid = gen / "mesh_bc_preserved.obj"
            cmd = [str(PYTHON), str(POST), "--config", str(cfg), "--in", str(aligned_path),
                   "--out", str(hybrid)]
            with (row["case"] / "post.log").open("w") as stream:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                print(f"[post] seed={record['seed']} failed rc={rc}", flush=True)
                continue
            cmd = [str(PYTHON), str(REMESH), "--config", str(cfg), "--in", str(hybrid),
                   "--out", str(final)]
            with (row["case"] / "remesh.log").open("w") as stream:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                print(f"[remesh] seed={record['seed']} failed rc={rc}", flush=True)
                continue
        record["final_mesh"] = str(final.resolve())
        done.append(record)
    return done


def clean_records(records: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--phase", choices=("dense", "select", "sparse", "post", "all"), default="all")
    ap.add_argument("--dense-workers", type=int, default=8)
    ap.add_argument("--sparse-workers", type=int, default=2)
    ap.add_argument("--promote", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = make_cases(out)
    rows_by_seed = {r["seed"]: r for r in rows}

    if args.phase in ("dense", "all"):
        result = run_parallel(rows, sparse=False, workers=args.dense_workers, force=args.force)
        dump(out / "dense_run_status.json", result)
    records = analyze_dense(rows, out)
    selected = select_diverse(records, args.promote)
    dump(out / "dense_metrics.json", clean_records(records))
    dump(out / "selection.json", {"selected_seeds": [r["seed"] for r in selected],
                                   "records": clean_records(selected)})
    report = write_report(records, selected, out, "dense")
    print(f"[selection] seeds={[r['seed'] for r in selected]} report={report}", flush=True)

    if args.phase in ("sparse", "post", "all"):
        selected_rows = [rows_by_seed[r["seed"]] for r in selected]
        if args.phase in ("sparse", "all"):
            result = run_parallel(selected_rows, sparse=True, workers=args.sparse_workers, force=args.force)
            dump(out / "sparse_run_status.json", result)
        completed = post_selected(selected, rows_by_seed, args.force)
        # Preserve dense descriptors but report the delivered sparse/post meshes.
        report = write_report(records, completed, out, "final")
        dump(out / "final_selection.json", {"selected_seeds": [r["seed"] for r in completed],
                                             "records": clean_records(completed)})
        print(f"[final] completed={len(completed)} report={report}", flush=True)


if __name__ == "__main__":
    main()
