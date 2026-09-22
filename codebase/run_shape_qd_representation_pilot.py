#!/usr/bin/env python
"""Build a frozen geometry-only shape space from existing bracket meshes.

This is representation validation, not a QD generation run.  It deliberately
uses no FEA signal as a descriptor: a mesh is represented only by occupancy
projections after fixed/load BC voxels have been masked out.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import trimesh
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "experiments/bracket/shape_qd_representation_pilot_2026-09-21"


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("not a nonempty triangle mesh")
    return mesh


def points_to_grid(points: np.ndarray, lo: np.ndarray, hi: np.ndarray, resolution: int) -> np.ndarray:
    """Rasterize voxel centers into a fixed world-coordinate occupancy grid."""
    grid = np.zeros((resolution, resolution, resolution), dtype=bool)
    scale = (resolution - 1) / np.maximum(hi - lo, 1e-12)
    idx = np.rint((points - lo) * scale).astype(int)
    valid = np.all((idx >= 0) & (idx < resolution), axis=1)
    idx = idx[valid]
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


def voxel_grid(mesh: trimesh.Trimesh, lo: np.ndarray, hi: np.ndarray, resolution: int) -> np.ndarray:
    pitch = float(np.max(hi - lo) / (resolution - 1))
    voxels = mesh.voxelized(pitch).fill()
    return points_to_grid(voxels.points, lo, hi, resolution)


def geometry_feature(occ: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Multi-view orthographic silhouettes plus normalized ray depth.

    The views are all geometry-only.  A two-channel map is retained per view:
    silhouette and mean occupied depth along its viewing ray.
    """
    maps: dict[str, np.ndarray] = {}
    features = []
    for name, axis in (("top", 2), ("front", 1), ("right", 0)):
        silhouette = occ.any(axis=axis).astype(np.float32)
        depth_axis = np.arange(occ.shape[axis], dtype=np.float32)
        shape = [1, 1, 1]
        shape[axis] = occ.shape[axis]
        weighted = occ * depth_axis.reshape(shape)
        count = occ.sum(axis=axis)
        depth = weighted.sum(axis=axis) / np.maximum(count, 1)
        depth = (depth / max(1, occ.shape[axis] - 1)) * silhouette
        maps[name] = np.stack([silhouette, depth], axis=0)
        features.append(maps[name].reshape(-1))
    return np.concatenate(features).astype(np.float32), maps


def fea_compliance(mesh_path: Path) -> float | None:
    candidates = [
        mesh_path.parent / "fea" / "fea_tet_summary.json",
        mesh_path.parents[1] / "fea" / "fea_tet_summary.json",
        mesh_path.parent / "metrics" / "fea_tet_summary.json",
    ]
    for path in candidates:
        try:
            value = float(json.loads(path.read_text())["compliance"])
            if np.isfinite(value) and value > 0:
                return value
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            continue
    return None


def render_representative(mesh_path: Path, out: Path) -> None:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv

    try:
        pv.start_xvfb()
    except Exception:
        pass
    mesh = load_mesh(mesh_path)
    plotter = pv.Plotter(off_screen=True, window_size=(480, 360))
    plotter.set_background("#f7f8fa")
    plotter.add_mesh(pv.wrap(mesh), color="#71859a", smooth_shading=True,
                     specular=0.30, specular_power=24)
    plotter.enable_parallel_projection()
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.18)
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out))
    plotter.close()


def write_html(out: Path, rows: list[dict], summary: dict) -> Path:
    cards = []
    for row in rows:
        img = row.get("thumbnail", "")
        cards.append(f"""<article><img src=\"{img}\" alt=\"niche {row['niche']}\">
<h2>Niche {row['niche']}</h2><p>{row['members']} bank meshes<br>
PCA: ({row['pca_2d'][0]:.2f}, {row['pca_2d'][1]:.2f})<br>
Compliance: {row['compliance_label']}</p><code>{row['mesh']}</code></article>""")
    page = f"""<!doctype html><meta charset=\"utf-8\"><title>Bracket shape-QD representation pilot</title>
<style>body{{font:15px system-ui;margin:32px;background:#f7f8fa;color:#18212b}} main{{max-width:1280px;margin:auto}} .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}} article{{background:white;border:1px solid #dde3e8;border-radius:10px;overflow:hidden;padding-bottom:14px}} img{{width:100%;display:block;background:#edf1f4}} h2,p,code{{margin:12px 14px}} code{{display:block;overflow-wrap:anywhere;color:#506070;font-size:11px}} .meta{{background:white;border-radius:10px;padding:16px;margin:16px 0;white-space:pre-wrap}}</style>
<main><h1>Frozen geometry-only shape space — bracket</h1><p>Representation pilot only. Diversity is multi-view mesh geometry; FEA is shown only as quality metadata.</p>
<div class=\"meta\">{json.dumps(summary, indent=2)}</div><div class=\"grid\">{''.join(cards)}</div></main>"""
    path = out / "index.html"
    path.write_text(page)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--resolution", type=int, default=48)
    ap.add_argument("--niches", type=int, default=12)
    ap.add_argument("--max-meshes", type=int, default=0, help="0 means all discovered meshes")
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    if args.resolution < 16 or args.niches < 2:
        raise ValueError("resolution must be >=16 and niches >=2")

    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    domain = load_mesh(ROOT / "data_real/bracket/original_DesignSpace.stl")
    lo, hi = domain.bounds
    fixed = voxel_grid(load_mesh(ROOT / "data_real/bracket/fixed.stl"), lo, hi, args.resolution)
    load = voxel_grid(load_mesh(ROOT / "data_real/bracket/load.stl"), lo, hi, args.resolution)
    bc_mask = fixed | load

    candidates = sorted(ROOT.glob("experiments/bracket/**/final.obj"))
    if args.max_meshes:
        rng = np.random.default_rng(args.seed)
        selected = sorted(rng.choice(candidates, size=min(args.max_meshes, len(candidates)), replace=False))
    else:
        selected = candidates
    print(f"shape-QD representation: {len(selected)} candidate meshes, R={args.resolution}", flush=True)

    feats, records, preview_maps = [], [], []
    for index, path in enumerate(selected, 1):
        try:
            mesh = load_mesh(path)
            occ = voxel_grid(mesh, lo, hi, args.resolution)
            occ[bc_mask] = False
            if occ.sum() < 16:
                raise ValueError("empty geometry after BC mask")
            feat, maps = geometry_feature(occ)
            feats.append(feat)
            records.append({"mesh": str(path.resolve()), "faces": int(len(mesh.faces)),
                            "watertight": bool(mesh.is_watertight), "feature_voxels": int(occ.sum()),
                            "compliance_J": fea_compliance(path)})
            preview_maps.append(maps)
        except Exception as exc:
            print(f"  skip {path}: {type(exc).__name__}: {exc}", flush=True)
        if index % 25 == 0 or index == len(selected):
            print(f"  encoded {index}/{len(selected)}; valid={len(records)}", flush=True)
    if len(records) < args.niches:
        raise RuntimeError(f"Only {len(records)} valid meshes for {args.niches} niches")

    x = np.stack(feats).astype(np.float64)
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.0
    normalized = (x - mean) / scale
    n_components = min(8, len(records), normalized.shape[1])
    pca = PCA(n_components=n_components, random_state=args.seed)
    z = pca.fit_transform(normalized)
    archive_space = z / np.maximum(z.std(0, keepdims=True), 1e-6)
    kmeans = KMeans(n_clusters=args.niches, n_init=32, random_state=args.seed)
    labels = kmeans.fit_predict(archive_space)
    centroids = kmeans.cluster_centers_

    np.savez_compressed(out / "frozen_shape_space.npz", feature_mean=mean, feature_scale=scale,
                        pca_components=pca.components_, pca_explained_variance=pca.explained_variance_ratio_,
                        archive_scale=z.std(0), cvt_centroids=centroids, embeddings=z, labels=labels,
                        domain_bounds=np.stack([lo, hi]), bc_mask=bc_mask)
    save_json(out / "protocol.json", {"representation": "48^3 fixed-frame occupancy -> top/front/right silhouette + mean depth",
                                        "geometry_only": True, "bc_mask": "fixed.stl union load.stl",
                                        "resolution": args.resolution, "niches": args.niches,
                                        "encoder": f"frozen PCA-{n_components}", "archive": "KMeans CVT approximation in standardized PCA space",
                                        "seed": args.seed, "candidate_count": len(selected), "valid_count": len(records),
                                        "note": "No QD generation or FEA evaluation ran in this representation pilot."})

    rows = []
    thumbs = out / "thumbnails"
    for niche in range(args.niches):
        members = np.flatnonzero(labels == niche)
        with_fea = [i for i in members if records[i]["compliance_J"] is not None]
        representative = min(with_fea, key=lambda i: records[i]["compliance_J"]) if with_fea else min(
            members, key=lambda i: np.linalg.norm(archive_space[i] - centroids[niche]))
        thumb = thumbs / f"niche_{niche:02d}.png"
        render_representative(Path(records[representative]["mesh"]), thumb)
        row = {"niche": niche, "members": int(len(members)), "representative_index": int(representative),
               "mesh": records[representative]["mesh"], "pca_2d": [float(z[representative, 0]), float(z[representative, 1])],
               "compliance_J": records[representative]["compliance_J"],
               "compliance_label": (f"{records[representative]['compliance_J']:.4g} J" if records[representative]["compliance_J"] else "not available"),
               "thumbnail": str(thumb.relative_to(out))}
        rows.append(row)
    summary = {"candidate_meshes": len(selected), "encoded_meshes": len(records), "niches": args.niches,
               "PCA_components": n_components, "PCA_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
               "representation": "masked multi-view silhouette + depth", "quality_metadata": "available final FEA compliance only"}
    save_json(out / "archive.json", {"summary": summary, "records": records, "niches": rows})
    html = write_html(out, rows, summary)
    print(f"DONE archive={out / 'archive.json'} html={html}", flush=True)


if __name__ == "__main__":
    main()
