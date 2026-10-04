#!/usr/bin/env python3
"""Training-free, fixed-topology refinement of a connected dense QD mesh."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh
from pysdf import SDF


def load_mesh(path: Path, process: bool = True) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=process)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    if len(mesh.faces) == 0:
        raise ValueError(f"empty mesh: {path}")
    return mesh


def sdf(mesh: trimesh.Trimesh) -> SDF:
    return SDF(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.uint32))


def query(field: SDF, points: np.ndarray) -> np.ndarray:
    return field(np.ascontiguousarray(points, dtype=np.float32), n_threads=4)


def adjacency(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    neighbors = [set() for _ in range(len(mesh.vertices))]
    for a, b in mesh.edges_unique:
        neighbors[int(a)].add(int(b)); neighbors[int(b)].add(int(a))
    return [np.fromiter(v, dtype=np.int64) for v in neighbors]


def laplacian_pass(vertices: np.ndarray, nbrs: list[np.ndarray], fixed: np.ndarray,
                   amount: float) -> np.ndarray:
    updated = vertices.copy()
    for i, ids in enumerate(nbrs):
        if not fixed[i] and len(ids):
            updated[i] += amount * (vertices[ids].mean(axis=0) - vertices[i])
    return updated


def allowed_sdf(points: np.ndarray, fields: tuple[SDF, SDF, SDF]) -> np.ndarray:
    return np.maximum.reduce([query(field, points) for field in fields])


def project_outside(vertices: np.ndarray, allowed_meshes: tuple[trimesh.Trimesh, ...],
                    fields: tuple[SDF, SDF, SDF], tolerance: float) -> tuple[np.ndarray, int]:
    values = allowed_sdf(vertices, fields)
    outside = values < -tolerance
    if not outside.any():
        return vertices, 0
    points = vertices[outside]
    candidates = []
    for mesh in allowed_meshes:
        try:
            closest, _, _ = trimesh.proximity.closest_point(mesh, points)
        except Exception:
            from scipy.spatial import cKDTree
            closest = mesh.vertices[cKDTree(mesh.vertices).query(points)[1]]
        candidates.append(closest)
    stack = np.stack(candidates, axis=1)
    dist2 = ((stack - points[:, None, :]) ** 2).sum(axis=-1)
    nearest = stack[np.arange(len(points)), dist2.argmin(axis=1)]
    result = vertices.copy(); result[outside] = nearest
    return result, int(outside.sum())


def metrics(mesh: trimesh.Trimesh, fields: tuple[SDF, SDF, SDF],
            fix_field: SDF, load_field: SDF, contact_m: float) -> dict:
    v = np.asarray(mesh.vertices)
    components = mesh.split(only_watertight=False)
    allowed = allowed_sdf(v, fields)
    d_fix, d_load = query(fix_field, v), query(load_field, v)
    return {
        "vertices": int(len(v)), "faces": int(len(mesh.faces)),
        "components": int(len(components)), "watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number), "volume_m3": float(abs(mesh.volume)),
        "area_m2": float(mesh.area),
        "outside_vertex_fraction": float((allowed < -1e-5).mean()),
        "fixed_contact_vertices": int((d_fix >= -contact_m).sum()),
        "load_contact_vertices": int((d_load >= -contact_m).sum()),
    }


def render_pair(source: trimesh.Trimesh, refined: trimesh.Trimesh, out: Path) -> None:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv
    from PIL import Image, ImageDraw, ImageFont
    try: pv.start_xvfb()
    except Exception: pass
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    cards = []
    for title, mesh in (("Dense QD scaffold", source), ("Fixed-topology refinement", refined)):
        p = pv.Plotter(off_screen=True, window_size=(600, 480)); p.set_background("#f6f8fa")
        p.add_mesh(pv.wrap(mesh), color="#788b9a", smooth_shading=True,
                   ambient=.25, diffuse=.65, specular=.22, specular_power=22)
        p.enable_parallel_projection(); p.camera_position = "iso"
        image = Image.fromarray(p.screenshot(return_img=True)).convert("RGB"); p.close()
        card = Image.new("RGB", (600, 516), "white"); card.paste(image, (0, 36))
        ImageDraw.Draw(card).text((12, 9), title, font=font, fill="#18232d"); cards.append(card)
    sheet = Image.new("RGB", (1200, 516), "white")
    for i, card in enumerate(cards): sheet.paste(card, (i * 600, 0))
    sheet.save(out)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--envelope", type=Path, required=True)
    ap.add_argument("--fix", type=Path, required=True)
    ap.add_argument("--load", type=Path, required=True)
    ap.add_argument("--edge-mm", type=float, default=2.0)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--lambda-step", type=float, default=.25)
    ap.add_argument("--mu-step", type=float, default=-.26)
    ap.add_argument("--pin-mm", type=float, default=2.0)
    ap.add_argument("--projection-tol-mm", type=float, default=.05)
    args = ap.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    source = load_mesh(args.input.resolve())
    if len(source.split(only_watertight=False)) != 1:
        raise ValueError("fixed-topology refinement requires a single connected input mesh")
    envelope, fix, load = (load_mesh(path.resolve()) for path in (args.envelope, args.fix, args.load))
    env_field, fix_field, load_field = sdf(envelope), sdf(fix), sdf(load)
    fields = (env_field, fix_field, load_field)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(source.vertices, source.faces))
    ms.meshing_isotropic_explicit_remeshing(
        iterations=8, targetlen=pymeshlab.PureValue(args.edge_mm / 1000.0), adaptive=False)
    remeshed = trimesh.Trimesh(ms.current_mesh().vertex_matrix(),
                               ms.current_mesh().face_matrix(), process=True)
    remeshed.export(out / "remeshed.obj")
    if len(remeshed.split(only_watertight=False)) != 1:
        raise RuntimeError("isotropic remesh changed the connected-component topology")

    vertices = np.asarray(remeshed.vertices).copy()
    pin_m = args.pin_mm / 1000.0
    fixed_vertices = np.maximum(query(fix_field, vertices), query(load_field, vertices)) >= -pin_m
    nbrs = adjacency(remeshed)
    for _ in range(args.iterations):
        vertices = laplacian_pass(vertices, nbrs, fixed_vertices, args.lambda_step)
        vertices = laplacian_pass(vertices, nbrs, fixed_vertices, args.mu_step)
        vertices, _ = project_outside(vertices, (envelope, fix, load), fields,
                                      args.projection_tol_mm / 1000.0)
    vertices, projected = project_outside(vertices, (envelope, fix, load), fields, 0.0)
    refined = trimesh.Trimesh(vertices, remeshed.faces, process=False)
    refined.export(out / "refined.obj")

    report = {
        "method": "isotropic remesh + fixed-connectivity BC-pinned Taubin fairing",
        "training": False, "source": str(args.input.resolve()),
        "parameters": {"edge_mm": args.edge_mm, "iterations": args.iterations,
                       "lambda_step": args.lambda_step, "mu_step": args.mu_step,
                       "pin_mm": args.pin_mm},
        "pinned_vertices": int(fixed_vertices.sum()), "projected_vertices_final": projected,
        "source_metrics": metrics(source, fields, fix_field, load_field, pin_m),
        "remeshed_metrics": metrics(remeshed, fields, fix_field, load_field, pin_m),
        "refined_metrics": metrics(refined, fields, fix_field, load_field, pin_m),
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    render_pair(source, refined, out / "comparison.png")
    page = f'''<!doctype html><meta charset="utf-8"><title>Topology-preserving QD refinement</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:32px auto;color:#18232d}}img{{max-width:100%;border:1px solid #d8dfe5}}pre{{background:#f4f6f8;padding:16px;white-space:pre-wrap}}</style>
<h1>Topology-preserving dense QD refinement</h1><p>No model training and no sparse topology generation. BC-near vertices are pinned and mesh connectivity is fixed.</p><img src="comparison.png"><pre>{json.dumps(report, indent=2)}</pre>'''
    (out / "index.html").write_text(page)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
