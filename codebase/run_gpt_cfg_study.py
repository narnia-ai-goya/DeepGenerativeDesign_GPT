#!/usr/bin/env python3
"""Dense-stage parameter study for GPT bridge conditioning.

Runs a paired CFG x FlowDPS-eta grid with one fixed seed, evaluates every mesh
in the source CAD camera frame, and writes a compact HTML/JSON comparison.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
CODEBASE = ROOT / "codebase"
GEN = CODEBASE / "code" / "generate_with_physics_guidance.py"
EVAL = CODEBASE / "evaluate_conditioning_case.py"
PYTHON = Path("/home/goya/miniconda3/envs/direct3ds2/bin/python")
BASE_CONFIG = ROOT / "experiments/bracket/gpt_image_bridge_validation_2026-09-14/cases/gpt_bridge_loadpath_00/config.json"
CONDITIONING = ROOT / "data/bracket/conditioning/gpt_bridge_arch_v2"
OUT = ROOT / "experiments/bracket/gpt_cfg_parameter_study_2026-09-14"


def case_id(cfg: float, eta: float, steps: int) -> str:
    def tag(x: float) -> str:
        return (f"{x:g}").replace(".", "p")
    return f"cfg{tag(cfg)}_eta{tag(eta)}_s{steps}"


def build_cases() -> list[dict]:
    return [dict(cfg=cfg, eta=eta, dense_steps=50, seed=53000)
            for cfg in (2.0, 5.6, 10.0, 15.0)
            for eta in (0.0, 150.0, 300.0)]


def config_for(case: dict, target: Path) -> None:
    cfg = json.loads(BASE_CONFIG.read_text())
    cfg["seed"] = case["seed"]
    mesh = cfg["stages"]["mesh"]
    mesh.update(cfg=case["cfg"], eta=case["eta"], dense_steps=case["dense_steps"],
                skip_sparse=True, fea_w=0.0)
    target.write_text(json.dumps(cfg, indent=2) + "\n")


def generation_command(case_dir: Path) -> list[str]:
    return [str(PYTHON), str(GEN), "--config", str(case_dir / "config.json"),
            "--target-dir", str(CONDITIONING), "--out", str(case_dir / "gen")]


def run_generation(cases: list[dict], max_parallel: int) -> None:
    pending = list(cases)
    active: list[tuple[subprocess.Popen, object, dict, float]] = []
    gpu_count = min(max_parallel, 8)
    while pending or active:
        while pending and len(active) < gpu_count:
            case = pending.pop(0)
            cid = case["id"]
            case_dir = OUT / "cases" / cid
            case_dir.mkdir(parents=True, exist_ok=True)
            config_for(case, case_dir / "config.json")
            mesh = case_dir / "gen/mesh_dense.obj"
            if mesh.exists():
                print(f"[skip] {cid}", flush=True)
                continue
            (case_dir / "gen").mkdir(parents=True, exist_ok=True)
            log = open(case_dir / "gen/gen.log", "w")
            gpu = len(active) % gpu_count
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), FEA_NORMALIZE="0",
                       D3DS2_PATCH_FEATS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                       DETERMINISTIC="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                       D3DS2_ROOT=str(ROOT))
            env["PYTHONPATH"] = str(ROOT / "external/Direct3D-S2") + (
                os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            p = subprocess.Popen(generation_command(case_dir), cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
            active.append((p, log, case, time.time()))
            print(f"[start gpu={gpu}] {cid}", flush=True)
        time.sleep(1)
        keep = []
        for p, log, case, started in active:
            rc = p.poll()
            if rc is None:
                keep.append((p, log, case, started))
                continue
            log.close()
            mesh = OUT / "cases" / case["id"] / "gen/mesh_dense.obj"
            case["generation_rc"] = rc
            case["generation_seconds"] = time.time() - started
            case["generated"] = rc == 0 and mesh.exists()
            print(f"[done rc={rc} {case['generation_seconds']:.0f}s] {case['id']}", flush=True)
        active = keep


def run_evaluation(cases: list[dict]) -> None:
    for case in cases:
        if not case.get("generated"):
            # A resumed run may not have populated the in-memory status.
            case["generated"] = (OUT / "cases" / case["id"] / "gen/mesh_dense.obj").exists()
        if not case["generated"]:
            continue
        case_dir = OUT / "cases" / case["id"]
        metrics = case_dir / "metrics/metrics.json"
        if not metrics.exists():
            cmd = [str(PYTHON), str(EVAL), "--mesh", str(case_dir / "gen/mesh_dense.obj"),
                   "--conditioning", str(CONDITIONING), "--domain-dir", str(ROOT / "data_real/bracket"),
                   "--out", str(case_dir / "metrics")]
            with open(case_dir / "evaluate.log", "w") as log:
                case["evaluation_rc"] = subprocess.run(cmd, cwd=ROOT, stdout=log,
                                                        stderr=subprocess.STDOUT).returncode
        if metrics.exists():
            d = json.loads(metrics.read_text())
            views = [v for v in d["views"].values() if "input" in v]
            case.update(
                mean_iou=sum(v["input"]["mesh_foreground_iou"] for v in views) / len(views),
                mean_mesh_holes=sum(v["mesh"]["projected_holes_count"] for v in views) / len(views),
                mean_target_holes=sum(v["input"]["projected_holes_count"] for v in views) / len(views),
                mean_solid_fraction=sum(v["mesh"]["solid_fraction_in_cad"] for v in views) / len(views),
                volume_mm3=d["volume_mm3"], vertices=d["vertices"], components=d["components"],
                watertight=d["watertight"],
                preview=str((case_dir / "metrics/final_preview.png").resolve()),
                mesh=str((case_dir / "gen/mesh_dense.obj").resolve()),
                metrics=str(metrics.resolve()))


def render(cases: list[dict]) -> None:
    valid = [c for c in cases if c.get("mean_iou") is not None]
    valid.sort(key=lambda c: (-c["mean_iou"], -c["mean_mesh_holes"]))
    full_root = OUT / "full_validation"
    full = []
    for case_dir in sorted(full_root.glob("*")) if full_root.exists() else []:
        metrics_path = case_dir / "gen/metrics/metrics.json"
        fea_path = case_dir / "gen/fea/fea_tet_summary.json"
        if not metrics_path.exists() or not fea_path.exists():
            continue
        d = json.loads(metrics_path.read_text())
        f = json.loads(fea_path.read_text())
        views = [v for v in d["views"].values() if "input" in v]
        full.append({
            "id": case_dir.name,
            "mean_iou": sum(v["input"]["mesh_foreground_iou"] for v in views) / len(views),
            "mean_mesh_holes": sum(v["mesh"]["projected_holes_count"] for v in views) / len(views),
            "mean_target_holes": sum(v["input"]["projected_holes_count"] for v in views) / len(views),
            "mean_solid_fraction": sum(v["mesh"]["solid_fraction_in_cad"] for v in views) / len(views),
            "components": d["components"], "watertight": d["watertight"],
            "volume_mm3": d["volume_mm3"], "compliance_J": f["compliance"],
            "vm_max_Pa": f["vm_max"],
            "mesh": str((case_dir / "gen/final.obj").resolve()),
            "preview": str((case_dir / "gen/metrics/final_preview.png").resolve()),
            "metrics": str(metrics_path.resolve()), "fea": str(fea_path.resolve()),
        })
    summary = {
        "study": "paired dense CFG x FlowDPS eta",
        "conditioning": str(CONDITIONING.resolve()),
        "base_config": str(BASE_CONFIG.resolve()),
        "fixed": {"seed": 53000, "dense_steps": 50, "skip_sparse": True, "fea_w": 0.0},
        "cases": cases,
        "ranking_by_mean_silhouette_iou": [c["id"] for c in valid],
        "full_validation": full,
        "interpretation": {
            "best_visual_bridge_candidate": "cfg10_eta300",
            "best_compliance_candidate": min(full, key=lambda x: x["compliance_J"])["id"] if full else None,
            "warning": "Silhouette IoU rewards broad filled plates and is not a bridge-style score."
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = "".join(
        f"<tr><td>{html.escape(c['id'])}</td><td>{c['cfg']:g}</td><td>{c['eta']:g}</td>"
        f"<td>{c.get('mean_iou', float('nan')):.3f}</td><td>{c.get('mean_mesh_holes', float('nan')):.2f}</td>"
        f"<td>{c.get('components', 0)}</td><td>{c.get('mean_solid_fraction', float('nan')):.3f}</td><td>{c.get('volume_mm3', float('nan')):.0f}</td></tr>"
        for c in valid)
    cards = "".join(
        f"<article><img src=\"{html.escape(os.path.relpath(c['preview'], OUT))}\"><h3>{html.escape(c['id'])}</h3>"
        f"<p>IoU {c['mean_iou']:.3f} · holes {c['mean_mesh_holes']:.2f}/{c['mean_target_holes']:.2f}<br>"
        f"solid {c['mean_solid_fraction']:.3f} · volume {c['volume_mm3']:.0f} mm³</p></article>"
        for c in valid)
    best = valid[0]["id"] if valid else "없음"
    full_rows = "".join(
        f"<tr><td>{html.escape(c['id'])}</td><td>{c['mean_iou']:.3f}</td><td>{c['mean_mesh_holes']:.2f}/{c['mean_target_holes']:.2f}</td>"
        f"<td>{c['mean_solid_fraction']:.3f}</td><td>{c['compliance_J']*1000:.3f}</td><td>{c['vm_max_Pa']/1e6:.2f}</td></tr>"
        for c in full)
    full_cards = "".join(
        f"<article><img src=\"{html.escape(os.path.relpath(c['preview'], OUT))}\"><h3>{html.escape(c['id'])}</h3>"
        f"<p>IoU {c['mean_iou']:.3f} · holes {c['mean_mesh_holes']:.2f}/{c['mean_target_holes']:.2f}<br>"
        f"C {c['compliance_J']*1000:.3f} mJ · vm {c['vm_max_Pa']/1e6:.2f} MPa</p></article>"
        for c in full)
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPT bridge CFG parameter study</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1250px;margin:32px auto;padding:0 22px;color:#17212b;line-height:1.55}}.note{{background:#eef7f4;border-left:4px solid #177b65;padding:13px 17px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd4da;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef2f4}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-top:20px}}article{{border:1px solid #d5dce2;border-radius:10px;padding:12px}}article img{{width:100%}}code{{overflow-wrap:anywhere}}</style>
<h1>GPT bridge CFG parameter study</h1><p class="note">동일 GPT 6-view와 seed 53000에서 CFG × FlowDPS eta만 바꾼 paired 비교다. 12개 dense study 후 세 후보를 sparse·post·FEA까지 검증했다.</p>
<h2>핵심 결과</h2><p><b>교량 아치가 가장 분명한 후보는 CFG=10, eta=300이다.</b> CFG=2, eta=150은 compliance가 가장 낮지만 이미지의 아치가 거의 사라진다. CFG=15, eta=300은 silhouette IoU가 가장 높지만 넓은 판으로 채워진다. 따라서 IoU 단독 1위({html.escape(best)})를 스타일 1위로 사용하면 안 된다.</p>
<h2>전체 파이프라인 상위 후보</h2><table><tr><th>case</th><th>6-view IoU</th><th>mesh/target holes</th><th>solid fraction</th><th>Compliance mJ</th><th>von Mises MPa</th></tr>{full_rows}</table><div class="cards">{full_cards}</div>
<h2>12-case dense sweep</h2><table><tr><th>case</th><th>CFG</th><th>eta</th><th>6-view IoU</th><th>mesh/target holes</th><th>components</th><th>solid fraction</th><th>volume mm³</th></tr>{rows}</table><img src="comparison_montage.png" style="width:100%"><div class="cards">{cards}</div>
<p><code>{OUT / 'summary.json'}</code><br><code>{OUT / 'report.html'}</code></p></html>'''
    (OUT / "report.html").write_text(page)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-parallel", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    for c in cases:
        c["id"] = case_id(c["cfg"], c["eta"], c["dense_steps"])
    run_generation(cases, a.max_parallel)
    run_evaluation(cases)
    render(cases)
    print((OUT / "report.html").resolve(), flush=True)


if __name__ == "__main__":
    main()
