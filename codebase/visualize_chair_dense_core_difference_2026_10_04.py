"""Show interior material added by Dense-core guidance on common 128^3 sections."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from pysdf import SDF
import trimesh

from run_chair_existing_fea_on_015_2026_10_04 import OUT, SPEC


BASE = OUT / "diagnostics/dense_core_pilot"
MESHES = {
    "Baseline": OUT / "diagnostics/same_dense_sparse_fea_off/generation/mesh.obj",
    "Core 6 mm, FEA OFF": BASE / "core_6mm_w30_fea_off/generation/mesh.obj",
    "Core 6 mm, FEA ON": BASE / "core_6mm_w30_fea_on/generation/mesh.obj",
}


def occupancy(path: Path, origin: np.ndarray, pitch: np.ndarray, resolution: int) -> np.ndarray:
    mesh = trimesh.load(path, force="mesh", process=False)
    sdf = SDF(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    out = np.zeros((resolution, resolution, resolution), dtype=bool)
    y = origin[1] + (np.arange(resolution, dtype=np.float32) + .5) * pitch[1]
    z = origin[2] + (np.arange(resolution, dtype=np.float32) + .5) * pitch[2]
    yy, zz = np.meshgrid(y, z, indexing="ij")
    for x in range(resolution):
        xx = np.full(yy.size, origin[0] + (x + .5) * pitch[0], dtype=np.float32)
        points = np.column_stack((xx, yy.ravel(), zz.ravel()))
        out[x] = (sdf(points) > 0).reshape(resolution, resolution)
    return out


def main() -> None:
    frame = np.load(SPEC / "native_frame_spec.npz")
    resolution = 128
    origin = frame["origin"]
    pitch = frame["pitch_xyz"] * 64 / resolution
    occ = {name: occupancy(path, origin, pitch, resolution) for name, path in MESHES.items()}
    old = occ["Baseline"]
    new = occ["Core 6 mm, FEA ON"]
    additions = new & ~old
    removals = old & ~new
    # Choose the same physical planes for all models, at the greatest changed area.
    planes = [("X-Y, constant Z", 2), ("X-Z, constant Y", 1), ("Y-Z, constant X", 0)]
    indices = {axis: int(np.argmax(additions.sum(axis=tuple(i for i in range(3) if i != axis))))
               for _, axis in planes}
    cmap = ListedColormap(["#ffffff", "#697580", "#e34a33", "#338ab8"])
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    for row, (plane_name, axis) in enumerate(planes):
        index = indices[axis]
        select = [slice(None)] * 3
        select[axis] = index
        select = tuple(select)
        for col, name in enumerate(MESHES):
            image = np.asarray(occ[name][select]).T
            axes[row, col].imshow(image, origin="lower", cmap=ListedColormap(["#ffffff", "#5d6b78"]),
                                  vmin=0, vmax=1, interpolation="nearest")
            axes[row, col].set_title(name, fontsize=12)
        overlay = np.zeros_like(old[select], dtype=np.uint8)
        overlay[old[select] & new[select]] = 1
        overlay[additions[select]] = 2
        overlay[removals[select]] = 3
        axes[row, 3].imshow(overlay.T, origin="lower", cmap=cmap, vmin=0, vmax=3,
                            interpolation="nearest")
        axes[row, 3].set_title("Difference: orange added, blue removed", fontsize=11)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"{plane_name} @ voxel {index}")
    fig.suptitle("Same Dense cache | baseline Sparse vs Dense-core-guided Sparse", fontsize=16)
    output = BASE / "difference_slices.png"
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)
    common_voxels = int((old & new).sum())
    summary = {
        "resolution": resolution,
        "pitch_native_mm": (pitch * 1000).tolist(),
        "baseline_voxels": int(old.sum()), "core_fea_on_voxels": int(new.sum()),
        "common_voxels": common_voxels,
        "added_voxels": int(additions.sum()), "removed_voxels": int(removals.sum()),
        "net_added_voxels": int(additions.sum() - removals.sum()),
        "slice_indices": indices, "figure": str(output),
    }
    (BASE / "difference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
