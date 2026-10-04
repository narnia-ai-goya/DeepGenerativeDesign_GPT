"""Visualize measured native-grid-to-physical-FEA alignment for the chair study."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import meshio
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/chair/sofa_style_2026-09-28'
SPEC = (BASE / 'text_reasoned_front_axes_2026-10-03/long_qd_fea_2026-10-03'
        / 'envelope_plus16_spec_2026-10-03')
OUT = (BASE / 'text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03'
       / 'llm_text_proposal_round_03_2026-10-03'
       / 'simultaneous_load_dense_sparse_2026-10-04/diagnostics'
       / 'registration_overlay_2026-10-04.png')


def main() -> None:
    physical = np.load(SPEC / 'voxel.npz')
    native = np.load(SPEC / 'native_frame_spec.npz')
    reg = json.loads((BASE / 'single_view_spec_2026-10-03/specification.json').read_text())[
        'native_frame_registration']
    origin, pitch = physical['origin'], physical['pitch_xyz']
    indices = np.indices((64, 64, 64)).reshape(3, -1).T
    centers = origin + (indices + .5) * pitch
    source, target = np.array(reg['source_center_m']), np.array(reg['physical_center_m'])
    rotation = Rotation.from_euler('x', reg['rotation_x_degrees'], degrees=True).as_matrix()
    scale = reg['uniform_scale']
    to_physical = lambda xyz, s=scale: (xyz - source) @ rotation.T * s + target
    to_native = (centers - target) @ rotation / scale + source
    sample_index = ((to_native - origin) / pitch - .5).T.reshape(3, 64, 64, 64)
    actual = physical['bracket'].astype(bool)
    recovered = map_coordinates(native['bracket'].astype(np.uint8), sample_index,
                                order=0, mode='constant', cval=0).astype(bool)
    missing = actual & ~recovered
    overlap = actual & recovered

    fig = plt.figure(figsize=(16, 10), facecolor='#fbfcfe', layout='constrained')
    gs = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(gs[0, 0])
    # YZ projection; extent offsets restore physical coordinates.
    side_overlap = overlap.any(axis=0)
    side_missing = missing.any(axis=0)
    for mask, color, label in [(side_overlap, '#59aec8', 'Both domains'),
                               (side_missing, '#e25b59', 'Physical envelope only')]:
        iy, iz = np.where(mask)
        ax.scatter((origin[1]+(iy+.5)*pitch[1])*1000,
                   (origin[2]+(iz+.5)*pitch[2])*1000,
                   s=25, marker='s', color=color, linewidths=0, label=label)
    ax.set(title='Side view · envelope coverage', xlabel='Front ← Y (mm) → Back',
           ylabel='Height Z (mm)', ylim=(0, 990))
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(alpha=.15)

    ax = fig.add_subplot(gs[0, 1])
    for mask, color, label in [(overlap.any(axis=1), '#59aec8', 'Both domains'),
                               (missing.any(axis=1), '#e25b59', 'Physical envelope only')]:
        ix, iz = np.where(mask)
        ax.scatter((origin[0]+(ix+.5)*pitch[0])*1000,
                   (origin[2]+(iz+.5)*pitch[2])*1000,
                   s=25, marker='s', color=color, linewidths=0, label=label)
    ax.set(title='Front view · envelope coverage', xlabel='X (mm)',
           ylabel='Height Z (mm)', ylim=(0, 990))
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(alpha=.15)

    ax = fig.add_subplot(gs[1, 0])
    colors = {'fix': '#d17a68', 'load': '#8f6cce', 'back_load': '#198966'}
    names = {'fix': '4 feet · fixed', 'load': 'Seat · 800 N',
             'back_load': 'Back · 200 N'}
    for key in ('fix', 'load', 'back_load'):
        native_pts = origin + (np.argwhere(native[key]) + .5) * pitch
        physical_pts = origin + (np.argwhere(physical[key]) + .5) * pitch
        pp = physical_pts.mean(axis=0) * 1000
        mapped = to_physical(native_pts).mean(axis=0) * 1000
        ax.scatter(pp[1], pp[2], s=110, color=colors[key], marker='o', label=names[key])
        ax.scatter(mapped[1], mapped[2], s=175, marker='o', facecolors='none',
                   edgecolors='#202a38', linewidths=1.6)
        ax.plot([pp[1], mapped[1]], [pp[2], mapped[2]], color='#202a38', lw=1)
        ax.annotate(names[key], (pp[1], pp[2]), xytext=(8, 7),
                    textcoords='offset points', fontsize=10)
    ax.set(title='BC centroids · filled = physical, ring = registered',
           xlabel='Y (mm)', ylabel='Z (mm)', xlim=(-320, 350), ylim=(-25, 970))
    ax.grid(alpha=.15)
    ax.legend(loc='upper left', fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    fem = meshio.read(SPEC / 'fea_domain/chair_035.msh').points
    native_env = origin + (np.argwhere(native['bracket']) + .5) * pitch
    distance = cKDTree(to_physical(native_env)).query(fem)[0] * 1000
    sc = ax.scatter(fem[:, 1]*1000, fem[:, 2]*1000, c=distance,
                    s=3, cmap='viridis', vmin=0, vmax=40, rasterized=True)
    far = distance > 25
    ax.scatter(fem[far, 1]*1000, fem[far, 2]*1000, s=7,
               facecolors='none', edgecolors='#ec5153', linewidths=.55,
               label=f'>25 mm: {far.sum()} nodes')
    ax.set(title='FEM nodes · nearest registered envelope voxel',
           xlabel='Y (mm)', ylabel='Z (mm)', ylim=(-25, 990))
    ax.grid(alpha=.15)
    ax.legend(loc='lower left', fontsize=10)
    fig.colorbar(sc, ax=ax, label='Nearest voxel distance (mm)', shrink=.8)

    for ax in fig.axes:
        if ax.get_label() != '<colorbar>':
            ax.set_facecolor('#ffffff')
    fig.suptitle('Chair generation grid ↔ physical FEM registration · scale 0.94, Rx(+90°)',
                 fontsize=19, weight='bold')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, facecolor=fig.get_facecolor())
    print(OUT)


if __name__ == '__main__':
    main()
