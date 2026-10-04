"""Explain apparent chair thickness versus measured sparse solid volume."""
from __future__ import annotations

import json
import numpy as np
import trimesh
from pysdf import SDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from run_chair_existing_fea_on_015_2026_10_04 import OUT, OLD, SPEC


def main():
    folder = OUT / 'evaluation/volume_difference'
    folder.mkdir(parents=True, exist_ok=True)
    spec = np.load(SPEC / 'voxel.npz')
    idx = np.indices((64, 64, 64)).reshape(3, -1).T
    xyz = spec['origin'] + (idx + .5)*spec['pitch_xyz']
    voxel_l = float(np.prod(spec['pitch_xyz'])*1000)
    sources = {
        'dense_off': OLD / 'dense_evaluation/round_01/mesh_cases/dense_control/aligned_main.obj',
        'dense_new_on': OUT / 'evaluation/new_dense_on/aligned_main.obj',
        'sparse_off': OUT / 'evaluation/off/direct_full_mesh_15mm/aligned_full.obj',
        'sparse_new_on': OUT / 'evaluation/new_sparse_on/direct_full_mesh_15mm/aligned_full.obj',
    }
    volumes = {}
    occupancy = {}
    for name, source in sources.items():
        mesh = trimesh.load(source, force='mesh', process=False)
        volumes[name] = float(abs(mesh.volume)*1000)
        if name.startswith('sparse'):
            occupancy[name] = (SDF(mesh.vertices.astype('float32'),
                                   mesh.faces.astype('uint32'))(xyz.astype('float32')) > 0)
    off, new = occupancy['sparse_off'], occupancy['sparse_new_on']
    added, removed = new & ~off, off & ~new
    bands = [('<0.12 m feet', xyz[:, 2] < .12),
             ('0.12–0.45 m legs', (xyz[:, 2] >= .12) & (xyz[:, 2] < .45)),
             ('0.45–0.65 m seat/arms', (xyz[:, 2] >= .45) & (xyz[:, 2] < .65)),
             ('≥0.65 m back', xyz[:, 2] >= .65)]
    by_band = {label: {'off_L': float((off & mask).sum()*voxel_l),
                       'new_L': float((new & mask).sum()*voxel_l),
                       'added_L': float((added & mask).sum()*voxel_l),
                       'removed_L': float((removed & mask).sum()*voxel_l)}
               for label, mask in bands}
    result = {'exact_OBJ_volume_L': volumes, 'voxel_pitch_mm': float(spec['pitch_xyz'][0]*1000),
              'new_only_voxel_volume_L': float(added.sum()*voxel_l),
              'off_only_voxel_volume_L': float(removed.sum()*voxel_l),
              'bands': by_band}
    (folder / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    fig = plt.figure(figsize=(17.5, 5.3), facecolor='white')
    gs = fig.add_gridspec(1, 3, width_ratios=[.9, 1.05, 1.0],
                          left=.10, right=.95, bottom=.22, top=.84, wspace=.40)
    ax = fig.add_subplot(gs[0, 0])
    labels = ['Dense OFF', 'Dense new ON', 'Sparse OFF', 'Sparse new ON']
    vv = [volumes[n] for n in ('dense_off', 'dense_new_on', 'sparse_off', 'sparse_new_on')]
    ax.barh(labels, vv, color=['#9ea9b0', '#6d8b78', '#9ea9b0', '#6d8b78'])
    ax.invert_yaxis(); ax.set_xlim(0, 43); ax.set_xlabel('Exact watertight OBJ volume (L)')
    ax.set_title('Dense versus Sparse')
    for i, v in enumerate(vv): ax.text(v+.6, i, f'{v:.2f}', va='center')
    ax.grid(axis='x', alpha=.2)
    ax = fig.add_subplot(gs[0, 1])
    keys = [k for k, _ in bands]
    y = np.arange(len(keys))
    ax.barh(y-.18, [by_band[k]['off_L'] for k in keys], .35, color='#8b9ba7', label='Sparse OFF')
    ax.barh(y+.18, [by_band[k]['new_L'] for k in keys], .35, color='#6d8b78', label='Sparse new ON')
    ax.set_yticks(y, ['Feet', 'Legs', 'Seat / arms', 'Back']); ax.invert_yaxis(); ax.set_xlabel('64³ voxel solid volume (L)')
    ax.set_title('Where volume changed'); ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=.2)
    ax = fig.add_subplot(gs[0, 2])
    z = (spec['origin'][2] + (np.arange(64)+.5)*spec['pitch_xyz'][2])
    slab = (z >= .45) & (z < .65)
    diff = (new.reshape(64,64,64).astype(int) - off.reshape(64,64,64).astype(int))[:, :, slab].sum(axis=2)
    bound = max(1, int(np.max(np.abs(diff))))
    im = ax.imshow(diff.T, origin='lower', cmap='RdBu_r',
                   norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
    ax.set_title('Seat/arms top view · new minus OFF')
    ax.set_xlabel('X grid'); ax.set_ylabel('Y grid')
    cb = fig.colorbar(im, ax=ax, fraction=.047, pad=.03)
    cb.set_label('Voxel count per X–Y column')
    fig.suptitle('Why the new chair can look thicker but contain less material',
                 fontsize=18, y=.97)
    fig.text(.06, .055, 'Red = new material; blue = removed material. The largest net loss is inside the seat/arm height band.',
             fontsize=11, color='#425769')
    fig.savefig(folder / 'volume_difference.png', dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
