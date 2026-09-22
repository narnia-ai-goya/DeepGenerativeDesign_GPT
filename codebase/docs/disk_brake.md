# Disk-brake caliper recipe

The disk-brake example uses the existing `caliper` domain.  Its design envelope,
mounting surfaces, piston/load surfaces, shared in-loop FEA tet mesh, and six matched
conditioning views live under `data_real/caliper/` and `data/caliper/`.

Run the complete image-first path with:

```bash
cd /home/goya/SDL/3d_qd
export D3DS2_PY=/home/goya/miniconda3/envs/direct3ds2/bin/python
export FENICS_PY=/home/goya/miniconda3/envs/fenics/bin/python
python codebase/run_disk_brake.py --style builtin_kagome_consistent
```

For a structural-style sweep, pass several supplied image sets as one comma-separated
argument.  They run serially on one GPU and each receives its own output directory:

```bash
python codebase/run_disk_brake.py --style builtin_kagome_trial,de_Ktruss,de_Xbrace,de_prattframe,de_trilattice,de_warren
```

The `de_*` images predate the six-view camera contract and only have a matching
`v06_left_lo` view. Use the explicit single-view runner for those images; it does not
pretend that their oblique views are front, back or top views:

```bash
python codebase/run_disk_brake_legacy_single.py
```

For eight diverse, camera-calibrated six-view examples, use the geometry-only batch.
It uses `caliper_ffff` inputs and defers FEA until the best visual candidates are selected:

```bash
python codebase/run_disk_brake_examples.py
```

The command calls `run_from_image.py`; it does not carry hidden flags.  All settings
are declared in `codebase/configs/caliper.json`:

1. `stages.mesh`: dense and sparse FEA guidance plus the `spread:x` caliper load case.
2. `stages.post`: design-envelope clip, BC union and final remesh.
3. `stages.fea`: the independent 1,000 N verification solve.

Set `EXP_ROOT` to isolate a new run.  For example,
`EXP_ROOT=/home/goya/SDL/3d_qd/experiments/disk_brake_clean_2026-09-17` keeps all
outputs separate from earlier caliper studies.
