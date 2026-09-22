# Experiment configs

One config per domain drives its full pipeline. **Every tunable — generation
hyperparameters, post-processing settings, FEA fixtures/material/load, data paths and
the seed — lives in the domain config** and is applied by each stage's script via
`apply_config(stage=...)`:

| stage | config block | applied by |
|---|---|---|
| input preparation | `stages.prep` | `remesh_domain_bc.py`, `make_bc_proper_pysdf.py` |
| generation | `stages.mesh` (+ top-level `seed`) | `generate_with_physics_guidance.py` |
| post-processing | `stages.post` | `post_hybrid_union_clip.py`, `surface_remesh_pre.py` |
| a-posteriori FEA | `stages.fea` | `fea_prep_and_run.py` |
| conditioning | `stages.cond_stylize` + `curated_prompts.json` | `run_conditioning.py` |

The orchestrator (`run_prep.py` / `run_from_image.py` / `run_ablation.py`) passes **only
`--config` and per-run I/O paths** (`--domain-dir`/`--target-dir`/`--in`/`--out`) on the
command line — nothing else — so a config is the single source of truth for a domain's
settings.

All data paths point into that domain's geometry directory, `data_real/<domain>/`. Every mesh a
stage consumes is the remeshed one (`*_remesh.stl`), with two deliberate raw-STL references:
`stages.mesh.fea_bracket_stl` (plus the design-domain input of `stages.prep`'s voxel step),
which define the 64³ grid frame, and `fea_domain/original_DesignSpace.stl`, which Gmsh
tetrahedralizes. See `code/README.md`, **Input preparation**, for the measurements behind both.

## Config files
| File | Purpose |
|---|---|
| `bracket.json` | bracket domain (design-domain-baked guidance recipe) |
| `motor_mount.json` | motor-mount domain |
| `link.json` | suspension-link domain |
| `curated_prompts.json` | the paper's curated prompt sets per domain (bracket 100, motor-mount 20, link 20); each style holds its `prompt` and per-prompt `negative`, plus the `default_styles` shipped as conditioning |

## Input-preparation settings (`stages.prep`)
Read by `remesh_domain_bc.py` (`envelope`, `edge_envelope_mm`, `edge_peg_mm`) and
`make_bc_proper_pysdf.py` (`res`, `mode`, `margin`, `dilate_fix`, `dilate_load`,
`dilate_bracket`) — the two scripts share the block, each picking up only the keys that are its
own arguments. Driven by `run_prep.py`.

| key | bracket | motor-mount | link |
|---|---|---|---|
| `envelope` (design-domain STL basename) | `original_DesignSpace` | ← | ← |
| `edge_envelope_mm` (isotropic remesh, design domain) | 1.5 | 1.5 | 1.5 |
| `edge_peg_mm` (isotropic remesh, fixture/load) | 1.0 | 1.0 | 1.0 |
| `res` (voxel grid per axis) | 64 | 64 | 64 |
| `mode` (cubic grid, uniform pitch) | `uniscale` | ← | ← |
| `margin` (bbox padding → grid pitch) | 0.05 → 3.097 mm | **0.25** → 1.875 mm | 0.05 → 2.062 mm |
| `dilate_fix` / `dilate_load` (voxels) | **2 / 2** (≈ 6.2 mm) | 0 / 0 | 0 / 0 |
| `dilate_bracket` (voxels) | 1 | 1 | 1 |

## Per-domain generation settings (`stages.mesh`)
The three domains share the same generator and post/FEA code and differ only in the
config values below (matching the hyperparameter table in the paper's Supplementary):

| key | bracket | motor-mount | link |
|---|---|---|---|
| `fea_w` (dense FEM) | 1.5e-12 | 7.5e-12 | 4.5e-14 |
| `sp_fea_w` (sparse FEM) | 3e-11 | 1.5e-10 | 9e-13 |
| `fea_penal` (SIMP exponent) | 2 | 2 | 2 |
| `rmin_radius` | 3 | 2 | 4 |
| `cw` (reachability) | 0.5 | 5.0 | 5.0 |
| `vol_target` | 0.4 | 0.4 | 0.4 |
| `tw_rmin` | 0.025 | 0.025 | 0.025 |
| `force_envelope_clip` / `aug_lag` / `heaviside_proj` | — | true | true |
| `bce_boundary`, `bce_mode`, `dense_index_threshold`, `mc_threshold` | — | 0.1, inout, 0.1, 0.3 | 0.1, inout, 0.1, 0.3 |
| `load_mode` (in-loop FEM load direction) | -z | -z | x |

**`load_mode` must state the same load case as `stages.fea.force_dir`.** It is the direction the
in-loop FEM guidance optimizes; `force_dir` is the direction the a-posteriori FEA verifies. The
generator exports it as the `LOAD_MODE` environment variable its FEA subprocess reads (the solver
falls back to `diag` = (1,−1,−1)/√3 when unset, which is *not* any domain's verified load case),
and `run_from_image.py` refuses to launch a run whose two keys disagree — the pair drifted apart
once, leaving bracket and link optimizing a diagonal load while verification loaded `-z` / `x`
(measured sensitivity agreement cos 0.78 / 0.48). All other keys are consumed directly by the
generator's argument parser.

Data paths in this block: `bc_proper` / `bracket_occ` (both `data_real/<domain>/voxel.npz` —
the 64³ grid), `fea_mesh_cache` (`data_real/<domain>/fea_shared.msh` — the in-loop FEM tet mesh
built once by `run_prep.py --stage fea_mesh` and shared by every prompt), `fea_domain_dir` (`data_real/<domain>/fea_domain` — the staged in-loop FEM domain dir: **raw**
design domain for Gmsh, **remeshed** pegs for boundary-node picking), `fix_stl` / `load_stl` (the remeshed fixture/load pegs the
in-loop FEA reads) and `fea_bracket_stl` (the **raw** design-domain STL — it defines the voxel
grid frame, so it must match how `voxel.npz` was rasterized).

## Post-processing settings (`stages.post`)
Per domain: `bracket_stl` / `fix` / `load` (the remeshed design-domain + fixture/load STLs used
for the mesh-level union and clip), `peg_dilate_mm` (fixture/load collar, bracket 6 / mm 2 / link 0),
`edge_mm` (isotropic remesh, 1.5 mm), `laplacian_iters` /
`laplacian_lamb` (final smoothing), and `spike_trim` (link only, `true`) — a voxel
morphological opening that strips the thin protruding flaps left on the link's bow-tie web.
Read via `apply_config(stage='post')`.

## a-posteriori FEA settings (`stages.fea`)
Per domain: `fix_stl` / `load_stl` (the remeshed fixture/load surfaces — the clamped and loaded
boundary of the verification solve), `tet_size` (Gmsh tet size — bracket
0.005 / mm 0.006 / link 0.0015, i.e. link is tet-fine to match its 1.5 mm surface),
`target_edge_mm` (prep remesh — bracket/mm 2.5 mm, link 1.5 mm), `force_dir` (validation load
direction — bracket/mm `0,0,-1`, link `1,0,0`), the material/load spec `E_GPa` = 110, `nu` = 0.3,
`force_N` = 1000, and `prep_mode` — `remesh` (default; isotropic re-remesh + edge escalation,
bracket/mm) or `direct` (link; feed the polished mesh straight to `fea_tet_from_mesh`'s own
decimate+repair, since an isotropic re-remesh collapses the link's slender members).
Read by `fea_prep_and_run.py` via `apply_config(stage='fea')`.

## Style prompts (`curated_prompts.json`)
Per domain: `count`, the list of `default_styles` (the five shipped as conditioning), and
`styles` mapping every curated style key to `{ "prompt": ..., "negative": ... }` — the
positive prompt and its per-prompt negative (bracket 100, motor-mount 20, link 20).

These are the human-authored specification used to synthesize each style's multi-view
conditioning images. `run_conditioning.py` reads the prompt and negative from here and passes
them to the conditioning scripts. The generator itself consumes the rendered conditioning
images, so the prompts document intent rather than being read at generation time.
