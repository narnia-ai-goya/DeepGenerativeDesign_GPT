# Code

Inference-time physics guidance for specification-first 3D structural geometry generation. Contains only the core code needed for reproduction. Absolute paths have been normalized to relative paths (`./`), and author/environment-identifying information has been removed.

## File layout

### Generation (inference-time guidance)
| File | Role |
|---|---|
| `generate_with_physics_guidance.py` | Main generation script. Backprops geometric guidance (3-region BCE, reachability, r_min, etc.) and differentiable-FEM sensitivity into the latent across the two cascades (dense 64³ / sparse 512³). |
| `fea_compliance_loss.py` | Differentiable FEM compliance loss. Computes adjoint sensitivity (∂C/∂ρ) and returns it as a guidance gradient. In-loop FEA calls `fenics_fea_bracket.py`. |
| `fenics_fea_bracket.py` | In-loop linear-elastic FEM solver (FEniCSx/DOLFINx). Assembles element stiffness with SIMP penalization and computes compliance and sensitivity. |
| `run_config.py` | Helper for applying a JSON config (`apply_config`). |

Both FEM stages place their voxel centers in the **same world frame as the occupancy grid**: the
`origin` / `pitch_xyz` stored in `voxel.npz` (the `[env_frame]` line the generator prints), with
the STL-bbox formula used only as a fallback for a grid that carries no `origin`. The dense stage
resolves this via `env_origin` / `env_pitch`, the sparse stage via
`fea_compliance_loss.voxel_nodes_from_stl()`. The load *direction* is likewise shared: the
generator exports `stages.mesh.load_mode` as `LOAD_MODE` for its FEA subprocess, and
`run_from_image.py` errors out if that disagrees with the verified `stages.fea.force_dir`. This matters because `fenics_fea_bracket.py` maps
the density field onto the tet mesh by nearest-neighbour search over those coordinates — a frame
mismatch silently evaluates the sensitivity on a translated density field.

### Post-processing + a-posteriori validation
| File | Role |
|---|---|
| `post_hybrid_union_clip.py` | Mesh-level union (manifold3d) of the generated body ∪ fixture/load pegs, with locally marching-cubed dilated-peg collars, then the design-domain clip. The body itself is never resampled. |
| `surface_remesh_pre.py` | Isotropic remesh (edge from `stages.post.edge_mm`, ~1.5 mm) + Laplacian smoothing → watertight surface. Reads `--config`. |
| `fea_tet_from_mesh.py` | Tetrahedralize a clean surface (Gmsh) and compute validation compliance with DOLFINx. Called with `--no-repair` by `fea_prep_and_run.py`. |
| `fea_prep_and_run.py` | **A-posteriori FEA entry point (all domains).** Isotropic remesh + pymeshfix → clean watertight manifold (uniform edge, no quadric decimation) → `fea_tet_from_mesh.py --no-repair`, with HXT→Delaunay and edge-escalation fallback and a compliance sanity bound. Invoked by `run_from_image.py`'s FEA stage. |

### Input preparation — Stage 0 (`../run_prep.py`)
A domain is defined by three raw STLs under `../data_real/<domain>/`: `original_DesignSpace.stl`
(design domain), `fixed.stl` (fixture surface), `load.stl` (load surface). `../run_prep.py`
derives everything else from them into the same directory, config-driven via `stages.prep`:

| File | Role |
|---|---|
| `remesh_domain_bc.py` | **Step 1 (`--stage remesh`).** Weld (`merge_close_vertices`) + isotropic-remesh the three raw STLs → `*_remesh.stl`. Raw CAD STLs are unwelded triangle soup (V = 3F) and low-poly at the pegs, which shows up as fan/wrinkle artifacts at the peg↔body weld in post-processing. Edge lengths from `stages.prep` (`edge_envelope_mm` 1.5, `edge_peg_mm` 1.0). |
| `make_bc_proper_pysdf.py` | **Step 2 (`--stage voxel`).** Rasterize the three STLs to the 64³ grid → `voxel.npz`, read by generation as both `bc_proper` and `bracket_occ` (`stages.mesh`). Voxel **centers** are tested against the STL with pysdf (`inside > 0`), so the occupied volume matches the STL volume up to discretization instead of the ~1-pitch inflation a bbox-overlap rasterizer gives. NPZ fields: `bracket` (design-domain mask), `bc` = (`fix` ∪ `load`) ∩ `bracket`, `design` = `bracket` ∩ ¬`bc`, `fix`, `load`, plus `pitch_xyz` / `origin` / `pitch` (the grid frame — voxel *i* center = `origin + (i + 0.5) · pitch_xyz`). Resolution / margin / dilations from `stages.prep`. |
| (`../run_prep.py` itself) | **Step 3 (`--stage fea_domain`).** Stage `fea_domain/{original_DesignSpace,fixed,load}.stl` — the three filenames `fenics_fea_bracket.py` resolves inside its `--domain-dir` (= `stages.mesh.fea_domain_dir`). The generator only copies into that dir when a file is missing, so pre-staging here decides which geometry the in-loop FEM sees: the design domain is the **raw** STL (Gmsh tetrahedralizes it, and rejects the bracket's remeshed domain — *"Invalid boundary mesh (overlapping facets)"*, i.e. self-intersections from the isotropic remesh), while `fixed.stl` / `load.stl` are the **remeshed** pegs (read only as a vertex cloud, for picking boundary nodes by proximity). |
| (`../run_prep.py` itself) | **Step 4 (`--stage fea_mesh`).** Build `fea_shared.msh`, the in-loop FEM tet mesh, once per domain — `fenics_fea_bracket.py --build-mesh-only` on `fea_domain/` at `stages.mesh.fea_mesh_size`. The in-loop FEA tetrahedralizes the **design domain**, which is identical for every prompt, so `stages.mesh.fea_mesh_cache` points every run at this one file and the solver just reports *"reusing cached mesh"* instead of re-tetrahedralizing per style/variant. (Not to be confused with `voxel.npz`: that is a voxel occupancy grid with no connectivity. `--fea-mesh-cache` also accepts an `.npz` holding `nodes`+`tets` — a tet mesh in npz form — but the shipped configs use the gmsh `.msh`.) |

```bash
# generation env active (needs pymeshlab, pysdf, scipy); no GPU, no network
python run_prep.py --domain all                      # all four steps, all three domains
python run_prep.py --domain all --verify             # report what exists, build nothing
python run_prep.py --domain link --stage voxel --force
```

Each step is resumable (skipped when its output exists), so this is a no-op on a fresh
checkout — the derived files are shipped. Manual equivalents, if you want to bypass the
orchestrator (it passes nothing but `--config` and the I/O paths):

```bash
python code/remesh_domain_bc.py    --config configs/link.json --domain-dir data_real/link
python code/make_bc_proper_pysdf.py --config configs/link.json \
    --bracket data_real/link/original_DesignSpace.stl \
    --fix data_real/link/fixed.stl --load data_real/link/load.stl --out data_real/link/voxel.npz
```

**Which mesh each later stage reads.** Everything consumed downstream is the remeshed
geometry: `stages.post` (`bracket_stl` / `fix` / `load`), the a-posteriori FEA boundary surfaces
(`stages.fea.fix_stl` / `load_stl`), the in-loop BC pegs (`stages.mesh.fix_stl` / `load_stl`)
and `fea_domain/{fixed,load}.stl`. Two references stay on the **raw** design-domain STL, each
for a measured reason:

| raw reference | why not the remesh |
|---|---|
| step 2's `--bracket`, i.e. `voxel.npz`, and `stages.mesh.fea_bracket_stl` | That STL's bbox fixes the 64³ grid's pitch and origin. Remeshing rounds off sharp outline corners — no bbox change for bracket / motor mount, but up to **1.47 mm** for the link — which would shift the whole generation frame. |
| `fea_domain/original_DesignSpace.stl` (step 3) | Gmsh tetrahedralizes it. All three raw CAD tessellations mesh cleanly; the bracket's *remeshed* domain is rejected outright (*"Invalid boundary mesh (overlapping facets)"* — the isotropic remesh can self-intersect). |

Per-domain `stages.prep` differences: `margin` is the padding around the design-domain bbox as
a fraction of its longest extent, which fixes the cubic grid's pitch — 0.05 gives pitch
3.097 mm for the bracket and 2.062 mm for the link, while **0.25** for the motor mount gives
1.875 mm (≈ 21 voxels through its 40 mm thickness). The bracket additionally dilates its
rasterized fixture/load regions by **2 voxels** (`dilate_fix` / `dilate_load`, intersected back
with the design-domain mask) — ≈ 6.2 mm at its 3.097 mm pitch, i.e. the same ≈ 6 mm collar as
`stages.post.peg_dilate_mm`; the motor mount and link use the raw fixture/load rasterization.
`dilate_bracket: 1`, used by all three, grows the design-domain mask by one voxel so the STL is
contained in the voxelized domain.

### Visualization
| File | Role |
|---|---|
| `../render_results_grid.py` | Render per-domain result grids (ON \| OFF \| post-hoc + verified compliance) from `experiments/`. Also invoked by `run_from_text.py` / `run_from_image.py --grid`. |

### Conditioning-image generation (optional — `conditioning/`)
Reproduces the multi-view conditioning images from a style prompt. Provided images
under `data/<domain>/conditioning/` already let the pipeline run, so this stage is
optional. Orchestrated by `../run_conditioning.py` (`prompt → render + masks → SDXL
reference → inpaint → nonblack_to_white`); the luminance-normalized output (a solid-silhouette
image with the boundary-condition regions kept solid) is the conditioning the generator reads.
This is the same image the shipped `data/<domain>/conditioning/` provides, so text-first and
image-first are identical.
| File | Role |
|---|---|
| `conditioning/cond_render_pv.py` | Multi-view render of the design body (pyvista). |
| `conditioning/cond_render_masks.py` | Occlusion-aware stylable / boundary-condition masks. |
| `conditioning/text2img_style_ref.py` | SDXL text-to-image style reference image. |
| `conditioning/cond_stylize_inpaint_xattn.py` | SDXL inpainting with cross-view self-attention. |
| `conditioning/cond_stylize_inpaint_refip_xattn.py` | Inpainting + external-reference IP-Adapter (+ optional ControlNet). |
| `conditioning/nonblack_to_white.py` | Luminance normalization of the stylized views (boundary-condition regions kept solid) → **the final conditioning the generator reads**. |
| (prompt text) | Read from `../configs/curated_prompts.json` and passed to the scripts by `../run_conditioning.py`. |

## External dependencies (installed separately — not included)
- **Direct3D-S2 backbone** (pretrained, used zero-shot): public release. `generate_with_physics_guidance.py` imports the `direct3d_s2` package and assumes `external/Direct3D-S2` is added to `sys.path`. See `../README.md` ("External dependency") for the download, placement, and `D3DS2_ROOT`.
- Python packages: `torch` 2.5.1 (CUDA 12.1), `diffusers` 0.31.0, `trimesh`, `pymeshlab`, `pymeshfix`, `pysdf`, `manifold3d`, `scikit-image`, `scipy`, `numpy`, `Pillow`.
- Validation (FEM): `FEniCSx` (DOLFINx 0.9.0), `gmsh`, `mpi4py`, `petsc4py`, `ufl`, `meshio`. (Separate conda environment recommended — `fenics`.)
- Conditioning stage (optional): `pyvista`, `compel`, and public SDXL models (SDXL base 1.0, SDXL inpainting 0.1, IP-Adapter `h94/IP-Adapter`, `madebyollin/sdxl-vae-fp16-fix`, and ControlNet depth/canny only with `--control`) auto-downloaded from the Hugging Face Hub. See `../README.md` ("Conditioning-image generation").

## Environment variables (all paths relative/overridable — no hardcoded absolute paths)
| Variable | Purpose | Default |
|---|---|---|
| `D3DS2_PY` | python for generation/post (direct3ds2 env) | current interpreter (`sys.executable`) |
| `FENICS_PY` | python for FEA (fenics/dolfinx env) | `python` |
| `D3DS2_ROOT` | root containing the Direct3D-S2 backbone (`external/Direct3D-S2`) | parent directory of this package |
| `EXP_ROOT` | root for all run outputs | `experiments/` in the package |
| `FEA_THREADS` | CPU threads for the FEM solves | `9` |
| `NVML_PRELOAD` | (optional) libnvidia-ml LD_PRELOAD path on driver mismatch | unset (standard paths auto-detected) |

Recommended run: **with the direct3ds2 env activated**, run the commands below, and for FEA just set `export FENICS_PY=<fenics env python>`.

## Running — automatic orchestrator (recommended)
Two entry points, one per starting point (both take `--domain/--style/--stage/--force`), plus
Stage 0 input preparation:
```
cd supplementary
export FENICS_PY=/path/to/fenics-env/bin/python   # for FEA (anonymized: use your own path)

# stage 0 (optional — outputs are shipped): raw domain STLs → *_remesh.stl + voxel.npz + fea_domain/
python run_prep.py --domain all

# image-first (default): shipped conditioning image → generation → post → a-posteriori FEA
python run_from_image.py --domain all --grid              # all domains × styles (3 × 5); --grid renders per-domain result grids
python run_from_image.py --domain bracket --stage all --force   # bracket from the conditioning image (generation needs GPU + backbone)
python run_from_image.py --domain link --stage post,fea         # post+fea (after gen has written mesh.obj)

# text-first (full chain): style prompt → SDXL conditioning image → generation → post → FEA
python run_from_text.py --domain all --grid               # all domains × 5 default styles (needs SDXL); --grid renders result grids
python run_from_text.py --domain bracket --style de_crackle
```
**FEA ablation (on/off/post-hoc at once)** — compares the in-loop FEM guidance effect:
```
python run_ablation.py --domain bracket        # on/off/posthoc generation→post→FEA, prints compliance comparison table
python run_ablation.py --domain all --variant on,off
```
Variants differ only in generation flags (on=`--fea-w/--sp-fea-w`, off=`0`, posthoc=`0 --posthoc-fea-steps 60 --posthoc-fea-lr 2e-3`); post/FEA use the same code. Output is `experiments/<dom>/abl/<variant>/`.
- `run_from_image.py` selects the per-domain config, env (generation=`direct3ds2`, FEA=`fenics`), NVML LD_PRELOAD, and paths.
  It passes NO generation hyperparameter on the CLI: all of them live in `configs/<domain>.json` (`stages.mesh`).
  Resumable: a stage is skipped if its output exists, re-run with `--force`.
- The package ships inputs only (domain geometry in `data_real/<domain>/`, conditioning images in `data/<domain>/conditioning/`), so a fresh run starts from generation; once `gen` has written
  `mesh.obj`, the `post` and `fea` stages need no GPU.
- Only the generation stage needs the Direct3D-S2 backbone (`../external/Direct3D-S2`) + GPU.

### Manual per-stage execution (same as inside the orchestrator)
For the per-domain config values, see `../configs/README.md`. Summary:
0. **Input preparation** (`--config`, `stages.prep`): `remesh_domain_bc.py` → `*_remesh.stl`, then `make_bc_proper_pysdf.py` → `voxel.npz`, then the `fea_domain/` staging — see **Input preparation** above. Domain-only and shipped, so normally skipped.
1. **Generation**: `python generate_with_physics_guidance.py --config ../configs/<domain>.json --target-dir <conditioning> --out <out>` — all generation hyperparameters come from the config's `stages.mesh`.
2. **Post-processing** (`--config`, `stages.post`): `post_hybrid_union_clip.py` (mesh-level union + design-domain clip; peg-dilate bracket 6 / mm 2 / link 0) → `surface_remesh_pre.py` (isotropic remesh `edge_mm` + Laplacian `laplacian_iters`/`laplacian_lamb`). Identical code for all three domains; only the values differ.
3. **Validation** (`--config`, `stages.fea`): `fea_prep_and_run.py` on `final.obj` (the delivered surface, not the intermediate boolean `hybrid.obj` — the smoothed surface also tetrahedralizes on the first HXT rung in every domain, where the raw boolean needed edge escalation or failed on fine lattices) — isotropic remesh (`target_edge_mm`: 2.5 mm bracket / motor-mount, 1.5 mm link) + pymeshfix → clean watertight manifold, then `fea_tet_from_mesh.py --no-repair` (HXT→Delaunay + edge escalation, compliance sanity bound). Fixtures/load, load direction, tet edge size and the material/load spec (`E_GPa`, `nu`, `force_N`) all come from the config.

## Notes
- seed 42, `--deterministic`, `FEA_NORMALIZE=0` (required for reproduction).
- Inputs are bundled under `data_real/<dom>/` (raw design-domain / fixture / load STLs plus the `run_prep.py` outputs: `*_remesh.stl`, the boundary-condition `voxel.npz`, `fea_domain/`), `data/<dom>/conditioning/<style>/` (the conditioning images) and `configs/`. Running the code writes outputs under `experiments/<dom>/`.
- Ablation output path: `experiments/<dom>/<style>/abl/<variant>/`.
