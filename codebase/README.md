# Supplementary code — Form Embodies Mechanics: Inference-Time Physics Guidance for Specification-First 3D Structural Geometry Generation

Self-contained, anonymized code to reproduce the paper's pipeline: a style prompt and
a design domain are turned into a watertight 3D structural part whose stiffness is
improved by inference-time physics guidance, then verified with an independent
finite-element solve.

The package ships **inputs only**. Everything else is produced by running the code, starting
from the three raw STLs per domain in `data_real/<domain>/`. The two pretrained models it
builds on (the Direct3D-S2 backbone and the SDXL family) are public and downloaded on first
use; they are not bundled.

## Package layout

```
3d_qd/                   # DATA_ROOT — the code tree and the bundled data sit side by side
├── codebase/            # this package (ROOT) — code only, independent of the data
│   ├── README.md        # this file — setup, layout, and run commands
│   ├── LICENSE          # CC BY-NC 4.0
│   ├── run_prep.py      # Stage 0     : raw domain STLs → remeshed STLs + BC voxel grid
│   ├── run_from_text.py    # text-first  : style prompt → conditioning image → mesh → post → FEA
│   ├── run_from_image.py   # image-first : shipped conditioning image → mesh → post → FEA (+ stage engine)
│   ├── run_conditioning.py # Stage 1 only: style prompt → multi-view conditioning images
│   ├── run_ablation.py  # FEM-guidance ablation (on / off / post-hoc)
│   ├── refea.py         # serial FEA recovery (re-run FEA on existing meshes; see Troubleshooting)
│   ├── render_results_grid.py  # per-domain result grids (ON | OFF | post-hoc + compliance)
│   ├── docs/            # pipeline structure: the stage-by-stage data-flow figure
│   ├── code/            # core scripts (+ conditioning/ subpackage, + code/README.md)
│   └── configs/         # one config per domain + curated prompts (+ configs/README.md)
├── data_real/           # per-domain geometry: the three raw STLs (design domain / fixture /
│                        #   load) + everything run_prep.py derives from them
├── data/                # per-domain conditioning images (data/<domain>/conditioning/<style>/)
├── external/            # Direct3D-S2 backbone clone (see Generation backbone below)
└── experiments/         # run outputs (created on first run; redirect with EXP_ROOT)
```

The data roots (`data/`, `data_real/`) and the run outputs live **next to** the package,
not inside it, so the code tree stays independent of the bundled geometry. `DATA_ROOT`
overrides that location; it defaults to the package's parent directory and is also the
working directory every stage runs in — which is what the configs' relative `data_real/`
paths resolve against.

In this workspace, `external/` and the conditioning directories are local copies,
including the checkpoint files under `external/trellis2_local/ckpts`. They do not
depend on the original sibling workspace. Keep `D3DS2_ROOT` and `DATA_ROOT` unset
to use these local defaults, or point them at this workspace. Python/Conda packages
and model weights downloaded by the pipelines still use the configured environments
and model caches; this is not a bundled offline environment.

Only the three raw STLs per domain are tracked in git; everything `run_prep.py` derives
(`*_remesh.stl`, `voxel.npz`, `fea_domain/`, `fea_shared.msh`) and every run output under
`data/` and `experiments/` is generated locally — build them with
`python run_prep.py --domain all`, and point `EXP_ROOT` elsewhere if you want run outputs
off the data root.

Three domains are provided (`bracket`, `motor_mount`, `link`), each with five
representative style prompts under `data/<domain>/conditioning/<style>/`, so a default
run covers **3 domains × 5 styles**. The full curated prompt sets (bracket 100,
motor-mount 20, link 20) are listed in `configs/curated_prompts.json`; the five shipped
per domain are its `default_styles`.

### Domain geometry: `data_real/<domain>/`

A domain is defined by three raw STLs — `original_DesignSpace.stl` (the design domain the part
must stay inside), `fixed.stl` (fixture surface) and `load.stl` (load surface). Everything else
is derived from them by **`run_prep.py`** (Stage 0, `python run_prep.py --domain all`):

| file | built by | consumed by |
|---|---|---|
| `*_remesh.stl` | `code/remesh_domain_bc.py` | post-processing (`stages.post`), a-posteriori FEA boundary surfaces (`stages.fea`) |
| `voxel.npz` | `code/make_bc_proper_pysdf.py` | generation — the 64³ design-domain / boundary-condition grid (`bc_proper`, `bracket_occ`) |
| `fea_domain/*.stl` | `run_prep.py` (staged copies) | the in-loop FEM solver's domain directory (`stages.mesh.fea_domain_dir`) |
| `fea_shared.msh` | `code/fenics_fea_bracket.py --build-mesh-only` | the in-loop FEM tet mesh (`stages.mesh.fea_mesh_cache`) — the design domain is the same for every prompt, so all runs share this one file |

**Every mesh a later stage consumes is the remeshed one** — post-processing, the a-posteriori FEA
boundary surfaces, and the in-loop FEA's fixture/load surfaces all read `*_remesh.stl`. Raw CAD
STLs are unwelded triangle soup (V = 3F) and low-poly at the pegs, which shows up as fan/wrinkle
artifacts at the peg↔body weld. Two places deliberately stay on the **raw** design-domain STL:

- **the voxel grid frame** — `voxel.npz` and `stages.mesh.fea_bracket_stl`. The 64³ grid's pitch
  and origin come from that STL's bounding box, and remeshing rounds off sharp outline corners
  (measured: no change for bracket / motor mount, up to **1.47 mm** for the link), which would
  shift the whole generation frame.
- **the in-loop FEM's tetrahedralized domain** — `fea_domain/original_DesignSpace.stl`. Gmsh
  meshes the raw CAD tessellation of all three domains cleanly, but rejects the bracket's
  remeshed domain outright (*"Invalid boundary mesh (overlapping facets)"*): the isotropic
  remesh can introduce self-intersecting facets. The fixture/load surfaces in the same directory
  *are* the remeshed ones — they are only ever read as a vertex cloud for picking boundary nodes
  by proximity, never meshed.

The derived files are shipped, so `run_prep.py` is a no-op on a fresh checkout (each step is
skipped when its output exists); it exists so the inputs are auditable and regenerable — see
**Input preparation** in `code/README.md`. `python run_prep.py --domain all --verify` lists what
is present without building anything.

## The pipeline

```
raw domain STLs ─(run_prep)─▶ remeshed STLs + BC voxel grid
                                    │
prompt ─(SDXL)─▶ conditioning images ─(Direct3D-S2 + physics guidance)─▶ mesh
                                                        │
                                          post-processing (boolean + remesh)
                                                        │
                                          a-posteriori FEM verification (compliance)
```

Two entry points cover the two starting points (see **How to run**):

- **`run_from_text.py`** — text-first, the full chain. Starts from a style prompt and
  synthesizes the conditioning images with SDXL before generating. Needs SDXL / IP-Adapter
  (auto-downloaded from the Hugging Face Hub) plus a GPU.
- **`run_from_image.py`** — image-first, the default. Starts from the conditioning images
  already shipped under `data/<domain>/conditioning/`, so no text-to-image stage is needed.

The stages themselves:

- **Input preparation** (Stage 0, `run_prep.py`): the three raw domain STLs → welded/remeshed
  STLs, the 64³ boundary-condition voxel grid, and the in-loop FEM solver's domain directory.
  Domain-only (no style, no prompt), deterministic, no GPU and no network. Its outputs are
  shipped, so a fresh checkout can skip it.
- **Conditioning** (Stage 1, `run_conditioning.py`, used by `run_from_text.py`): renders the
  design domain, then drives SDXL to synthesize the multi-view conditioning images from a
  style prompt. Skipped entirely by the image-first flow.
- **Generation** (`run_from_image.py --stage gen`): the Direct3D-S2 backbone, steered by
  the inference-time geometric and differentiable-FEM guidance. Needs a GPU and the backbone.
- **Post + FEA** (`run_from_image.py --stage post,fea`): boolean fusion + design-domain
  clip + isotropic remesh, then an independent tetrahedral FEM solve. No GPU.

Every per-domain difference — input-preparation values (remesh edge lengths, grid margin and
dilations), generation hyperparameters (FEM weights, guidance flags, minimum feature size),
post-processing settings, FEA fixtures/material/load, seed and data paths — lives in that
domain's config (`configs/<domain>.json`, under `stages.prep` / `stages.mesh` /
`stages.post` / `stages.fea` and top-level `seed`). Each stage's script reads it via
`apply_config(stage=...)`; the orchestrator passes only `--config` and per-run I/O paths on
the command line (see `configs/README.md`).

## Environment

Used for the reported experimental setup (Reproducibility Checklist items 4.1 / 4.7):

- GPU: **NVIDIA GeForce RTX 4090 (24 GB)**; both generation and verification run on a single GPU.
- OS: **Ubuntu 22.04.5 LTS**.

Generation/inpainting and FEM verification run in **two separate conda environments** to
avoid dependency conflicts. The in-loop differentiable FEA is driven by the generation
environment (`direct3ds2`), which invokes the verification environment's (`fenics`)
Python as a subprocess.

**`direct3ds2`** (generation + SDXL, Python 3.10.20): torch 2.5.1+cu121, torchvision
0.20.1+cu121, diffusers 0.31.0, transformers 4.40.2, accelerate 1.13.0, numpy 2.2.6,
scipy 1.15.3, scikit-image 0.25.2, trimesh 4.12.2, pymeshlab 2025.7.post1, pymeshfix
0.18.1, pysdf 0.1.9, manifold3d 3.4.1, Pillow 12.1.1, safetensors 0.8.0rc0,
huggingface-hub 0.36.2 (+ `pyvista`, `compel` for the conditioning stage). Full pins:
`code/requirements_generation.txt`.

**`fenics`** (FEM verification + in-loop FEA, Python 3.11.15): fenics-dolfinx 0.9.0, ufl
2024.2.0, basix 0.9.0, petsc4py 3.23.0, mpi4py 4.1.1, gmsh 4.15.0, meshio 5.3.5, numpy
2.4.3, scipy 1.17.1, trimesh 4.11.4. Full pins: `code/requirements_fenics.txt`.

```bash
# generation environment
conda create -n direct3ds2 python=3.10 && conda activate direct3ds2
pip install -r code/requirements_generation.txt
# then install the Direct3D-S2 backbone (see "External dependency" below)

# FEM verification environment (separate)
conda create -n fenics python=3.11 && conda activate fenics
conda install -c conda-forge fenics-dolfinx=0.9.0 gmsh   # or follow requirements_fenics.txt
```

### External dependency: the Direct3D-S2 backbone

The generation stage builds on a public, pretrained 3D generator used zero-shot. It is
**not** redistributed here (third-party code and multi-GB weights, own license) and must
be obtained separately. Post-processing and FEM verification do **not** need it.

```bash
git clone https://github.com/DreamTechAI/Direct3D-S2.git
git clone https://github.com/mit-han-lab/torchsparse   # backbone requirement; follow its build guide
```

The generation script imports the `direct3d_s2` package and expects the clone at
`external/Direct3D-S2`, resolved relative to `D3DS2_ROOT` (default: the parent directory
of this package, i.e. the same `DATA_ROOT` the data sits in):

```
<D3DS2_ROOT>/
├── external/Direct3D-S2/    # <- the cloned repository (contains direct3d_s2/)
└── codebase/                # <- this package
```

So place the clone at `../external/Direct3D-S2`, or set `D3DS2_ROOT` to whatever
directory contains `external/Direct3D-S2`. The weights download automatically: the
pipeline calls `Direct3DS2Pipeline.from_pretrained('wushuang98/Direct3D-S2',
subfolder='direct3d-s2-v-1-1')`, fetched from the Hugging Face Hub on the first run
(zero-shot, no manual checkpoint, no fine-tuning).

### Conditioning-image generation (optional)

`run_conditioning.py` reproduces the conditioning images from a style prompt using the
public SDXL models below, all downloaded from the Hugging Face Hub on first use:
`stabilityai/stable-diffusion-xl-base-1.0`,
`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`,
`h94/IP-Adapter` (`sdxl_models/ip-adapter_sdxl.safetensors`),
`madebyollin/sdxl-vae-fp16-fix`, and (only with `--control`)
`diffusers/controlnet-depth-sdxl-1.0` / `-canny-sdxl-1.0`.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `DATA_ROOT` | root holding `data/` and `data_real/`; also the cwd every stage runs in | parent directory of this package |
| `D3DS2_ROOT` | root directory that contains `external/Direct3D-S2` | parent directory of this package |
| `D3DS2_PY` | Python for generation / post-processing (the `direct3ds2` env) | current interpreter (`sys.executable`) |
| `FENICS_PY` | Python for FEM verification (the `fenics` / DOLFINx env) | `python` |
| `SD_PY` | Python for the optional conditioning stage | falls back to `D3DS2_PY` |
| `EXP_ROOT` | root for all run outputs (generation / post / FEA / grids / prep logs) | `experiments/` inside `DATA_ROOT` |
| `FEA_THREADS` | CPU threads for the FEM solves (in-loop and a-posteriori) | `9` (tuned for the reported box) |
| `NVML_PRELOAD` | (optional) `libnvidia-ml` path to `LD_PRELOAD` on a driver/library version mismatch | unset (standard paths auto-detected) |

## Running

```bash
conda activate direct3ds2
export FENICS_PY=/path/to/fenics-env/bin/python
export D3DS2_ROOT=/path/to/root-containing-external   # only if not the parent dir
```

**Stage 0 — input preparation** (optional; the derived inputs are shipped, so every step
reports `SKIP` on a fresh checkout). No GPU, no network, a few seconds per domain:

```bash
python run_prep.py --domain all             # raw STLs → *_remesh.stl + voxel.npz + fea_domain/ + fea_shared.msh
python run_prep.py --domain all --verify    # just report which inputs exist
python run_prep.py --domain link --force    # rebuild one domain from its raw STLs
```

**Image-first (default)** — start from the shipped conditioning images (no SDXL needed):

```bash
# full chain (generation → post → FEA), all domains × all provided styles (3 × 5).
# --grid renders a per-domain result grid (ON | OFF | post-hoc) when everything finishes.
python run_from_image.py --domain all --grid

# a single domain / single style
python run_from_image.py --domain bracket --style de_crackle --stage all --force

# post + FEM verification only (no GPU) once gen has written mesh.obj
python run_from_image.py --domain bracket --stage post,fea
```

**Text-first (full chain)** — start from the style prompt and synthesize the conditioning
images with SDXL first (needs the SDXL / IP-Adapter models, auto-downloaded on first run):

```bash
# prompt → conditioning image → mesh → post → FEA, all domains × 5 default styles.
# --grid renders a per-domain result grid when everything finishes.
python run_from_text.py --domain all --grid

# a single domain / single style
python run_from_text.py --domain bracket --style de_crackle --force
```


`--domain`, `--style` and `--stage` all accept `all` (default) or a comma-separated
list. Each stage is resumable: skipped when its output exists, re-run with `--force`.
Ablation outputs go to `experiments/<domain>/<style>/abl/<variant>/`; the `on` variant
uses the config weights as-is, while `off`/`posthoc` switch the FEM guidance off on the
CLI. **The primary ablation comparison (compliance reduction, on vs off vs post-hoc) is
reproduced by `run_ablation.py --domain all`.**

## Determinism and settings

- Deterministic: fixed seed **42**, `--deterministic`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  deterministic cuDNN/cuBLAS kernels, `FEA_NORMALIZE=0`. One deterministic run per prompt,
  not an average over stochastic restarts.
- a-posteriori FEM verification: linear elasticity, Young's modulus **110 GPa**, Poisson
  ratio **0.3**, total load **1000 N**, fixture surfaces clamped. The delivered surface
  (`final.obj`, post-remesh/smoothing) is what gets verified: it is uniformly remeshed
  (`stages.fea.target_edge_mm`: 2.5 mm for bracket / motor-mount, 1.5 mm for the
  link) and repaired to a clean watertight manifold,
  then tetrahedralized with Gmsh (no quadric decimation, so thin lattice members are
  preserved); `run_from_image.py --stage fea` runs this in one step via `fea_prep_and_run.py`.
  The in-loop FEA uses a normalized modulus (only the sensitivity direction is used; its
  scale is absorbed by the guidance weight), and loads the same direction the verification does
  (`stages.mesh.load_mode` = `stages.fea.force_dir`: bracket / motor-mount `-z`, link `x`) so the
  guided objective is the load case being verified.

## Troubleshooting — FEA compliance shows a failure

The a-posteriori FEA tetrahedralizes each polished mesh with Gmsh. Gmsh's fast **HXT**
algorithm can crash ("double free" / PLC fault) when **several tetrahedralizations run at
the same time**, so a *parallel* run (`--domain all`, or several styles on different GPUs at
once) occasionally leaves a few styles with a missing or `FAIL` compliance. The FEA stage
already escalates automatically (HXT → Delaunay → coarser remesh → decimate+repair fallback)
and recovers in most cases; the only way to *guarantee* no concurrency crash is to run the
tetrahedralizations one at a time.

If any style still shows a failed / missing compliance after a parallel run, re-run the FEA
**serially** with the recovery script — it reuses the meshes already on disk (generation and
post-processing are not repeated, only the cheap FEA is redone):

```bash
export FENICS_PY=/path/to/fenics-env/bin/python
export EXP_ROOT=/path/to/experiments        # same EXP_ROOT the run used (if you set one)
python refea.py --domain all                 # fill any missing/failed FEA, serially
python refea.py --domain link --variant on,posthoc   # a subset
python refea.py --domain all --force         # recompute every FEA from scratch
```

`refea.py` is config-driven exactly like the FEA stage (`stages.fea`); it only re-runs
`fea_prep_and_run.py` one variant at a time, so there is no concurrency and every mesh that
can be tetrahedralized gets a compliance.

## More documentation

| Document | Contents |
|---|---|
| `docs/README.md` | **Start here.** The pipeline as one data-flow figure (`docs/pipeline_flow.png`): what each stage consumes, what it writes, and which later stage reads it. `docs/pipeline_flow.html` adds the file-lineage table — including why the raw and remeshed versions of the same geometry each have their own consumers. |
| `code/README.md` | Every script's role, the conditioning subpackage, manual per-stage commands, and how the shipped `data/` inputs were prepared. |
| `configs/README.md` | The per-domain config values (hyperparameter table) and the curated prompt sets. |

## Notes

- **No author-identifying information**; all paths are relative to the package or driven
  by the environment variables above. **English only.** The only dependencies outside the
  package are the public Direct3D-S2 backbone and the SDXL / IP-Adapter / ControlNet
  models, reached through environment variables and downloaded on first use.
- License: CC BY-NC 4.0 (see `LICENSE`). The third-party pretrained models keep their own licenses.
