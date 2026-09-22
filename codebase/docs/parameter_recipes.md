# Parameter recipes: bracket and disk-brake caliper

## Bracket: frozen working recipe

The bracket recipe below is the current reproducible reference. Its source config is
`experiments/bracket/load_fea_union_peel_2026-09-17/dense_fea5e11_union6_peel/config.json`.
Do not silently copy these values to the caliper: the CAD frame, BC topology and load case differ.

| concern | fixed value | reason |
|---|---:|---|
| envelope attraction | `out_w=20`, `bc_w=5`, `cw=0` | keeps the generated body inside the bracket design envelope without reachability overconstraint |
| BC treatment | `force_bc_dilate_mm=6`, `bc_inloop_only=false` | preserves a 6 mm load/fixture collar through generation |
| final geometry | union of dilated BC, then intersect with `envelope ∪ original BC` | retains a connected collar but removes the oversized BC exterior |
| dense FEA | `fea_w=5e-11`, every step, warmup `0.2`, 6 mm FEM mesh | the tested FEA-on result lowered a-posteriori compliance, while increasing material |
| sparse geometry | `sp_thick_w=100`, `mc_threshold=0.4`, no Laplacian/r-min/interior term | avoids the over-smoothed sparse result seen with the generic recipe |
| sparse FEA | `sp_fea_w=0` for the single reference; `3e-12` for the 20-image batch | keep the reference and the multi-image sweep explicitly separate |

The FEA-on bracket has more volume than FEA-off, so it is a stiffness reference, not a
mass-matched claim. Compare it only after adding a volume constraint.

## Disk-brake caliper: search strategy

The caliper is much thinner in its short axes than the bracket. Its original 64-cube has
a 3.181 mm pitch while the delivered median wall is about 3 mm. Therefore `tw_*` dense-grid
minimum-thickness losses are invalid here; use sparse-stage `sp_thick_*` instead.

The baseline (`configs/caliper.json`) is retained as the control. Existing evidence rules out
three false shortcuts: lowering the guidance peak did not improve image fidelity, lowering the
Laplacian alone did not improve it, and topology sign correction alone made a high-genus mesh.

The active search holds the load case (`spread:x`), material, 1,000 N force, six calibrated views,
post-processing and FEM mesh fixed. It varies only the following coupled sparse parameters:

| candidate | sign fix | r-min | sparse thickness | Laplacian |
|---|---:|---:|---:|---:|
| control | off | 1 | 0.5 | 50 |
| sign-only | on | 1 | 0.5 | 50 |
| guarded-light | on | 1.5 | 1 | 60 |
| guarded-mid | on | 2 | 2 | 75 |
| guarded-strong | on | 3 | 3 | 100 |
| fidelity-mid | on | 2 | 2 | 25 |
| regularized-legacy-sign | off | 2 | 2 | 75 |

The first pass sets dense and sparse FEA weights to zero. Rank candidates only if they are
watertight and retain BC contacts, then compare volume, projected IoU, genus and thin-ridge
fraction. Re-run the two best candidates with FEA enabled and compare compliance and maximum
stress. A lower compliance alone does not win if it comes from extra mass or micro-perforations.
