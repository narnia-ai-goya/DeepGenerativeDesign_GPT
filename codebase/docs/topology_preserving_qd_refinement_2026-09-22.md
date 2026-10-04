# Topology-preserving refinement for image-conditioned structural QD

## Decision

Do not use the pretrained sparse Direct3D-S2 stage as a second topology generator.
The dense QD scaffold defines topology; the high-resolution stage may only deform
that scaffold's surface. This is the only training-free formulation that makes a
shape-QD claim about the final mesh defensible.

## Evidence from the current pipeline

For the same image, seed, envelope and BCs:

| stage | output | components | interpretation |
|---|---|---:|---|
| OC-Flow dense, BC-connected, cleaned | `niche_01_w20_cw50_t04_meshclean/mesh_dense.obj` | 1 | connected topology is attainable at 64^3 |
| sparse refiner from that exact cache | `niche_01_w20_cw50_t04_full/mesh.obj` | 117 | the sparse model re-samples topology instead of refining it |

The token field given to sparse was itself one 6-connected, BC-anchored component.
The failure is therefore not a dense QD failure or an interface-component filter
failure. It is a mismatch between the desired coarse-to-fine relation and the
unconditionally pretrained sparse prior.

The sparse run also had all scaffold-preservation terms disabled (`sp_hole_w=0`,
`sp_interior_w=0`, `sp_edt_w=0`). Turning them on is useful as an ablation, but
they are soft latent losses. They cannot provide a topology guarantee, and the
legacy topology loss requires the explicit corrected SDF-sign option.

## Recommended method: QD scaffold + constrained surface deformation

Let `M0` be the connected dense mesh after envelope and BC projection. It is the
individual stored in the QD archive. Create a high-resolution mesh `M` by isotropic
subdivision/remeshing of `M0`; its face connectivity is fixed for the entire
refinement.

The sparse SDF decoder can still contribute a learned geometric prior, but it is
not allowed to define a new zero level set. At each vertex `v` of `M0`, with normal
`n`, optimize only a bounded normal displacement:

```
v' = v + d(v) n(v),     |d(v)| <= r(v)
```

where `r(v)` is at most a fraction of local medial thickness (for example 20--25%).
Tangential displacement is omitted or strongly regularized. The objective is:

```
E(d) = lambda_sdf * phi_sparse(v + d n)^2
     + lambda_fair * ||L d||^2
     + lambda_img * silhouette_error(render(M(d)), reference)
     + lambda_BC * BC_attachment_error
```

with three hard projections after every update:

1. project vertices inside the envelope union peg region;
2. pin vertices in the fixed/load attachment bands to the BC surface;
3. clip `d` to the local topology-safe radius `r(v)`.

Because the mesh connectivity never changes, detached fragments and new holes
cannot appear. The high-resolution model can only make the connected QD scaffold
smoother and better aligned to image/SDF evidence.

### Practical first implementation

Start with no sparse model at all:

1. dense QD mesh -> marching cubes -> largest BC-bearing component;
2. isotropic remesh to 1.5--2.5 mm target edge;
3. Taubin/bilateral surface fairing with BC vertices pinned;
4. envelope projection and independent FEA.

This gives a deterministic topology-preserving control baseline. Then add the
bounded normal-displacement term from the sparse SDF and compare it against that
baseline. If it does not improve rendered consistency or FEA without violating the
offset bound, it should not be retained.

## Why this is a better research framing

The QD genotype is the dense connected structural scaffold. Its descriptor is
measured on the final refined mesh but is constrained to remain in the descriptor
cell of its scaffold. Quality is independent FEA plus manufacturability and
containment. High-resolution refinement is a feasible-set map, not a source of
new QD behavior.

This separates two questions that the current pipeline conflates:

- **Where does topology diversity come from?** Dense QD control / QD archive.
- **How is the selected topology realized smoothly at manufacturing resolution?**
  topology-preserving geometric refinement.

## Alternatives considered

| approach | training-free | final topology guarantee | judgement |
|---|---:|---:|---|
| More sparse guidance weights (`sp_hole`, `sp_edt`, etc.) | yes | no | required ablation only; not the main method |
| Per-step SDF projection (CPS/PnP) | yes | only if projection is hard | useful as a sparse ablation, but requires a stable projection/re-encoding operator |
| Replace Direct3D-S2 with another image-to-3D foundation model | yes | no | changes aesthetic prior, not BC/load/topology control |
| Dense SDF upsample + final marching cubes | yes | yes | robust baseline, but limited geometric detail |
| Fixed-connectivity normal deformation | yes | yes | recommended main method |
| Train a conditional sparse structural adapter | no | learned, not exact | longer-term extension after the no-training baseline |

## Literature basis

- Constrained Posterior Sampling projects each posterior estimate onto a hard
  constraint set at every denoising step, rather than trusting a soft guidance
  loss: https://arxiv.org/abs/2410.12652
- Diffusion Plug-and-Play separates a learned prior sampler from a proximal
  consistency sampler. This supports treating structural feasibility as an
  explicit projection operator: https://arxiv.org/abs/2403.17042
- Projection-constrained diffusion explicitly distinguishes soft physical losses
  from exact manifold projection: https://arxiv.org/abs/2602.17773
- Direct3D-S2 provides a sparse SDF prior and a unified sparse VAE, but does not
  provide an image/BC-conditioned guarantee that a manually guided coarse
  topology is retained by a separately sampled fine output:
  https://papers.nips.cc/paper_files/paper/2025/hash/f9666a092153f281b5116cd1f64b5c91-Abstract-Conference.html
- Latent Space Diffusion for Topology Optimization instead trains with dense
  physical fields and connectivity-related objectives. It is evidence that a
  learned structural generator needs structural conditioning; it is not a
  training-free replacement here: https://arxiv.org/abs/2508.05624

## Go/no-go evaluation

For every candidate use the same input image, BCs, envelope, seed schedule and
final FEA. Report:

1. connected components (must be 1), watertightness and containment;
2. BC contact coverage for both fixed and load regions;
3. final-mesh descriptor displacement from the assigned dense QD cell;
4. FEA compliance/stress;
5. multi-view image agreement;
6. diversity across final meshes, not only dense scaffolds.

Do not advance a method that satisfies only dense-stage diversity.
