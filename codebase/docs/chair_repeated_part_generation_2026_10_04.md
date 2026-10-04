# Shared-geometry generation for repeated chair parts

## Correction to the SNAP3D-inspired pilot

The colored left/right frames in `other_stages/snap3d_stages_chair.png` are spatial
surface labels on one existing mesh. They are neither learned semantic parts nor
copies of one geometry. The connector experiment also began from an already
generated monolithic chair. It does not satisfy repeated-part identity during
generation.

SNAP3D starts from part meshes and repairs their physical assembly. Exact reuse
of one canonical geometry for repeated chair parts is our proposed additional
constraint, rather than a claim about the published SNAP3D method.

## Generative representation

For a symmetric chair use a semantic part graph with a central seat, a backrest,
one canonical leg, one canonical foot cap, and side-frame/brace families.
Generate each truly repeated geometry **once**, then place its instances with
known rigid transforms. A whole side frame may instead require a left/right
mirror pair:

```
G = {seat S, backrest B, master leg L, master foot F, side family P}
L_i = T_i(L), F_i = U_i(F),  i = 1...4, T_i and U_i are rigid transforms
P_left = V_left(P)
P_right = mirror_x(V_left(P))  when a mirror pair is intended
M = weld_or_union(S, B, all L_i, all F_i, P_left, P_right, joints)
```

The four legs and foot caps are identical rigidly reusable parts by construction.
Mirrored side frames are equal only *up to reflection*: a chiral frame and its
mirror are not necessarily the same manufactured part. If physical part reuse
is required, use rigid placement of an achiral master or keep a separate left/
right pair and report two unique parts. This is stronger than a bilateral loss on a whole-chair
occupancy grid. The selected experiment has `sw=0`; even `sw>0` in the current
Dense generator is only a soft global occupancy penalty and does not share part
parameters. Sparse currently has no corresponding hard shared-part constraint.

## Where it enters the pipeline

1. **Image and design intent:** Specify which parts repeat. A single front
   image may suggest symmetry, but side/back/top checks are needed to establish
   the master part's depth and contacts. AI-generated extra views are proposals,
   not independent geometric measurements.
2. **Dense generation:** Optimize canonical leg/foot fields and seat/back fields,
   plus side-frame families. Compose all placed instances before evaluating
   envelope, BC, image, and FEA losses. Repeated-part active support uses the
   same canonical field under each placement.
3. **Sparse generation:** Decode/refine each canonical part once, transform its
   complete SDF or mesh at every step, and assemble with seat/back. The full
   assembly is used for FEA; gradients from all instances update the same master.
4. **Contacts:** Infer actual interfaces from the generated parts. Optimize
   corresponding left/right connector families with shared geometry and mirrored
   placement parameters. Preserve mandatory fix/load solids and check the
   envelope/keepout before accepting a candidate.
5. **QD:** Vary canonical part morphology *between* candidates while retaining
   exact repetition *within* each candidate. Evaluate full-assembly
   compliance, mass and geometric diversity. Explicit asymmetric designs can
   form a separate design family, not an accidental violation of this constraint.

## Auditable current baseline

Selected mesh:
`/home/goya/SDL/3d_qd/experiments/chair/sofa_style_2026-09-28/text_reasoned_front_axes_2026-10-03/calibrated_image_qd_2026-10-03/llm_text_proposal_round_03_2026-10-03/simultaneous_load_dense_sparse_2026-10-04/existing_generator_fea_on_015_2026-10-04/diagnostics/fea_log_search_2026-10-04/dense_f00010_sparse_f00100/independent_fea_15mm/aligned_full.obj`

For the selected physical-frame mesh, symmetric sample points were checked on
a 10 mm grid over x=0.12–0.26 m, y=-0.24–0.26 m, z=0.12–0.85 m. The reflected
occupancy IoU was **0.9634**. This is a coarse side-region audit, not a semantic
part metric; it shows that the sides are similar but not identical.

## Experiment to validate the method

Keep the same input, seed, BC, envelope and simultaneous 800 N -Z seat / 200 N
+Y back loads. Compare the current unshared baseline, existing soft dense
symmetry, and the proposed hard shared geometry at both Dense and Sparse.
Report left/right correspondence error (should be zero for the shared part),
image projection error, BC/envelope validity, watertightness, compliance, mass,
and realized shape diversity. A 3D-native part segmentation model can initialize
part masks, but surface labels alone cannot substitute for closed solids.

## Relation to SNAP3D

SNAP3D: https://arxiv.org/html/2609.13146v1

Official method: part geometry editing, contact graph, parameterized connectors,
physics-based connector optimization. Shared master geometry is our adaptation
for repeating structural parts in design generation.
