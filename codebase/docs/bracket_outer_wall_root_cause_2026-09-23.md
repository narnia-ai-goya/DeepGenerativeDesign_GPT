# Bracket outer-wall defects: cause and prevention

The defects are small cavities on the *outer side walls near the four fixed bosses*, not the mounting holes. The input is one top view. The dense mesh has a broadly continuous side wall; the sparse-decoder mesh already contains cavities before the refiner. The side-by-side evidence is at `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/shape_language_angular_2026-09-23/outer_wall_root_cause_2026-09-23/index.html`.

Controlled runs reuse the same dense cache, image, seed, BC and camera. Sparse FEA OFF leaves the cavities. All sparse guidance OFF makes them worse and introduces long side slits. Disabling both BC union and envelope clip in that unguided run still leaves the cavities. Refiner-before and refiner-after views have cavities in similar locations. Laplacian 10/50, topology-preservation 10/30, and minimum-feature-size 10/30 did not clear them. Consequently, the earliest demonstrated source is the sparse generation/decoder prior; FEA, the refiner, and the Boolean operations are not required for the defect to occur.

There is also a concrete *guidance convention error*. `sparse2mesh` initializes unobserved voxels to `+1`, and the MC wrapper unions solid BC by `minimum(volume, peg_sdf)`. Thus material corresponds to **SDF below the extraction level**. Legacy sparse thickness and FEA guidance used `sdf > mc_threshold` as material. The final refiner extracts at `2*mc_threshold` (0.8 here) while the legacy guidance uses 0.4. An opt-in `sp_sdf_inside_low` flag and `sp_sdf_guidance_threshold` now permit controlled comparison without changing legacy runs. Correcting only the level left cavities; correcting sign and level also left cavities in this seed. The sign-corrected raw mesh at iso 0.8 had 56 components; it is not an accepted design. Do not present the lower in-loop FEA surrogate value as independently validated structural improvement.

The most promising non-local-patch method is a **3D feasible-field constraint** during sparse sampling. Let `phi` be a continuous signed-distance field with negative values in material. Optimize the image-conditioned shape subject to:

- exact BC inclusion and envelope containment;
- a global minimum solid ligament measured on the *actual 3D surface*;
- a bounded deviation from the trusted dense surface outside an uncertainty band, with the band derived from dense resolution and conditioning confidence rather than hand-drawn boss coordinates.

A differentiable penalty can guide samples but cannot guarantee feasibility. A projection of the full 3D field into this feasible set would give a stronger guarantee. Because one top image does not observe the side walls, a side-view depth or normal condition is the best additional evidence when available. First prototype the field constraint on this fixed dense cache, then test multiple images/seeds. Accept a result only if it improves both side walls, preserves intended open voids, yields one connected final mesh, and passes independent FEA, mass, and envelope checks.

Reproducible candidate configs and raw meshes:

- `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/shape_language_angular_2026-09-23/outer_wall_sign_corrected_2026-09-23/iso04/`
- `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/shape_language_angular_2026-09-23/outer_wall_sign_corrected_2026-09-23/iso08/`
- `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/shape_language_angular_2026-09-23/outer_wall_prior_no_boolean_diagnostic_2026-09-23/`

These are diagnostic raw meshes, not replacements for the accepted QD archive or post-processed final mesh.

## Sparse-stage prevention experiments (2026-09-23)

The comparison page is `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/shape_language_angular_2026-09-23/sparse_hole_prevention_2026-09-23/index.html`. All candidates below reused the same dense cache, seed, BC and FEA-on sparse configuration. New generator options default to disabled.

An SDF loss that keeps the dense interior solid (0.5 mm margin) reduced the number of raw disconnected components, but did not remove the small wall cavities. A CVaR version concentrated the gradient on the worst 0.1% or 0.5% of interior violations. The resulting raw meshes had 7 and 17 components, respectively, versus 56 for the top-only sign-corrected baseline, but still had cavities; their volumes fell from 546.2 to 523.7 and 480.8 cm³. The CVaR loss approached 0.30486 late in sampling, corresponding to an SDF violation near the decoder's upper value of 1.0. This suggests saturated or otherwise unguidable samples dominate the worst-case objective; the exact gradient blockage remains to be verified. Do not increase CVaR weight based on component count alone.

A hard dense-core projection and a full 3D dense SDF trust field suppressed some cavities but produced conspicuous low-resolution stair steps and substantial volume changes (9.8–20.2% increases for the trust field). These are not accepted solutions.

Conditioning sparse generation on top plus two side renders of the same dense mesh was visually best and reduced the raw component count to 3 while retaining the open structure. Small wall pits remained. The current multi-view encoder concatenates image tokens without camera-pose geometry, so it cannot directly supervise which 3D surface is visible in each side view. A proper next implementation is a pose-calibrated side-view silhouette/depth or normal loss on the sparse surface during denoising, plus an explicit 3D wall-continuity constraint. Evaluate raw mesh, post-processed final mesh, mass and independent FEA before adopting it.
