# DeepGenerativeDesign_GPT

Research code for image-conditioned structural mesh generation and quality-diversity experiments.

The source lives in `codebase/`. Large experiment outputs, mesh caches, model weights, and the
private structural datasets are intentionally excluded. Paths in historical experiment scripts may
need to be configured for a local dataset and Direct3D-S2 installation.

The Shape-QD work is documented in:

- `codebase/docs/shape_aware_dqd_paper_plan.md`
- `codebase/docs/shape_qd_research_position.md`
- `codebase/docs/shape_qd_resolution_plan.md`

The implementation includes frozen shape descriptors, valid prototype-bank construction,
coarse morphology-scaffold guidance, and TSDF parity diagnostics.

Current chair experiments cover image-conditioned Dense/Sparse generation, envelope and
boundary-condition checks, simultaneous seat/back FEA, image-space QD, and a
SNAP3D-inspired structural connector pilot. The proposed generation-time sharing of
repeated chair-part geometry is described in
`codebase/docs/chair_repeated_part_generation_2026_10_04.md`; it is not yet implemented.
