# Bracket QD pilot

The pilot wraps the existing image → guided 3D generation → post-processing →
final FEA pipeline. It does not modify the generator or call an image API.
The implementation uses a small explicit MAP-Elites grid without a new runtime
dependency. Bayesian acquisition and differentiable QD are not implemented.

## Run and resume

```bash
/home/goya/miniconda3/envs/direct3ds2/bin/python /home/goya/SDL/3d_qd/codebase/run_qd_pilot.py --out /home/goya/SDL/3d_qd/experiments/bracket/qd_pilot_2026-09-13 --gpus 0,1,2
/home/goya/miniconda3/envs/direct3ds2/bin/python /home/goya/SDL/3d_qd/codebase/report_qd_pilot.py /home/goya/SDL/3d_qd/experiments/bracket/qd_pilot_2026-09-13
```

Use a fresh output path for a new experiment. Repeating the command resumes saved
plans and completed candidate results. A completed invalid candidate is not retried;
it counts toward the budget. An interrupted candidate without a result file resumes
its recorded pipeline stages. A process lock prevents two coordinators from using
the same output directory. Do not change code, frozen configs, or images mid-run;
the recorded hashes are checked when resuming.

## Frozen experiment

- Three existing six-view image sets: builtin Kagome, bionic bone, and DE trilattice.
  These have mixed historical image-generation provenance. No new image-model
  comparison or specific GPT model version is implied.
- Genome: image-set choice, dense CFG [3,9], sparse CFG [3,7], sparse guidance peak
  [20,80]. Post-processing, material, load, and number of diffusion steps stay fixed.
- MAP-Elites uniformly selects an occupied-cell elite and mutates each numerical
  variable by Gaussian noise with standard deviation 20% of its search range,
  clamped to the legal parameter range. Image choice is resampled with probability
  0.35. An empty archive falls back to random proposals.
- The comparison samples the same image choices and numerical ranges uniformly.
- Three common initial candidates, then two rounds of three candidates per method:
  nine evaluations each, fifteen unique evaluations. Generation seeds are paired
  across methods at each offspring position. Failures count. CUDA nondeterminism
  remains possible despite deterministic flags.
- A fixed 4×4 grid with both descriptor ranges [0,1]. No post-hoc bin tuning.
- The reference is the original CAD envelope minus fixed/load solids, sampled at
  1 mm voxel centers. Descriptor 1 is occupied design samples / all design samples.
  Descriptor 2 is occupied upper-Z design samples / all occupied design samples;
  upper-Z means above the original envelope's height midpoint.
- Lower final compliance wins within a cell. QD score is the sum of
  `1/(1+C/0.01 J)` over occupied cells. This fixed positive transform preserves
  compliance ordering within a cell; the score is specific to this experiment.
- Validity gates: closed single-component mesh, ≥99% containment with half-voxel
  tolerance, successful final FEA, finite compliance strictly between 0 and 1 J.
  This is a geometric/solver gate, not certification of contact, stress safety,
  minimum wall thickness, or image style reproduction.

## Files and interpretation

`protocol.json` records choices and input/code hashes. `jobs_*.json` records all
proposals and parents before each generation. `cases/<id>/result.json` records
fitness, descriptors, validity and paths; per-stage logs and FEA output remain in
the same isolated case directory. `summary.json` stores archives and evaluation
histories. `report.html` is a portable gallery and archive view; its links are
relative for portability and displayed artifact paths are absolute.

Mesh coordinates in OBJ and the evaluator's `final.stl` are metres. Candidate
`gen/final_mm.stl` is explicitly scaled by 1000 for millimetre-oriented CAD tools.

A single nine-evaluation comparison can verify the feedback loop and expose
descriptor collapse or inactive search variables. It cannot establish statistical
superiority. Before a larger benchmark, use independent repeats, account for
failures and wall time, and check whether the fixed bins capture useful geometric
diversity. If bins or variables are changed, start a new named experiment rather
than revising this pilot's reported coverage.
