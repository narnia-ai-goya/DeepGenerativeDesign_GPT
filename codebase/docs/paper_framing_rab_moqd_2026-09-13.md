# RAB-MOQD 논문 프레이밍

## 한 문장

RAB-MOQD는 multimodal generative model과 physics-guided realization pipeline을 하나의 비싼 확률적 black box로 보고, 해석 가능한 형태 공간의 각 영역에서 **강성–재료 Pareto front**를 표본 효율적으로 발견해 단일 최적해가 아닌 검증된 3D design atlas를 반환한다.

## 추천 제목

**From Concepts to Structural Repertoires: Realization-Aware Bayesian Multi-Objective Quality-Diversity for Physics-Guided 3D Design**

짧은 대안:

- **A Design Atlas, Not a Single Optimum: Bayesian Multi-Objective Quality-Diversity for Generative Structural Design**
- **Concept-to-Structure Repertoires with Realization-Aware Bayesian Quality-Diversity**

Advanced Engineering Informatics에는 첫 번째 제목이 가장 직접적이다. CVPR 계열에는 이미지 조건과 3D realization을 더 전면에 둔 세 번째 제목이 맞다.

## 문제 설정

입력은 concept text와 reference image/multi-view conditioning이고, 출력은 하나의 mesh가 아니라 repertoire다. 하나의 생성 recipe `x`는 seed `ω`에 따라 서로 다른 realization을 만든다.

`G(x, ω) → (mesh, b₁, b₂, C, V, validity)`

- `b₁`: normalized void scale — 평균 void-to-material clearance를 설계 영역 대각선으로 나눈 값
- `b₂`: strain-energy concentration — 전체 에너지 크기와 독립적인 하중 전달 경로의 공간적 집중도
- `C`: 검증 FEA compliance, 최소화
- `V`: material volume fraction, 최소화
- `validity`: watertight, connected, in-domain, BC-present, solvable FEA

행동공간의 각 cell `k`는 단일 elite 대신 `(C,V)`의 비지배 Pareto set `Pₖ`를 가진다. 체적은 더 이상 hard feasibility gate가 아니다. 유효 형상은 모두 archive에 들어갈 수 있고, 사용자는 원하는 형태 cell 안에서 강성 중심 또는 재료 효율 중심 해를 사후에 선택한다.

## 핵심 acquisition

후보가 어느 cell에 실현될지, 유효할지, 그리고 그 cell의 Pareto front를 얼마나 개선할지 모두 posterior 아래에서 적분한다.

`α(x) = Eω,y|D [ Σₖ 1(valid) · 1(b(y)∈k) · ΔHV(Pₖ; (C(y),V(y))) ] / cost(x)`

이를 **Expected Joint Hypervolume Improvement of Repertoires**라고 부를 수 있다. BOP-Elites의 probabilistic cell assignment와 MOME의 cell-wise Pareto archive를 잇되, seed에 따른 realization 분포와 adaptive replication을 포함하는 것이 방법의 중심이다.

## 연구 공백

LMTO는 large model의 선호 지식과 topology optimization을 연결해 concept realization을 수행하지만, 하나의 concept에서 얻는 확률적 결과를 형태 공간 전체의 repertoire로 조직하거나 cell별 강성–재료 Pareto front를 표본 효율적으로 찾는 문제를 직접 다루지는 않는다.

BOP-Elites는 비싼 black-box objective와 descriptor를 GP로 모델링해 sample-efficient QD를 수행하지만 기본 문제는 cell당 단일 quality elite다. MOME는 cell마다 Pareto front를 유지하지만, 비싼 생성–후처리–FEA chain과 stochastic realization을 위한 Bayesian acquisition 및 선택적 반복평가가 중심은 아니다.

우리 문제는 세 조건이 동시에 존재한다.

1. 한 번의 3D realization과 FEA가 비싸다.
2. descriptor와 objective가 생성 전에는 알려지지 않는다.
3. 동일 recipe도 seed와 비결정적 sparse 연산에 따라 다른 cell과 성능으로 실현된다.

## 주장할 기여

1. **문제정의:** stochastic concept-to-structure generation을 단일 최적화가 아닌 realization-aware multi-objective QD로 정식화한다.
2. **알고리즘:** descriptor, compliance, volume, validity posterior를 이용한 cell-conditioned expected hypervolume acquisition을 제안한다.
3. **신뢰성:** provisional/verified Pareto archive와 adaptive seed replication으로 우연히 좋은 single realization이 elite가 되는 것을 억제한다.
4. **시스템:** concept conditioning, physics-guided dense/sparse generation, geometry validation, a-posteriori FEA를 닫힌 data acquisition loop로 통합한다.
5. **실증:** 고정된 full-pipeline budget에서 repertoire coverage, QD hypervolume, Pareto quality, 재현성을 비교한다. 이 항목은 향후 반복 실험에서 실제 우세가 확인된 뒤에만 강하게 주장한다.

`first` 또는 `state of the art` 주장은 더 넓은 문헌조사 전에는 사용하지 않는다.

## 기존 원고와의 분리

심사 중인 *Form Embodies Mechanics*의 중심은 physics-guided generative realization pipeline이다. 새 논문에서는 그 pipeline을 연구 대상 optimizer가 아니라 **고비용 stochastic evaluator**로 둔다.

- 기존 원고의 질문: 물리 유도가 생성 형상에 역학을 어떻게 반영하는가?
- 새 논문의 질문: 제한된 evaluation budget에서 어떤 realization들을 평가해야 디자이너에게 유용한 구조 repertoire를 얻는가?

따라서 새 논문의 headline은 generation architecture나 physics loss가 아니라 archive semantics, acquisition, uncertainty, decision support다. `sp_fea_step_size=0.03`과 체적 repair는 시스템 ablation 또는 implementation detail로 둔다.

## LMTO와의 차이

- LMTO: human concept/preference에서 performance-optimal direction으로 구조를 발전시킨다.
- RAB-MOQD: 동일 concept 아래에서 morphology와 engineering trade-off가 다른 여러 검증 구조를 발견하고 지도화한다.
- LMTO의 질문이 “이 concept를 구조로 만들 수 있는가?”라면, 우리의 질문은 “이 concept에서 실현 가능한 구조적 선택지의 지형은 무엇인가?”이다.

교량 사례에서는 Gothic, organic, truss-like 등의 concept가 하나의 승자 구조로 수렴하는 대신, **잘게 분산된 공극 ↔ 큰 아치형 개구**와 **분산된 하중 전달 ↔ 소수의 지배적 하중 경로**의 각 영역마다 강성–재료 Pareto 선택지를 제공하는 그림이 가장 설득력 있다.

`b₂ = 1 - exp(-KL(p || q))`로 두면 `p`는 element strain-energy share, `q`는 element volume share다. 균일하게 에너지를 분담하면 0에 가깝고, 일부 부재에 에너지가 집중될수록 1에 가까워진다. Compliance는 에너지의 총합이고 `b₂`는 정규화된 공간 분포이므로 둘의 역할이 구분된다.

## 현재 데이터가 허용하는 정직한 서술

체적 hard constraint를 제거해 기존 52개를 재분석하면 RA-QD coverage는 1/16에서 12/16으로, random은 2/16에서 13/16으로 증가한다. RA-QD의 QD score 8.288은 random 8.757보다 아직 낮다. 따라서 현재 결과는 우월성 증명이 아니라 다음 두 사실의 pilot evidence다.

1. 좁은 volume gate가 유효한 structural diversity 대부분을 버렸다.
2. 기존 single-objective posterior acquisition은 random보다 낫지 않아 multi-objective cell-wise acquisition이 필요하다.

이 정직한 negative result가 새 방법론의 동기가 된다.

### 2026-09-14 개발 실험이 추가로 보여준 것

No-volume 3-evaluation comparison에서 닫힌 4×4 범위의 QD-HV 증가는 posterior sampling 0.0074, posterior mean 0.0180, Sobol random 0.2855였다. 공동 잔차 bootstrap posterior는 0.1293으로 개선됐지만 target cell 적중은 0/3이었다. 반면 개발 자료의 분위수를 내부 경계로 사용하고 외곽 cell을 열어 두면 8×8에서 각각 0.5028, 0.4381, 0.3414, 0.3224로 순위가 달라졌다. 이는 현재 소표본 실험으로 방법의 우위를 주장할 수 없고, archive 경계와 해상도를 독립 실험 전에 고정해야 함을 보여준다.

반복한 두 recipe는 각각 세 seed가 서로 다른 cell에 도달해 modal-cell agreement가 둘 다 1/3이었다. 따라서 archive의 단위는 검증 후 고정된 mesh로 두고, replication은 이미 검증된 mesh를 폐기하는 gate가 아니라 recipe-level realization noise를 학습하는 용도로 사용한다. 다음 독립 실험은 열린 8×8 분위수 grid, grouped residual calibration, style-diverse batch, 25% frozen Sobol exploration을 사전 고정한다.

## 주 평가 지표

- **Coverage:** 하나 이상의 검증 Pareto solution을 가진 cell 비율
- **QD-HV:** 사전 고정한 reference point를 사용한 cell별 hypervolume의 합
- **Global HV:** 형태 구분을 무시한 전체 Pareto 성능; 다양성을 위해 성능을 희생했는지 확인
- **Pareto precision:** surrogate가 제안한 후보 중 실제 비지배 해가 된 비율
- **Cell hit/calibration:** 예측 cell 확률과 실제 cell의 일치 및 calibration
- **Reliability:** 재평가 시 validity, modal-cell agreement, compliance/volume dispersion
- **Cost:** wall-clock/GPU-hours와 full-pipeline evaluation 수

QD-HV의 compliance와 volume 범위 및 reference point는 결과를 보기 전에 protocol에 고정한다.

## 필수 baseline

1. Scrambled Sobol random + cell-wise Pareto archive
2. MOME/MAP-Elites + 동일 evaluator budget
3. single-objective BOP-Elites 또는 기존 RA-QD
4. Bayesian multi-objective optimization without behavior cells
5. RAB-MOQD full method

Ablation은 probabilistic cell assignment, adaptive replication, exact GP, sparse TO step size를 각각 제거한다. 체적 repair는 중심 방법이 아니므로 optional deployment ablation으로 둔다.

## 초록 골격

Conceptual structural design requires more than a single mechanically optimal shape: designers need distinct, interpretable alternatives and visibility into their stiffness–material trade-offs. Multimodal generative models can produce such alternatives, but their 3D realizations are expensive to validate, their geometric behaviors are unknown before generation, and repeated runs are stochastic. We formulate this setting as realization-aware multi-objective quality-diversity optimization. Our method, RAB-MOQD, models behavioral descriptors, compliance, material usage, and validity with probabilistic surrogates, and selects evaluations by their expected cell-conditioned Pareto hypervolume improvement. A provisional-to-verified archive adaptively repeats only candidates whose cell membership or Pareto status remains uncertain. The resulting design atlas organizes validated 3D structures by interpretable morphology while retaining a stiffness–material Pareto set in every region. [최종 문장에는 반복실험 후 확인된 수치만 삽입한다.]

## 논문에서 피할 프레이밍

- “GPT가 topology optimization을 한다” — 실제 QD 입력은 생성 recipe이고 FEA가 검증한다.
- “체적 0.5를 만족시키는 생성기” — 새 문제의 핵심이 아니며 구조 선택지를 불필요하게 제거한다.
- “random보다 이미 우수하다” — 현재 데이터는 반대다.
- “다양한 이미지를 만들었다” — image diversity가 아니라 verified 3D structural repertoire가 결과다.
- “모든 cell에서 하나의 최고 해” — 각 cell의 Pareto front가 핵심이다.

## 현재 산출물

- `/home/goya/SDL/3d_qd/experiments/bracket/raqd_online_2026-09-13/no_volume_constraint_reanalysis.json`
- `/home/goya/SDL/3d_qd/experiments/bracket/raqd_online_2026-09-13/v2_posterior_audit.json`
- `/home/goya/SDL/3d_qd/experiments/bracket/rcbqd_v2_microbatch_2026-09-13/analysis.json`
- `/home/goya/SDL/3d_qd/codebase/raqd_v2_posterior.py`
- `/home/goya/SDL/3d_qd/codebase/robust_qd_archive.py`
- `/home/goya/SDL/3d_qd/experiments/bracket/behavior_axis_audit_2026-09-13/behavior_axis_audit.json`

## 관련 연구

- Liang et al., *Integrating large models with topology optimization for conceptual design realization*, Advanced Engineering Informatics, 2025. https://doi.org/10.1016/j.aei.2025.103524
- Kent et al., *BOP-Elites, a Bayesian Optimisation Approach to Quality Diversity Search with Black-Box Descriptor Functions*. https://arxiv.org/abs/2307.09326
- Pierrot et al., *Multi-Objective Quality Diversity Optimization*. https://arxiv.org/abs/2202.03057
- Flageat and Cully, *Uncertain Quality-Diversity*. https://arxiv.org/abs/2302.00463
