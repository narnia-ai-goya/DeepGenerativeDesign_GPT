# CVPR / Advanced Engineering Informatics 실험 계획

> 프레임워크 특화 QD의 최신 정의는 `/home/goya/SDL/3d_qd/codebase/docs/realization_aware_qd_definition_2026-09-13.md`를 따른다. 목표 의도 archive와 실제 최종 구조 archive를 분리하고, 논문의 coverage와 QD score는 최종 검증 archive에서만 계산한다.

작성일: 2026-09-13. 이는 앞으로 수행할 연구 계획이며, 아래 대규모 실험이나 제안 모듈을 실행·검증했다는 뜻이 아니다. 진행 중인 bracket 파일럿과 구분한다.

## 1. 목표와 투고 방향

우선 AdvEI에 맞는 공학적 검증을 구축하고, 독립적인 vision 방법 기여가 확인되면 CVPR 2027로 확장한다. 단순한 GPT 이미지 생성 + 3D 생성 + FEA + MAP-Elites 연결은 충분한 신규성으로 가정하지 않는다. 두 저널/학회에 동일 원고를 동시에 제출하는 계획이 아니라, 공통 실험 기반 위에서 한 경로를 선택하는 계획이다.

중심 연구 질문:

> 이미지 조건부 3D 생성에서 중간 형상과 최종 형상의 차이를 예측·보정하면, 같은 계산 예산으로 더 다양한 물리적 유효 설계를 확보할 수 있는가?

| 경로 | 논문의 중심 | 추가로 필요한 증거 |
|---|---|---|
| AdvEI 우선 | 설계 지식·제약을 반영한 재현 가능한 설계 대안 탐색과 선택 지원 | 여러 CAD 문제, 해석 검증, 강한 설계 최적화 비교, 실무적 trade-off |
| CVPR 조건부 | 최종 3D 특징을 제어하는 일반적인 생성/탐색 방법 | 다른 3D 생성 모델에서도 효과, 보지 않은 CAD·스타일, 시각적 의미 보존, 강한 정보·비용 일치 비교군 |

CVPR 2027 공식 일정: 등록 2026-11-10 AoE, 본문 2026-11-16 AoE, 보충자료 2026-11-23 AoE. 이는 확인된 일정이며 아래 목표 실험 규모는 학회가 요구하는 최소 수량이 아니라 우리의 계획이다. [공식 일정](https://cvpr.thecvf.com/Conferences/2027/Dates), [CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)

## 2. 3D 선행연구와 차별화의 정확한 범위

LMTO (AdvEI 2025)는 실제 3D 선행연구다. §5.2는 Shap-E와 3D Soft-Kill BESO를 사용하며, Fig. 7은 3D 교량·출력물을 제시한다. 따라서 '기존에는 2D만 있었다'는 주장은 사용하지 않는다. [저자 원문](https://structoptlab.github.io/files/2025-Integrating%20large%20models%20with%20topology%20optimization%20for%20conceptual%20design.pdf)

로컬 원문: `/home/goya/SDL/3d_qd/references/LMTO_2025_Liang_3D_topology_optimization.pdf`

| 선행연구 | 이미 다룬 범위 | 우리 실험이 추가로 검증할 질문 |
|---|---|---|
| LMTO | 생성 prior와 3D 구조 최적화 | 고정 예산에서 검증된 최종 설계 집합을 체계적으로 확보하는가? |
| QDID / OIDD | QD와 구조/토폴로지 최적화 | 이미지 조건부 생성 prior가 탐색 가능한 형상과 비용에 어떤 영향을 주는가? |
| LSI / DQD | 생성 latent에서 QD, 미분 기반 QD | 우리의 전체 비미분 파이프라인에서 중간 proxy와 최종 지표 차이를 어떻게 처리하는가? |
| BOP-Elites / constrained mixed-variable Bayesian QD | 목적·특징·성공 가능성을 예측하는 비싼 탐색 | 단순 Bayesian 모델보다 단계별 형상 정보와 보정이 더 나은가? |
| Uncertain QD | 확률적인 목적·특징과 재평가 | 생성 recipe의 변동과 고정된 설계 artifact의 품질을 구분하는가? |

방법의 최초성은 아직 확정하지 않는다. 특히 multi-fidelity 최적화 자체도 신규하다고 주장하지 않으며, 최종 형상 제어를 위한 구체적 방법과 비교 결과가 필요하다. 기존 조사: `/home/goya/SDL/3d_qd/codebase/docs/qd_literature_review_2026-09-13.md`

## 3. 현재 확보한 것과 아직 없는 것

현재 구현: 고정된 이미지 bank, 설정 변이, 4×4 MAP-Elites, 동일 예산 random, 최종 mesh/FEA 평가, 입력 해시, 실행 기록, mm STL, 아카이브 보고서.

현재 파일럿은 bracket 한 문제, 방법당 9회, 공유 초기 후보 3개를 포함한 실제 생성 15개다. **개발용 자료**로만 사용한다. 대규모 성능 주장이나 방법 선택의 독립 검증 자료로 사용하지 않는다. 결과의 최종 상태는 다음 파일을 따른다.

`/home/goya/SDL/3d_qd/experiments/bracket/qd_pilot_2026-09-13/summary.json`

로컬 기존 원고 `/home/goya/SDL/3d_qd/references/21210_Form_Embodies_Mechanics_.pdf`는 이미 inference-time 물리 가이드 3D 생성을 다루며, **사용자가 2026-09-13 현재 심사 중이라고 확인했다.** 새 원고에서는 기존 생성기와 물리 가이드를 기반 방법으로 구분하고, 최종 상태 QD 제어와 설계 집합 평가의 추가 기여를 명확히 한다. 기존 원고의 수치를 이번 실험에서 재검증한 결과처럼 사용하지 않는다. 새 연구 진행 자체를 심사 종료까지 미루지는 않되, 제출 전 두 원고의 기여·실험 중복을 비교하고 각 venue의 관련 원고·동시 투고·익명성 규정을 확인한다. 기존 원고가 이미 공개되거나 출판됐다고 가정하지 않는다.

아직 없는 것: 아래 제안 모듈, 강한 Bayesian 비교군, 독립 CAD 테스트 세트, mesh convergence/하중 검증, 생성 recipe의 반복 통계, 다른 3D backbone에서의 검증.

관측된 문제는 별도 가설로 다룬다: 기본 격자의 충돌, 실제로 비활성인 volume target, 최종에서 사라지는 격자 스타일, CUDA 비결정성. 이들을 해결하지 않고 평가 수만 늘리지 않는다.

## 4. 제안 방법의 후보 구성

전체 파이프라인을 미분가능하다고 가정하지 않는다. GPT 내부 latent/gradient도 사용 가능한 것으로 가정하지 않는다.

1. **최종 특징 예측·보정**: dense와 sparse의 형상 특징, 현재 설정, 목표 특징으로 최종 descriptor와 성공 확률을 예측한다. 중간 특징을 그대로 최종 특징으로 취급하는 방법과 비교한다.
2. **목표 구역을 향한 후보 생성**: 개발 실험에서 검증한 연속 proxy guidance 또는 설정 변이를 사용하고, 예상되는 후처리 이동을 보정한다. 단순 CFG 변이 이상의 제어 효과가 있어야 한다.
3. **비용을 고려한 평가 선택**: 후보가 새 구역을 채우거나 현재 elite를 개선할 기대 이득과 불확실성을 이용해 정밀 평가를 배정한다. 탈락 후보 중 사전 고정된 10%를 무작위로 끝까지 평가해 잘못된 탈락을 측정한다. 이 감사 비용도 예산에 포함한다.
4. **검증된 최종 artifact 저장**: archive의 위치와 품질은 실제 최종 mesh/FEA로만 확정한다. 예측 archive는 별도 데이터로 둔다. 이 원칙은 실험 기반이며 단독 신규성으로 내세우지 않는다.

가설은 개발 실험으로 반박 가능해야 한다. 중간 특징이 최종을 잘 예측하지 못하거나 비용 절감이 없으면, 억지로 multi-fidelity 모듈을 유지하지 않는다.

## 5. 데이터와 분할

- 사용자와 의자·교량 사례를 추가하는 방향으로 갱신: 기본 4계열은 bracket, caliper, chair, bridge다. 기존 산업 부품 CAD와 새로 준비할 중립 의자·교량을 개발용으로 배정한다. Motor mount와 link는 후속 OOD 평가 후보로 둔다.
- AdvEI 기본안: **별도 CAD 8개** 확보, 계열당 2개. 총 개발 4개 + 테스트 8개. 아직 이 독립 CAD들이 확보돼 있다는 뜻은 아니다.
- 스케일 변환·회전·remesh·같은 CAD의 파생 파일은 독립 CAD로 세지 않는다. 유사 파생 형상은 같은 split에 둔다.
- 테스트 CAD별 nominal 하중/재료/고정 조건을 사전에 고정한다. 추가 하중 방향과 크기 변화는 별도 robustness 테스트다.
- CAD마다 6개 스타일 × 6개 view의 고정 이미지 세트를 준비한다. 12 CAD 기준 432 PNG다. 모든 방법에 동일한 bank를 제공하고 수작업 선별 규칙을 고정한다.
- 일부 스타일 계열은 보정 모델 개발에서 제외해 스타일 일반화를 평가한다. 테스트 출력에 맞춰 새 prompt를 반복 편집하지 않는다.
- 이미지 생성 backend를 확인할 수 없다면 특정 GPT 모델 버전명을 쓰지 않는다. PNG·prompt·도구·생성일·SHA256을 공개 가능한 범위에서 기록한다. **고정 입력 bank의 재현**과 **이미지 생성 자체의 재현**을 구분한다.
- 회귀/보정 모델은 개발 CAD와 허용된 online 평가 이력만 이용한다. 테스트의 아직 평가하지 않은 최종 결과는 학습·범위 설정·선별에 사용하지 않는다.

의자는 최종 3D 의미 보존을, 교량은 하중 경로와 공간적 재료 배치를 검증하는 대표 사례다. 8개 test CAD의 총 평가 수는 유지하되, **산업 부품 4개 + 독립 의자 문제 2개 + 독립 교량 문제 2개**로 구성한다. 새 사례의 실제 생성·FEA 비용은 아직 측정하지 않았다. 상세 사양은 `/home/goya/SDL/3d_qd/codebase/docs/chair_experiment_plan_2026-09-13.md`와 `/home/goya/SDL/3d_qd/codebase/docs/bridge_experiment_plan_2026-09-13.md`에 기록한다. 가구 후보 순위는 `/home/goya/SDL/3d_qd/codebase/docs/furniture_candidates_2026-09-13.md`를 따른다.

## 6. 평가기와 특징 공간을 먼저 고정

### 물리·형상 검증

- 단위, 총 하중, 재료, 구속 조건, design/non-design 영역을 명시한다. 하중이나 재료를 optimizer가 바꾸게 하지 않는다.
- dense/sparse의 SDF 부호 규약, proxy loss의 gradient 방향, 가능한 범위의 finite-difference 일치를 점검한다. 기존 topology 관련 부호 옵션의 실험 기록을 검토하고, 버그 수정과 알고리즘 기여를 구분한다. 테스트 비교군 사이에 서로 다른 수정 상태를 섞지 않는다.
- 최종 mesh의 watertightness, 연결성, 설계 영역 포함률, 실제 fixture/load 접촉 영역의 유지·연결을 확인한다. 현재 전체 BC 표면 근접률은 접촉 검증의 대체물이 아니다.
- 개발용 대표 형상 12개에서 3단계 tetrahedral mesh 해상도를 비교한다. compliance 변화 <5%를 초기 수렴 판정 목표로 두되 달성 여부와 기준의 한계를 공개한다.
- 반력 평형, `f^T u`와 `2U`의 일치, 경계조건 선택을 검증한다. 서로 다른 논문의 compliance 정의를 그대로 섞지 않고 동일 verifier로 다시 계산한다.
- 최대 응력은 mesh/하중 특이점 영향을 조사한다. 응력 제한을 주장하려면 재료 허용치와 수렴한 응력 측정법을 먼저 정한다.
- 두께는 현재 EDT ridge 근사만으로 제조 통과 판정하지 않는다. 최소두께를 주장하려면 독립적인 측정·해상도 검증을 추가한다.

### Descriptor와 archive

- 주 descriptor 1: 고정 design-only 영역의 재료 체적비.
- 주 descriptor 2 후보: CAD 공간 수용량으로 보정한 상·하부 점유 분포, 또는 사전에 정한 구조 기준축에 대한 정규화 2차 모멘트. 개발 CAD의 민감도 실험으로 하나를 선택한다.
- 현재 raw upper-share는 CAD 자체의 상부 체적과 강하게 결합한다. 이를 그대로 모든 CAD에 적용하지 않는다.
- 주 archive는 16 cells로 시작한다. 개발 자료와 CAD의 물리적 도달 범위로 특징 정규화와 grid/CVT를 결정하고 테스트 전에 고정한다.
- 모든 방법은 동일한 descriptor, cell, feasibility gate를 사용한다. 테스트 결과를 보고 구간을 축소하거나 coverage 분모를 바꾸지 않는다.
- 방향축은 CAD/하중 기준으로 사전 정의한다. 임의의 파일 좌표 방향으로 서로 다른 문제의 의미가 바뀌지 않게 한다.
- raw genus는 주 다양성 지표로 사용하지 않는다. 큰 개구부와 재료 배치의 차이를 독립적으로 확인한다.

## 7. 핵심 실험표

| ID | 질문 | 실험 | 주 지표 / 판정 |
|---|---|---|---|
| E0 | 평가기와 descriptor를 믿을 수 있는가? | 개발 CAD, 반복 생성, mesh refinement, BC/에너지 검증 | 해석 오차, 셀 이동, feature 민감도 |
| E1 | 같은 예산에서 더 좋은 설계 집합을 얻는가? | 8 test CAD × 4 방법 × 5 독립 탐색 반복 × 32 평가 | 최종 verified QD score, coverage, cost-quality curve |
| E2 | 최종 형상 보정이 실제 필요한가? | 중간 특징 그대로 / 단순 회귀 / 제안 보정; 공통 후보·holdout CAD | 최종 descriptor 오차, 목표 cell 적중률, 신뢰도 보정 |
| E3 | 어떤 구성 요소가 기여하는가? | 보정 제거, 선별 제거, 같은 저충실도 정보를 쓰는 Bayesian 비교, 내부 물리 가이드 제거 | 동일 예산 E1 지표, 실패율, 실제 계산 비용 |
| E4 | 외관의 다양성이 최종 3D에 남는가? | 고정 조명·카메라의 최종 multi-view 렌더와 geometry | 의미 일치, 형태 거리, macro-hole, 필요시 블라인드 전문가 평가 |
| E5 | 결과가 반복 생성·하중 변화에 견디는가? | 사전 규칙으로 뽑은 recipe 재생성, 고정 mesh의 추가 하중 해석 | recipe 성공률/셀 이동, 강성 변화, archive 재평가 |
| E6 | 기존 구조 설계 방법보다 유용한가? | multi-start SIMP/BESO 및 재현 가능한 LMTO/QDID류 비교 | 질량 조건별 강성, 설계 집합, 검증 비용, 동일 wall-time 비교 |
| E7 | CVPR 수준의 일반성이 있는가? | 미관측 CAD 확대, 두 번째 공개 image-to-3D backbone, 별도 스타일 | 방법의 전이성, 형태/의미 제어, backbone별 비용 |

E1의 네 방법은 (1) Sobol sampling + 동일 archive, (2) vanilla MAP-Elites, (3) constrained mixed-variable Bayesian QD, (4) 제안 방법이다. 현재 random 파일럿은 개발 비교로 보존한다. 구현 이름을 선행연구 알고리즘 이름과 혼동하지 않도록 충실 재현인지 변형인지 표시한다.

E3의 **같은 저충실도 정보에 접근하는 Bayesian 비교군**은 필수다. 추가 정보를 사용한 효과를 새로운 알고리즘의 효과로 오인하지 않도록 한다. 품질만 최적화하는 BO + 사후 archive도 4개 문제의 보조 실험에 포함한다.

E6에서는 단일 SIMP 결과 하나와 수십 개 QD 후보를 비교하지 않는다. 같은 예산의 다중 초기화·다중 조건 후보 집합을 만들고 동일한 최종 verifier를 쓴다. 직접 재현이 불가능한 방법은 'inspired baseline'으로 명시하며 원 논문의 성능 재현을 주장하지 않는다.

## 8. 지표·통계·공정한 예산

주 지표는 최종 검증 QD score의 예산별 곡선과 종료값이다. 품질은 문제별 고정 참조 compliance `C_ref`로 `q=1/(1+C/C_ref)`와 같이 양수로 정규화한다. 참조와 변환은 테스트 전에 고정한다.

함께 보고할 것:

- 같은 archive의 coverage, cell별 compliance 분포, feasible rate, best compliance와 그 설계의 질량.
- 생성·후처리·meshing·FEA·기하 gate별 실패율. 실패/재평가도 비용으로 계산한다.
- 완전 평가 횟수, 모든 저충실도 평가 수, GPU/CPU 실제 시간, surrogate 학습·선별 시간, 공통 이미지 bank 생성 비용.
- 방법별 동일 완전 평가 예산 곡선 **및** 동일 실측 wall-time 곡선. 조기 탈락이 많아 발생하는 숨은 생성 비용을 누락하지 않는다.
- 목표 cell 적중률, 중간→최종 cell 이동, 최종 특징 회귀 오차와 불확실성 신뢰도, 잘못된 탈락률.
- 최종 렌더의 고정 시각 encoder 지표와 기하 지표. 입력 이미지의 CLIP 점수만으로 최종 3D 다양성을 주장하지 않는다.
- 저장된 최종 mesh의 재해석과 동일 recipe의 재생성 결과를 구분한다. 후자는 별도 확률적 성능이다.

반복 단위는 후보 mesh가 아니라 **CAD 문제와 독립 탐색 run**이다. 같은 초기 집합/seed block으로 방법을 pairing하고, CAD·run 계층을 반영한 95% 신뢰구간과 paired effect size를 보고한다. 수천 후보를 독립 표본 수로 세지 않는다. 5회 반복은 초기 계획이며 pilot 분산으로 검정력을 확인하고 부족하면 사전 규칙에 따라 늘린다.

처음 8개의 초기 완전 평가는 각 방법의 32회 예산에 포함한다. 공통 초기 결과를 계산상 재사용한다면 논리 평가 예산과 실제 중복 제거 비용을 모두 기록한다. 중간 보고 checkpoint는 8/16/24/32로 고정한다.

## 9. 계산 예산: 먼저 현실적인 규모로

계획 환산 기준은 완전 파이프라인 1회 **13분**, GPU 3개, 30% 운영 여유다. 이는 초기 로컬 관측을 이용한 추정이며 다른 CAD/backbone의 실측 보장은 아니다. 아래 시간은 pipeline 전체 동안 GPU를 예약한다고 가정한 환산치로, 실제 GPU 연산시간 측정값과 구분한다. 공유 초기 평가에 따른 절감은 보수적으로 차감하지 않았다.

| 묶음 | 신규 full pipeline 수 | 이상적인 3 GPU wall time | 30% 여유 포함 |
|---|---:|---:|---:|
| 개발/보정 자료 | 256 | 0.77일 | 1.00일 |
| E1 기본 비교: 8×4×5×32 | 5,120 | 15.41일 | 20.03일 |
| E3 추가 ablation: 4변형×4문제×3run×32 | 1,536 | 4.62일 | 6.01일 |
| recipe 재생성: 64 recipe×3회 | 192 | 0.58일 | 0.75일 |
| 품질만 최적화하는 BO: 4문제×3run×32 | 384 | 1.16일 | 1.50일 |
| 기본안 합계 | **7,488** | **22.53일** | **29.29일** |

고정 mesh의 추가 FEA, 구조 TO baseline, 저충실도 감사 표본, 이미지 bank 생성, 구현 시간은 위 full-pipeline 환산 합계에 포함되지 않는다. 별도 계측 후 예산표에 더한다. 특히 저충실도 선별을 '무료'로 가정하지 않는다.

CVPR 확장 후보:

- 독립 test CAD를 8→12개로 확대: 2,560 full evaluations 추가, 같은 가정 약 10.0일.
- 두 번째 backbone: 4문제×2방법×3run×32 = 768회. 같은 속도라고 가정하면 약 3.0일이지만 실제 속도는 먼저 측정한다.
- 64회 예산 확대: 4문제×2방법×5run에서 32회씩 연장 = 1,280회, 약 5.0일 추가.

확장을 모두 수행하면 계산만 약 47일 + 별도 비용이 필요하다. 따라서 11월 투고를 위해 무조건 전부 수행하는 계획은 위험하며, 10월 초의 방법 검증으로 경로를 선택한다. 구체 계산은 `/home/goya/SDL/3d_qd/codebase/docs/publication_experiment_budget_2026-09-13.csv`에 기록한다.

## 10. 일정과 진행/중단 기준

| 기간 | 작업 | 다음 단계 조건 |
|---|---|---|
| 9/13–9/20 | 현재 pilot 종료, 독립 CAD 확보, evaluator/descriptor/volume 제어 점검, LMTO 등 차이 정리 | 해석과 특징 정의가 신뢰 가능하고 독립 데이터 확보 경로가 있음 |
| 9/21–10/4 | 개발 자료와 최종 특징 보정, 강한 Bayesian baseline, 소규모 ablation | holdout 개발 CAD에서 보정 효과와 비용상 이득이 반복 관측됨 |
| 10/5 | 투고 경로 결정 | vision 방법 기여·전이성 증거가 부족하면 AdvEI에 집중 |
| 10/5–11/5 | 동결한 프로토콜의 본 실험, CPU 물리 검증, 필요한 ablation; 원고 작성 병행 | 결과를 보고 baseline·descriptor·test split을 바꾸지 않음 |
| 11/2–11/9 | 통계·실패 사례·도표·원고, 재현 패키지 | 주장과 실제 결과 일치, 누락 비용 확인 |
| 11/10–11/16 | CVPR 선택 시 공식 일정에 맞춰 등록·제출 | 제안 방법이 기본 조합 이상의 기여를 입증했을 때만 진행 |
| AdvEI 선택 시 | 추가 설계 검증·실무 평가 후 원고 완성 | 정해지지 않은 저널 마감이나 acceptance 가능성을 가정하지 않음 |

개발 단계의 사전 목표 예시: 최종 descriptor 오차 20% 감소 또는 동일 verified QD 수준의 비용 20% 감소. 이는 채택 보장이 아닌 연구 지속 여부를 판단할 목표다. 효과가 작거나 신뢰구간이 넓으면 계산 규모 확대보다 원인 분석을 우선한다.

최소 논문 도표: 방법 개요, 동일 예산 QD 곡선, CAD별 최종 archive, dense/sparse/final 특징 이동, ablation, 물리/시각 실패 사례. 예쁜 결과만 선별한 그림으로 대체하지 않는다.

## 참고 자료

- [LMTO, AdvEI 2025](https://structoptlab.github.io/files/2025-Integrating%20large%20models%20with%20topology%20optimization%20for%20conceptual%20design.pdf)
- [QDID, 2024](https://doi.org/10.1145/3670693)
- [OIDD, 2024](https://arxiv.org/abs/2407.07591)
- [DQD, NeurIPS 2021](https://papers.nips.cc/paper_files/paper/2021/hash/532923f11ac97d3e7cb0130315b067dc-Abstract.html)
- [BOP-Elites](https://arxiv.org/abs/2307.09326)
- [Constrained mixed-variable Bayesian QD, 2024](https://www.sciencedirect.com/science/article/pii/S0952197624002768)
- [Uncertain QD](https://arxiv.org/abs/2302.00463)
- [CVPR 2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)
- [CVPR 2027 dates](https://cvpr.thecvf.com/Conferences/2027/Dates)
