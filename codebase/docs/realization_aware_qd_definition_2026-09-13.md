# 이 프레임워크를 위한 Realization-Aware Quality-Diversity 정의

작성일: 2026-09-13

> **구현 상태:** RA-QD v0 자료 구조, descriptor 분석, 체적 제어 보정과 target-directed probe를 실행했다. 최초 probe는 목표 셀 적중 0/3, 체적 제약 통과 0/3으로 실패했으며 방법의 효과는 아직 검증되지 않았다. 결과는 `/home/goya/SDL/3d_qd/experiments/bracket/raqd_v0_2026-09-13_findings.md`에 기록했다.

## 1. 문제의 출발점

이 프레임워크의 후보는 일반적인 QD처럼 하나의 유전자에서 하나의 phenotype으로 직접 대응하지 않는다.

```text
텍스트 + 참조 이미지 + 생성 설정
→ 확률적 3D 생성
→ 물리 가이드 및 topology optimization
→ 후처리와 mesh 생성
→ 최종 FEA 검증
```

따라서 사용자가 의도한 개념과 최종 구조 사이에 **realization gap**이 생긴다. 같은 recipe도 확률성 때문에 다른 최종 구조를 만들 수 있고, 입력 이미지에서 보인 특징은 topology optimization 중 사라지거나 다른 구조 특징으로 변할 수 있다.

표준 QD의 목적은 행동 공간의 각 niche에서 고품질 해를 찾는 것이다. 이 연구에서는 이를 다음과 같이 특수화한다.

> **Realization-Aware QD(RA-QD)는 서로 다른 설계 의도가 실제 최종 3D 구조로 얼마나 도달 가능한지를 학습하면서, 각 실현 가능한 구조 niche에서 물리적으로 검증된 최상의 설계 recipe를 찾는 탐색이다.**

`Realization-Aware QD`는 현재 작업명이다. 2026-09-13의 예비 정확 구문 검색에서는 동일한 명칭을 사용하는 직접적인 선행연구를 찾지 못했지만, 이는 신규성 확인을 대신하지 않는다. 투고 전에는 체계적인 문헌 검색과 명칭 검토가 필요하다.

## 2. 후보와 최종 결과의 정의

문제 인스턴스 `p`는 설계 영역, keep-in/keep-out, 하중, 지지, 재료와 해석 설정을 포함한다.

- 설계 의도 `c`: 텍스트, 참조 이미지, 구조 어휘와 목표 특징.
- recipe `x = (c, z, θ_g, θ_t)`: 설계 의도, 생성 seed, 3D 생성 설정, topology optimization 설정.
- 저비용 상태 `l(x)`: 입력 이미지 특징, dense voxel/SDF, 중간 물리량과 생성 로그.
- 최종 결과 `y = F_p(x; ξ)`: 최종 mesh, 고정 카메라 렌더, FEA 결과. `ξ`는 파이프라인의 확률성을 나타낸다.
- 목표 descriptor `b_T(c)`: 설계자가 요청한 목표 niche.
- 실현 descriptor `b_R(y,p)`: 최종 mesh와 검증 결과에서 다시 측정한 niche.
- realization gap `Δ(x,y) = d(b_T(c), b_R(y,p))`.

Archive 삽입 셀은 항상 `b_R`로 결정한다. Prompt label이나 목표 셀만으로 coverage를 계산하지 않는다.

## 3. 두 개의 archive와 전이 모델

### 3.1 Intent archive

`A_T`는 어떤 목표 셀에 어떤 recipe를 시도했는지 기록한다. 이 archive는 탐색 이력과 실패 원인 분석에 사용하며 논문의 최종 coverage로 보고하지 않는다.

### 3.2 Verified realization archive

`A_R`은 최종 mesh, 유효성 검사와 FEA를 통과한 결과만 저장한다. 논문의 coverage, QD score와 설계 repertoire는 이 archive에서 계산한다.

각 셀 `k`의 elite는 다음 조건으로 선택한다.

```text
A_R[k] = argmin compliance(y)
         subject to:
           b_R(y,p) ∈ niche k
           volume error ≤ ε_v
           geometry/FEA validity = true
           semantic preservation ≥ τ_s
           domain constraints = satisfied
```

최소 두께, 제조 조건 또는 robust load 조건을 주장할 경우에도 가중합 점수에 숨기지 않고 명시적인 제약으로 추가한다.

### 3.3 Intent-to-realization 전이

탐색은 다음 전이 분포를 학습한다.

```text
P(b_R, quality, validity | b_T, recipe, low-fidelity state, problem)
```

목표 셀 `i`를 요청했을 때 실제 셀 `j`에 도착한 횟수로 전이 행렬 `T[i,j]`를 만들 수 있다. 대각 성분은 target hit rate이며, 비대각 성분은 설계 의도가 어느 방향으로 무너지는지를 보여준다.

이 전이 모델이 현재 파이프라인에 특화된 QD의 핵심이다. 단순 MAP-Elites처럼 부모 recipe를 무작위로 변형하지 않고, 비어 있거나 품질이 낮은 **최종 셀**에 도달할 확률과 물리 성능을 함께 예측해 다음 평가를 고른다.

## 4. 다양성과 품질의 역할 분리

### 다양성

다양성은 최종 3D 구조에서 측정한다. 공통 archive에는 서로 독립적이고 정규화 가능한 두 축을 사용한다.

1. macro-void openness: 설계 영역 대비 연결된 거시적 빈 공간의 비율.
2. material organization: 정규화된 재료 중심, 주관성비 또는 방향성 중 개발 데이터에서 가장 안정적인 하나.

체적은 우선 고정 제약으로 둔다. 체적이 descriptor와 품질 양쪽에 동시에 영향을 주어 고체에 가까운 후보가 유리해지는 현상을 방지한다.

Domain-specific 특성은 별도의 보조 archive 또는 분석표로 둔다.

- 교량: 아치 높이, 하부 통과 공간, 횡방향 연결도, 비대칭 하중 응답.
- 의자: 좌판·등받이·하부 지지 사이의 재료 배분, 구조 분기, 최종 렌더의 의미 보존.
- bracket/caliper: load-support 경로 분산, 개구부 구조, 인터페이스 유지.

### 품질

주 품질은 고정 체적과 동일 verifier에서 계산한 최종 compliance다. 문제 간 집계에는 각 문제의 고정 reference로 정규화한 값을 사용한다. Stress, displacement, minimum-thickness 진단과 의미 보존은 사전에 정한 제약 또는 별도 Pareto 지표로 보고한다.

## 5. Framework-specific emitter

한 종류의 숫자 mutation 대신 실패 원인에 따라 세 emitter를 사용한다.

1. **Concept emitter**: 구조 어휘, 참조 이미지 또는 국소 이미지 편집을 바꾸어 목표 형태를 이동한다.
2. **Realization emitter**: 이미지 의도를 유지한 채 3D seed, conditioning과 생성 설정을 바꾼다.
3. **Physics emitter**: 생성 개념을 유지한 채 topology guide schedule과 물리 최적화 설정을 바꾼다.

각 평가 후 `목표 → dense 중간 상태 → 최종 구조`의 변화를 보고 어느 emitter가 필요한지 선택한다. 의미가 처음부터 부족하면 concept emitter, 이미지에는 있으나 3D에서 사라지면 realization emitter, TO에서 사라지면 physics emitter에 신호를 준다.

## 6. 비용을 반영한 후보 선택

전체 파이프라인이 비싸므로 다음 acquisition을 개념적으로 사용한다.

```text
획득값(x) =
  새 verified 셀에 도달할 확률
  × 셀 내부의 기대 성능 개선
  × 최종 유효성 확률
  ÷ 예상 계산 비용
```

Surrogate는 후보 순서를 정할 뿐이다. Archive 삽입과 최종 논문 수치는 전체 후처리와 FEA를 마친 결과만 사용한다. Surrogate가 거절한 후보 중 사전 고정한 비율을 무작위로 전체 평가해 filtering bias도 측정한다.

## 7. 이 정의에 필요한 핵심 평가 지표

- **Verified coverage**: `A_R`이 채운 최종 셀 비율.
- **Verified QD score**: 유효한 최종 elite의 정규화된 품질 합.
- **Target hit rate**: 요청한 셀과 실제 셀이 일치한 비율.
- **Realization drift**: 목표와 최종 descriptor 사이 거리.
- **Reliable coverage**: 동일 recipe 반복에서 같은 셀 또는 인접 셀에 안정적으로 도달한 elite만 센 coverage.
- **Cost per new verified cell**: 새로운 최종 셀 하나를 얻는 데 사용한 전체 생성·후처리·FEA 시간.
- **Semantic preservation**: 입력 의도가 최종 고정-view 렌더와 3D 형상에서 유지되는 정도. 자동 점수만 사용하지 않고 일부 표본의 blinded human assessment와 함께 검증한다.

## 8. 기존 파일럿을 활용하는 방법

현재 bracket 파일럿은 `A_T`와 `A_R`을 구분하지 않고 scalar parameter mutation을 사용한 **naive parameter MAP-Elites baseline**으로 보존한다. 16셀 중 2셀만 채워졌고 랜덤 탐색이 더 좋은 최저 compliance와 QD score를 얻었다. 이는 RA-QD의 성공 결과가 아니라 다음 두 가설을 검증할 출발점이다.

1. 최종 상태를 예측하는 target-directed selection이 verified coverage와 target hit rate를 개선하는가.
2. 단계별 emitter가 같은 평가 비용에서 realization drift와 실패율을 줄이는가.

## 9. 최소 구현 순서

1. 기존 15개 최종 mesh에서 descriptor 후보와 도달 가능한 공간을 오프라인 분석한다.
2. 체적 제약을 활성화하고 목표 체적 오차를 검증한다.
3. `target_cell`, `realized_cell`, 단계별 descriptor와 realization gap을 결과 schema에 추가한다.
4. 초기에는 전이 행렬과 간단한 회귀·분류 모델로 후보를 선택한다.
5. 같은 예산으로 random, naive MAP-Elites, RA-QD를 비교한다.
6. bracket에서 target hit와 coverage가 개선된 뒤 교량의 `구조 어휘 × 개방성` 실험으로 확장한다.

## 10. 논문에서 주장할 수 있는 범위

주장 후보는 “새로운 MAP-Elites 변형” 자체보다 다음 결합에 있다.

- multimodal concept에서 물리 검증 구조로 이어지는 확률적 연쇄를 QD 문제로 정식화.
- 목표 의도 archive와 실제 최종 archive를 구분하고 intent-to-realization 전이를 학습.
- 생성 단계별 emitter를 이용해 의미 손실과 물리적 실패의 원인을 구분.
- 최종 3D mesh와 FEA에 기반한 verified repertoire 제공.

표준 QD가 각 행동 niche에서 고품질 해를 찾는다는 정의와 topology optimization에 MAP-Elites를 적용한 선행연구는 이미 존재한다. 또한 설계 의도와 구조 성능을 함께 다루는 topology optimization도 등장했다. 따라서 신규성은 QD, 의미 보존 또는 3D TO 각각의 최초 적용이 아니라, 이 프레임워크의 단계적 realization gap을 측정·학습하고 최종 구조 archive를 능동적으로 채우는 방법과 실증에서 입증해야 한다.

관련 1차 자료:

- [Quality Diversity: A New Frontier for Evolutionary Computation](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00040/full)
- [Evolutionary Seeding of Diverse Structural Design Solutions via Topology Optimization](https://doi.org/10.1145/3670693)
- [Shape control methods for reflecting designer intent in topology optimization](https://www.sciencedirect.com/science/article/pii/S0965997826000852)
- [Integrating large models with topology optimization for conceptual design realization](https://structoptlab.github.io/files/2025-Integrating%20large%20models%20with%20topology%20optimization%20for%20conceptual%20design.pdf)
