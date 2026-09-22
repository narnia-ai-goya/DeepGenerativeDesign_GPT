# RA-QD를 위한 posterior sampling 선택

## 이번 실험에서 채택한 방법

이번 online 비교는 **제약형 batch Thompson sampling + greedy maximin batch diversity**를 사용한다.

각 후보 입력은 이미지 스타일 범주와 `cfg`, `sp_cfg`, `sp_guide_w_peak`의 연속 변수 세 개다. 출력별로 최종 mesh의 두 descriptor, 체적비, log compliance를 모델링한다. 범주형 스타일에는 같은 스타일과 다른 스타일의 상관을 분리한 kernel을, 연속 변수에는 RBF kernel을 쓴다. Random Fourier Features로 사전 함수를 만들고 Matheron rule로 관측값에 조건화하여 후보 풀 전체에 대해 하나의 일관된 함수 표본을 뽑는다.

표본 함수에서 다음 조건을 만족하는 후보를 우선한다.

1. 예측된 최종 체적비가 `0.500 ± 0.025`다.
2. 예측 descriptor가 고정된 4×4 archive 범위 안에 있다.
3. 빈 셀을 채우거나 기존 셀의 compliance를 개선한다.
4. 같은 batch의 후보와 입력 공간에서 충분히 떨어져 있다.

초기 12회는 네 Sobol recipe를 각각 세 seed로 반복한다. seed 숫자는 매끄러운 설계 변수가 아니므로 surrogate 입력에서 제외하고, 같은 recipe의 반복 편차로 관측 잡음을 추정한다. 이후 20회 중 두 회는 posterior가 잘못 확신하는지를 확인하기 위한 epsilon random audit이다.

이 선택은 현재 문제에 잘 맞는다. 설계 변수는 혼합형이지만 차원이 작고, full pipeline 평가는 비싸며, GPU 세 개로 batch 평가한다. 단일 평균 예측을 따라가는 방식보다 posterior의 불확실성을 자연스럽게 탐색에 반영하고, maximin은 동시에 제안된 후보의 중복을 줄인다.

## 더 발전시킬 posterior sampling

### 1. Constrained GP Thompson sampling + DPP

표본 수가 쌓이면 현재 RFF 근사를 exact GP 또는 sparse variational GP로 교체하고, maximin 대신 DPP로 batch를 선택하는 것이 첫 번째 개선안이다. DPP는 posterior에서 유망하면서 서로 비슷하지 않은 후보 묶음을 확률적으로 선택한다. 이번 v1의 maximin은 구현과 감사가 단순한 대체 수단이다.

- 혼합 변수와 제약을 포함한 Bayesian QD: <https://arxiv.org/abs/2310.05955>
- DPP를 이용한 diversified batch Thompson sampling: <https://proceedings.mlr.press/v151/nava22a.html>

### 2. qPOTS

품질을 compliance 하나로 두지 않고 semantic fidelity, 제조성, 질량, 응력 등을 서로 다른 목표로 유지한다면 qPOTS가 적합하다. posterior 표본에서 Pareto-optimal한 후보를 얻고 maximin으로 batch를 구성한다. 현재 실험에서는 체적을 명시적 제약으로 두고 compliance를 단일 품질로 두므로 바로 도입할 필요는 없다.

- qPOTS: <https://proceedings.mlr.press/v258/renganathan25a.html>

### 3. Multi-fidelity posterior sampling

`mesh_dense.obj → mesh.obj → final.obj`의 상관이 충분히 확인되면 dense 또는 sparse 단계에서 많은 후보를 싸게 거르고 일부만 FEA까지 진행할 수 있다. 단, fidelity 간 순위 상관과 feasibility 예측의 calibration을 먼저 별도 검증해야 한다. 이번 실행은 각 평가가 모두 final mesh와 FEA까지 가므로 방법 간 계산 예산이 명확하다.

### 4. Asynchronous Thompson sampling

케이스별 실행 시간이 크게 달라 GPU가 자주 놀게 되면 동기 batch 대신 GPU가 비는 즉시 posterior를 갱신해 다음 후보를 내는 비동기 방식이 유리하다. 논문용 공정 비교에서는 wall-clock 순서가 결과에 개입할 수 있으므로, 먼저 현재의 고정 checkpoint 동기 실험을 기준선으로 확보한다.

- 병렬 Thompson sampling: <https://proceedings.mlr.press/v84/kandasamy18a.html>
- scalable constrained max-value/posterior sampling 계열: <https://proceedings.mlr.press/v130/eriksson21a.html>

## 이번 비교의 판정 기준

각 방법의 논리 예산 12, 16, 24, 32에서 다음 값을 비교한다.

- 최종 mesh 체적 제약을 통과한 verified archive coverage
- verified QD score와 셀별 best compliance
- 체적 제약 통과율
- RA-QD의 목표 셀 적중률과 realization gap
- dense, sparse, final 단계 사이 descriptor 및 체적 drift

random이 더 좋다면 posterior의 평균 정확도만 보지 않고, feasibility calibration, 목표 셀 transition entropy, batch 중복, descriptor 범위 이탈을 나누어 원인을 진단한다.

실행 프로토콜: `/home/goya/SDL/3d_qd/experiments/bracket/raqd_online_2026-09-13/protocol.json`

실시간 보고서: `/home/goya/SDL/3d_qd/experiments/bracket/raqd_online_2026-09-13/report.html`
