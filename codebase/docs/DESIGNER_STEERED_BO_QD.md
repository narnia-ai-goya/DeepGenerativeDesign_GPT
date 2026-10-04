# Designer-Steered Semantic BO-QD

이 계층은 foundation model을 학습하지 않고 black-box 생성 후보의 순서만 조정한다. 디자이너의 입력은 soft preference이며 BC, envelope, 연결성, 최소 두께와 독립 FEA는 기존 평가 파이프라인의 hard gate로 남는다.

## 실행 단위

1. 이미지 후보를 만들고 최종 3D phenotype에서 semantic niche와 normalized mass를 측정한다.
2. surrogate가 cell 실현 확률, 성능 개선, validity와 intent-realization 확률을 예측한다.
3. `designer_steering.py`가 target cell, semantic anchor, mass range, lock/reject와 local exploration을 acquisition에 반영한다.
4. 선택 후보를 dense → sparse → post → independent FEA로 평가한다.
5. hard gate를 통과한 후보만 semantic × mass archive에 들어간다.
6. 매 batch 뒤 디자이너가 archive를 검토하고 다음 revision을 저장한다.

## 로컬 인터페이스

초기 template:

`/home/goya/SDL/3d_qd/codebase/configs/designer_steering_bracket.json`

실험별 interaction state는 template을 복사해 `experiments/` 아래에 두며, 현재 실행 파일은 다음과 같다.

`/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/runs/periodic_steering/seed_42/designer_steering_state.json`

웹 UI:

```bash
python /home/goya/SDL/3d_qd/codebase/designer_steering_dashboard.py \
  --state /home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/runs/periodic_steering/seed_42/designer_steering_state.json \
  --port 8770
```

브라우저 주소: `http://127.0.0.1:8770/`

CLI에서도 동일한 action을 기록할 수 있다.

```bash
python /home/goya/SDL/3d_qd/codebase/designer_steering.py STATE target arch medium 2.0
python /home/goya/SDL/3d_qd/codebase/designer_steering.py STATE lock candidate_012
python /home/goya/SDL/3d_qd/codebase/designer_steering.py STATE reject candidate_027
python /home/goya/SDL/3d_qd/codebase/designer_steering.py STATE explore candidate_012 --radius 0.12
```

각 action은 `revision`, UTC 시각, actor와 원래 payload를 `interaction_log`에 남긴다. 논문 실험에서는 이 로그가 intervention budget과 의사결정 과정을 재현하는 근거가 된다.

## 현재 코드와의 연결

기존 image BO-QD selector에 steering 파일을 전달할 수 있다.

```bash
python /home/goya/SDL/3d_qd/codebase/select_image_bo_qd_candidates.py \
  --steering /home/goya/SDL/3d_qd/experiments/bracket/designer_steered_bo_qd_2026-09-22/designer_steering_state.json \
  --output /home/goya/SDL/3d_qd/experiments/bracket/designer_steered_bo_qd_2026-09-22
```

이 selector의 기존 archive는 `void fraction × branch count`를 사용하는 호환성 확인용 adapter다. 논문 본 실험에서는 candidate record가 다음 필드를 제공해야 한다.

- `semantic_niche`: 최종 multi-view render에서 측정한 realized semantic niche
- `mass_bin`: normalized mass bin
- `normalized_mass`: envelope 또는 solid reference로 정규화한 질량
- `semantic_scores`: canonical text anchor별 최종 3D similarity
- `base_acquisition`: designer preference 적용 전 realization-aware BO-QD acquisition
- `parent_id`: local exploration 요청이 있을 때 기준 elite

`steered_acquisition`은 soft preference가 반영된 값이다. `valid`, BC coverage, thickness와 FEA 결과는 이 모듈이 수정하지 않는다.

## 비교 실험

동일한 이미지 생성, sparse generation과 FEA 예산에서 다음 네 조건을 비교한다.

1. Random
2. 완전 자동 BO-QD
3. 초기 intent만 고정한 BO-QD
4. 매 batch archive를 확인하는 designer-steered BO-QD

각 조건은 preferred-niche coverage, 전체 coverage, QD-HV, FEA 통과율, 선택된 Pareto set의 pairwise designer utility, 개입 횟수와 소요 시간을 보고한다.

## 바깥 형상 자유도

Bracket의 design envelope는 목표 외곽선이 아니라 최대 허용 영역으로 사용한다. 네 개 mounting bore의 중심과 최소 pad, 두 개 load lug의 위치와 접촉 영역만 고정한다. 이 interface에서 떨어진 perimeter, taper, waist, shoulder, concavity와 asymmetry는 생성 대상이다.

외곽 형상의 다양성은 final mesh의 plate-normal view(`v_top`)를 채운 silhouette mask로 만든 뒤 pairwise Jaccard distance로 측정한다. 내부 hole은 채워서 internal topology diversity와 분리한다. BC coverage, inside-envelope fraction, 단일 component, 최소 두께와 독립 FEA는 그대로 hard gate로 적용한다.

2026-09-22 pilot에서 hard-gate-valid 형상의 평균 outer-silhouette distance는 fixed-outline round 1의 `0.0258`에서 outline-free probe의 `0.1492`로 증가했다. 단, outline-free 6개 중 2개만 모든 gate를 통과했다. hourglass와 spoke-island는 load BC coverage, swept와 fork는 최소 두께에서 탈락했다. 따라서 다음 탐색은 전체 외곽선을 다시 채우지 않고 BC sleeve와 얇은 member만 국부 보강한다.

통합 결과:

`/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/report_full/index.html`

프로토콜 수정 기록:

`/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/protocol_amendment_round2_outline_free.json`
