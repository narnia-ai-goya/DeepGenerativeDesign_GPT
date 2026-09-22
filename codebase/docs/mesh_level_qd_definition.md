# Mesh-level Quality-Diversity definition

작성일: 2026-09-21  
상태: **v1 명세 제안 — 생성 guidance에는 아직 미구현**

## 목적

이 QD는 CAD feature tree 또는 prompt/seed의 다양성이 아니다. 동일한 envelope,
load/support BC, 재료 및 최종 검증 조건 아래에서 생성기가 만드는 **최종 mesh의
형상 repertoire**를 채우는 문제다. FEA는 형상 다양성의 축이 아니라, 각 형상이
충분히 좋은 구조인지 평가하는 quality/constraint다.

최종 mesh `M`에 대해 다음을 분리한다.

| 역할 | 정의 |
|---|---|
| Hard validity | watertight, 단일 connected component, envelope containment >= 0.99, 유한한 최종 FEA compliance |
| Behavior `b(M)` | multi-view shape embedding `(z_1, z_2)` — 어느 형상 family인가 |
| Quality `q(M)` | `(min compliance, min material volume)`의 Pareto quality — 같은 behavior niche에서 어느 해가 좋은가 |

재료량은 이미 quality의 목적이므로 behavior 축으로 다시 쓰지 않는다. 최소 부재
두께는 manufacturing/validity constraint로 취급한다. 이 구분이 QD와 단순 MOO를
분리한다.

## 최종 mesh의 고정 descriptor: multi-view shape manifold

모든 최종 mesh를 같은 정규화 좌표계와 orthographic camera로 top/front/side view에
render한다. 각 view는 배경색이나 금속 재질이 아니라 geometry-only soft silhouette
(필요하면 depth 포함)로 표현한다. 세 view를 concatenate한 vector `phi(M)`를 만든다.

reference set에서 `phi(M)`를 표준화한 뒤 PCA를 **한 번만** 학습하고,

`b(M) = (z_1, z_2) = PCA_2(phi(M))`

로 archive 좌표를 정의한다. `z_1`, `z_2`는 "더 좋음"의 순서가 아니라 서로 다른
형상 family의 좌표다. 이 방식은 구멍 수, 개구부 위치, branch/strut 분포, 외곽선의
질적 차이를 함께 반영한다. 축이 해석하기 어려워지는 문제는 각 grid cell에서
representative mesh와 세 view를 함께 보여 해결한다.

PCA가 너무 전역적인 변형만 잡으면 `phi`에 low-resolution silhouette뿐 아니라
multi-scale distance transform 또는 depth map을 넣어 내부 개구부와 두께 분포를
보존한다. BC와 envelope 영역은 mask로 제외하거나 고정값으로 처리해, 변하지 않는
BC pixel이 descriptor를 지배하지 않게 한다.

## Archive와 quality

검증을 통과한 최종 mesh만 `b(M)`의 2-D grid cell에 넣는다. 각 cell에는
`(compliance, volume)` Pareto front를 유지하며, 보고 지표는 셀별 Pareto front의
정규화 hypervolume 합(QD-HV)과 coverage다. 따라서 한 개의 점이 HV가 아니라,
각 niche의 해 집합이 만든 quality다.

기존 bracket expanded 비교의 8x8 threshold는
`experiments/bracket/rab_moqd_expanded_2026-09-14/protocol.json`에 동결돼 있다.
새로운 direct mesh-QD 실험은 이 threshold를 자동으로 재사용하지 않는다. shape
reference set으로 PCA와 2-D grid threshold를 pilot 전에 동결하고, 실행 뒤에는
재학습이나 재binning하지 않는다.

## 생성 중 적용할 정의

최종 descriptor `b(M)`는 평가의 진실값이며 바꾸지 않는다. dense/sparse 단계에는
그와 별개로 미분 가능한 proxy `b_tilde(rho)`만 쓴다.

`L = L_BC + L_envelope + L_regularization + L_FEA + lambda_shape ||z_tilde(rho) - z*||^2`

* `z_tilde(rho)`: dense/sparse soft occupancy에서 같은 orthographic soft silhouette와
  depth proxy를 미분 가능하게 만들고, 동결된 PCA transform으로 투영한 좌표다.
* 이는 final mesh의 `z(M)`와 같은 geometry feature를 측정한다. 단순 material volume,
  compliance, FEA energy를 shape descriptor 대용으로 쓰지 않는다.

`BC`와 `envelope`은 behavior와 trade-off하지 않는 강한 제약이다. 목표 cell center
`z*`를 먼저 고른 뒤 dense와 sparse guidance에 같은 geometry proxy를 넣고, 마지막에는
반드시 독립적인 final mesh render로 실제 cell을 판정한다. FEA는 그 후 quality를
판정한다.

## 구현 전 성립 검증

2-D archive를 열기 전에 PCA 각 축을 한 축씩 조준한다. 고정 BC, envelope,
image/seed 조건에서 각 목표값을 5개 이상 두고 final descriptor의 target error,
render 상의 식별 가능한 형상 변화, validity, compliance/volume 변화를 측정한다.
`z_tilde`와 final `z`의 상관 및 target hit가 확인된 경우에만 archive를 연다. 이
검증을 통과하지 못하면 post-hoc archive 결과를 direct QD라고 주장하지 않는다.
