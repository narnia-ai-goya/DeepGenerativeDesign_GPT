# Realization-aware QD v2: 진단과 첫 루프

작성일: 2026-09-23. 대상은 bracket 이미지 조건부 3D 생성이다. 생성 모델은 재학습하지 않았다.

## 왜 기존 QD가 어색했나

기존 archive는 **전체 prompt의 MiniLM embedding → PCA → CVT cell**을 descriptor로 썼다. `low/medium/high` 질량 문구와 공통 렌더 지시가 구조 의도와 함께 embedding된다. 실제로 같은 `branching`, `diagonal_truss`, `arch_tie`, `ring_lattice`, `radial_fan` 의도가 각각 2개 cell로 갈라졌다. 반대로 `hourglass`와 `swept`처럼 눈에 띄게 다른 입력 이미지가 같은 `branching` prompt/cell로 처리되었다. 따라서 8/9 text-cell 점유는 실현 형상 다양성의 증거가 아니다.

기존 19개 최종 mesh를 BC 마스크를 제외한 동일 물리 격자의 3-view Jaccard로 재측정했다. Prompt PCA 거리와 최종 형상 거리의 순위 상관은 `-0.057`이었다. 투영 면적을 동일하게 맞춰 단순 부재 폭 차이를 줄여도 `0.018`이었다. 같은 prompt cell 내 거리 중앙값 `0.221`은 다른 cell 간 `0.205`보다 작지 않았다. 수치와 전체 행렬: `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/framecorrected_semantic_pareto_qd/geometry_audit.json`.

## 이번에 연결한 QD 루프

1. 질량 표현을 제거한 **canonical structural intent**를 semantic niche로 고정한다. 서로 다른 intent 사이의 MiniLM 거리는 탐색 지도에 기록하되 mass 때문에 cell을 옮기지 않는다.
2. 검증된 입력 이미지와 비교하여 아직 평가하지 않은 이미지의 형태 novelty를 계산한다. 이는 gradient-free **제안 점수**이며, 최종 3D novelty나 FEA 성능으로 해석하지 않는다.
3. 선택 이미지를 고정 seed/BC/FEA ON 설정으로 dense → sparse → BC Boolean/remesh에 통과시킨다.
4. 최종 watertight mesh에 독립 FEA를 수행한다. 저장된 동일 mesh에 대해 fix/load 접촉, envelope 포함률, 두께를 재평가한다. 형상 hard gate를 통과한 후보만 elite 경쟁에 넣는다.
5. 최종 mesh의 BC 제외 3-view 실현 형상 거리를 계산한다. 단순 폭 차이의 영향을 줄이기 위해 투영별 면적을 동일하게 맞춘다. `(intent, realized-shape cluster)`별 compliance–volume 비지배 해를 보존하고 다음 이미지를 제안한다.

첫 실제 루프의 후보는 `r1_periodic_radial_thick__low`였다. 최종 compliance `0.003922 J`, 체적 `243,516 mm³`, 최대 von Mises `21.0 MPa`; 최근접 기존 최종 형상과의 거리는 `0.149`였다. 독립 FEA와 형상 hard gate를 통과했다. fix 표면 1.5 mm 이내 비율 `0.725`, load `0.617`, envelope 포함률 `0.9998`, 두께 p10 `3.0 mm`였다. 다음 후보 목록에서 동일 이미지가 제거되는 것도 확인했다. 다만 입력 이미지의 가는 fan 리브 여러 개는 최종 mesh의 왼쪽 하중 영역에서 뭉쳤다. 이미지 novelty만으로 실현 다양성을 보장할 수 없다는 한계가 보인다.

22개 최종 mesh 전체를 동일 형상 gate로 재평가한 결과 20개가 통과했다. `r2_outline_swept__low`, `r2_outline_tapered__high`는 두께 p10 `2.598 mm`로 실패했으며, archive 기록에는 남기되 shape cluster와 Pareto elite 경쟁에서 제외했다. gate 원자료는 `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/realization_aware_qd_archive_v2/geometry_gate_results.json`이다. 22개 중 20개 통과라는 수치는 연구 방법의 개선 효과가 아니라, 과거 설정이 섞인 파일들을 동일 판정식에 통과시킨 결과다.

결과:

- 새 archive: `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/realization_aware_qd_archive_v2/index.html`
- 첫 루프 입력–top–iso 비교: `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/realization_aware_qd_archive_v2/loop_01/r1_periodic_radial_thick__low/index.html`
- 다음 이미지 후보: `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/realization_aware_qd_archive_v2/next_image_batch/index.html`
- 이미지–최종 mesh 거리 진단: `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/realization_aware_qd_archive_v2/image_mesh_realization_audit.json`

## 아직 주장하면 안 되는 것

`shape cluster`를 3-view 거리 `0.15`로 자른 결과는 탐색용이다. gate를 통과한 20개 mesh에서 임계값 `0.10/0.12/0.15/0.18`에 따라 cluster 수가 `18/17/9/5`로 달라진다. 현재 13개 intent × shape cell의 19개 Pareto elite는 cell당 후보가 적어서 대부분 비지배로 남은 결과다. 디자이너가 판단한 “실제로 다른 형상” 쌍으로 거리를 보정하기 전에는 cluster coverage를 주요 논문 지표로 삼을 수 없다. 기존 후보는 생성·FEA guidance 설정도 섞여 있어 서로 다른 방법의 성능 비교 자료가 아니다. 전체 22개 이미지–mesh 쌍에서 거리 순위 상관 `0.216`은 진단값일 뿐 고정 프로토콜의 전이 성능 추정치가 아니다.

다음 실험에서는 (a) **동일한 FEA ON·BC·seed·후처리** 설정으로 여러 이미지 후보를 같은 예산에 평가하고, (b) `random` 대 image-novelty proposal을 비교하며, (c) 디자이너가 30–50개 형상 쌍의 구별 가능성을 표시해 거리/cluster 임계값을 고정하고, (d) 매 라운드 최종 mesh의 phenotype novelty·Pareto HV·BC/FEA 통과율을 누적해 보고해야 한다. 이 실험 전에는 BO-QD나 DQD라고 부르지 않고 **training-free realization-aware QD pilot**로 표기한다.
