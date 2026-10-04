# Bracket 네 fix 홀의 찢어진 듯한 표면: 원인과 보정 실험

**정정:** 사용자가 가리킨 결함은 고정 홀 안쪽이 아니라 네 fix 보스의 **바깥쪽 벽**에 난 작은 패임이다. 아래는 그 전에 수행한 홀·링 주변 실험의 기록이며, 해당 결함의 해결책으로 해석하면 안 된다. 외측 벽의 단계별 원인과 검증된 보정 후보는 `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/shape_language_angular_2026-09-23/outer_wall_sparse_bc_dilate_2026-09-23/envelope_wall_strip/index.html`에 있다.

대상: `r5_longitudinal_spine__angular__fea_on`. 원본 최종 OBJ는 `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/shape_language_angular_2026-09-23/post/final.obj`이며, 어떤 실험도 이를 덮어쓰지 않았다.

## 단계별 확인

- 512×512 입력 이미지에는 네 개의 fix 홀과 바깥 링이 명확하다.
- `angular_dense_aligned.obj`의 top 렌더에서는 링이 이미 계단형이며 홀 일부가 좁다. Dense의 약 64³ 형상 격자로는 지름 약 10.3 mm인 fix 홀을 3–4칸 정도로만 표현한다. 찢어진 듯한 형상의 핵심은 최종 smoothing 단계보다 앞에서 발생한다.
- 원본 최종 OBJ는 watertight 단일 성분이다. fix 표면 1.5 mm 이내 비율 0.725, load 0.617, envelope 포함률 0.99976, 두께 p10 4.24 mm로 형상 기준을 통과하지만, 시각적으로 BC 외곽이 거칠다. 따라서 기존 gate만으로는 이 결함을 검출하지 못한다.

## 실제 실험

| 방법 | 관찰 | 판정 |
|---|---|---|
| fix 6 mm dilation collar를 envelope clip에서도 보존 | 네 홀을 막았다. 홀을 다시 잘라도 fix 접촉률 0.403, envelope 포함률 0.946 | 불합격 |
| post dilation 6→2 mm | fix 접촉률 0.725, 포함률 0.9998. 외관은 거의 같음 | 개선 부족 |
| 생성 body의 fix 주변을 제거하고 BC collar로 재결합 | 좁은 cut은 collar가 다시 채워 변화가 없고, 넓은 cut은 remesh 후 큰 body가 사라짐 | 불합격 |
| fix STL 기반 링 삽입 | 홀 및 형상 기준 통과, 외곽 찢어짐 잔존 | 개선 부족 |
| 원본 최종 mesh에서 fix 표면을 고정한 국소 smoothing | 링 주변 외관 약간 개선, fix 0.725, envelope 0.99978, compliance 0.004664→0.004641 J | 부분 개선 |

시각·수치 비교: `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/shape_language_angular_2026-09-23/fix_collar_repair_2026-09-23/index.html`.

## 필요한 본 수정

BC의 네 홀을 **생성 중 높은 해상도의 signed-distance 제약**으로 다뤄야 한다. Dense는 구조 경로를 정하는 용도로 두되, sparse에서는 원본 fix STL의 홀 내부를 void, 링을 solid로 강제하고, 바깥 2–4 mm 전이대만 생성 형상과 매끄럽게 섞는 것이 타당하다. 최종 결과에는 각 홀의 관통성·최소 개구 반경·외곽 곡률/요철, BC 접촉, envelope, 독립 FEA를 별도로 측정해야 한다. 현재의 1.5 mm BC 근접률과 watertight 판정만으로는 이 시각 결함을 걸러낼 수 없다.

이 제약을 구현·검증하기 전에는 국소 smoothing OBJ를 연구용 수정 후보로만 사용하고, QD archive의 원본 mesh를 교체하지 않는다.
