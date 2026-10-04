# 전체 코드 및 Shape-QD 개선점 점검

작성일: 2026-09-22
점검 대상: `/home/goya/SDL/3d_qd`
GitHub 복제본: `/home/goya/SDL/DeepGenerativeDesign_GPT`
점검 당시 Git commit: `76d43dc`

## 1. 결론

기존 QD 실험 코드와 archive 관련 단위 테스트는 정상적으로 동작한다. 반면 새로 추가한
Shape-QD/DQD 경로는 아직 개념 검증 단계다. 현재 가장 큰 병목은 dense 단계에서 만든
형상 변화가 sparse refinement를 지나며 거의 모두 사라지는 현상이다.

현재 구현은 엄밀한 의미의 DQD emitter라기보다 target niche 또는 prototype을 향한
**differentiable shape-targeted guidance**에 해당한다. 논문에서 DQD를 주장하려면 최종 mesh의
target niche 적중, archive feedback, objective/measure gradient 기반 proposal, 독립 FEA 검증이
모두 필요하다.

## 2. 확인된 정상 항목

- 전체 Python 구문 검사: 통과
- 기존 QD 관련 테스트: `21 passed, 1 warning`
- `/home/goya/SDL/3D_GEN`을 직접 참조하는 Python, shell, JSON, YAML 파일: 없음
- GitHub 복제본 작업 트리: clean
- 원본 `codebase`와 GitHub 복제본의 내용 차이: 없음
  - `docs/` 디렉터리 timestamp 차이만 존재
- 저장소 내 실제 API key로 의심되는 문자열: 발견되지 않음

## 3. 최우선 수정 항목

### 3.0 Dense residual guidance 추가 실험 (2026-09-22)

Dense가 실제로 target morphology를 바꿀 수 있는지 확인하기 위해 prototype 전체 BCE 대신,
warm-up 시점의 dense posterior와 target prototype이 달라지는 8³ macro cell만 solid/empty로
유도하는 `ContrastiveMacroMorphology`를 추가했다.

구현 파일:

- `codebase/code/shape_qd_loss.py` — `ContrastiveMacroMorphology`
- `codebase/code/generate_with_physics_guidance.py` — `--shape-residual-*` CLI와 dense 적용
- `codebase/tests/test_shape_qd_loss.py` — residual-cell gradient 단위 테스트

모든 실험은 기존 `lr162`의 single-view image, BC frame, 50 dense step을 고정하고 sparse를
완전히 끈 상태에서 수행했다.

| 방법 | 설정 | V / F | component | baseline dense와 평균 TSDF 거리 | 판정 |
|---|---|---:|---:|---:|---|
| baseline 재생성 | FlowDPS | 7,376 / 14,816 | 1 | - | 기준 |
| residual + FlowDPS | `w=60`, pool 8, delta .12 | 7,416 / 14,900 | 1 | `4.20e-7` | 변화 없음 |
| residual + persistent Adam | `w=60`, pool 8, delta .12, lr .02, inner 5 | 5,223 / 10,466 | 3 | `4.74e-4` | 변화는 있으나 붕괴 |

참고로 같은 seed의 기존 baseline과 재생성 baseline 사이 평균 TSDF 거리는 `1.75e-5`였다.
따라서 FlowDPS residual 결과의 `4.20e-7` 변화는 sampler의 재실행 변동보다도 작다. 반면 Adam은
분명히 dense 형상을 바꾸지만 volume이 약 31% 줄고 component가 3개가 되어 유효한 structural
proposal이 아니다.

결론: prototype field 전체 또는 residual cell의 BCE를 latent에 직접 거는 방식은 dense 구조
제어의 메인 방법으로 적합하지 않다. 현재 bank의 condition mismatch와 latent-space gradient의
locality 때문에, 약하면 prior에 흡수되고 강하면 image/connection prior를 벗어나 붕괴한다.

결과 절대 경로:

- baseline 재생성: `/home/goya/SDL/3d_qd/experiments/bracket/shape_dqd_dense_residual_2026-09-22/baseline_rebuilt`
- FlowDPS residual: `/home/goya/SDL/3d_qd/experiments/bracket/shape_dqd_dense_residual_2026-09-22/niche_01_w60_pool8`
- Adam residual: `/home/goya/SDL/3d_qd/experiments/bracket/shape_dqd_dense_residual_2026-09-22/niche_01_adam_w60_lr002`

다음 dense 방법은 raw prototype matching이 아니라 **local descriptor transport**로 바꿔야 한다.
warm-up posterior의 descriptor `m(z_ref)` 주변에서 작은 target displacement `delta m`을 정하고,
Jacobian `J = dm/dz`로 최소 latent 이동 `delta z = J^T (J J^T + lambda I)^-1 delta m`을 구한다.
BC, envelope, connection, volume은 trust-region likelihood로 유지한다. 이 방식은 prototype 하나를
복제하지 않고 image-conditioned dense posterior 안에서 조절 가능한 phenotype 방향만 탐색한다.

### 3.0.1 Local PCA transport pilot (2026-09-22)

위 재설계를 최소 구현으로 시험했다. warm-up step 15에서 frozen PCA의 첫 두 phenotype coordinate에
대해 niche 04 방향으로 `0.75` 이동을 요청하고, `J^T (J J^T + lambda I)^-1`의 ridge pseudo-inverse로
one-shot VAE latent transport를 수행한 뒤 기존 FlowDPS denoising을 계속했다. Sparse는 끈
dense-only 검증이다.

| 항목 | baseline 재생성 | local transport |
|---|---:|---:|
| vertices / faces | 7,376 / 14,816 | 7,637 / 15,334 |
| component | 1 | 1 |
| mesh volume | `4.6784e-4` | `4.6868e-4` |
| baseline과 평균 TSDF 거리 | - | `1.0331e-4` |

이 값은 같은 seed baseline 재생성 변동(`1.75e-5`)의 약 6배다. 즉 raw prototype residual과 달리
dense geometry에 관측 가능한 변화를 만들었고, Adam residual처럼 연결성이 깨지거나 volume이
급락하지 않았다.

첫 실행의 requested descriptor displacement는 `0.75`였지만 ridge regularization으로 first-order
predicted displacement는 `0.103`에 머물렀다. 다음 실험에서는 ridge와 trust-region 크기를
grid search해 target displacement 대비 실제 final displacement의 calibration curve를 만들어야 한다.

구현과 결과:

- `codebase/code/shape_qd_loss.py` — `LocalPcaTransport`
- `codebase/code/generate_with_physics_guidance.py` — `--shape-transport-*`
- `/home/goya/SDL/3d_qd/experiments/bracket/shape_dqd_local_transport_2026-09-22/niche_04_r075`

### 3.0.2 Repeated local transport timing test (2026-09-22)

One-shot transport이 후속 flow update에서 사라지는지 확인하기 위해 transport를 재선형화해
세 번 적용할 수 있도록 확장했다.

- early repeat: step 15, 20, 25, radius `.50`, ridge `1e-6`, relative trust `.015`
- late repeat: step 32, 37, 42, 동일 parameter

각 intervention 직후 Jacobian first-order prediction은 각각 최대 `.173` (early) 및 `.184`
(late)까지 증가했다. 그러나 final mesh의 baseline 대비 평균 TSDF 차이는 다음과 같았다.

| schedule | component | volume | final mean TSDF distance |
|---|---:|---:|---:|
| early repeat | 1 | `4.5375e-4` | `1.50e-5` |
| late repeat | 1 | `4.6470e-4` | `1.18e-8` |

두 경우 모두 최종 형상 차이는 재실행 변동 이하로 수렴했다. 이는 transport timing 또는 반복 횟수
문제가 아니라, 현재 one-step latent correction이 이어지는 Direct3D-S2 flow update에 의해
복원된다는 증거다. 따라서 다음 제안법은 decoder latent에 외부 correction을 더하는 방식이 아니라
score/velocity prediction 자체에 conditional structural residual을 학습하거나, 해당 residual을
각 denoising step의 update 식에 명시적으로 결합해야 한다.

결과 절대 경로:

- early: `/home/goya/SDL/3d_qd/experiments/bracket/shape_dqd_local_transport_2026-09-22/niche_04_repeat3_r050_ridge1e6_rel015`
- late: `/home/goya/SDL/3d_qd/experiments/bracket/shape_dqd_local_transport_2026-09-22/niche_04_repeat3_late065_r050_ridge1e6_rel015`

### 3.1 Dense 형상 의도가 sparse 단계에서 사라짐

최근 macro scaffold 실험에서 측정한 값은 다음과 같다.

| 비교 | 평균 절대 TSDF 차이 |
|---|---:|
| baseline ↔ scaffold dense | `1.1123e-3` |
| baseline ↔ scaffold final | `1.2813e-5` |
| scaffold dense ↔ scaffold final | `1.1115e-3` |

Dense 결과는 baseline과 충분히 달라졌지만 sparse refinement를 통과한 최종 mesh는 baseline과
거의 같아졌다. 따라서 현재 병목은 dense loss의 세기보다 dense-to-sparse 형상 보존이다.

관련 코드:

- `codebase/code/shape_qd_loss.py:70` — `MacroShapeScaffold`
- `codebase/code/generate_with_physics_guidance.py:1605` — dense scaffold 적용
- `codebase/code/generate_with_physics_guidance.py:3730` — sparse Shape-QD 집계

필요한 수정:

1. `MacroShapeScaffold`를 sparse occupancy에도 적용한다.
2. dense 결과와 sparse 중간 결과 사이에 consistency loss를 둔다.
3. dense에서 확정한 핵심 support를 sparse가 삭제하지 못하도록 별도 core mask로 전달한다.
4. 목표 support를 현재 형상과 가까운 voxel 순서로만 추가하지 말고 skeleton 또는 branch
   단위로 전달한다.
5. sparse iteration 중간 결과를 64³ macro field에 재투영해 target drift를 기록한다.

### 3.2 공통 geometry measurement operator 부재

현재 단계별 표현이 서로 다르다.

| 단계 | 현재 표현 |
|---|---|
| archive | hard voxelization 및 silhouette/depth PCA |
| dense | 64³ sigmoid occupancy |
| sparse | 512³ active sample의 64³ 평균 |
| final 평가 | 별도의 96³ mesh TSDF |

문서에서 정의한 공통 `Phi`가 실제 생성 코드에는 연결되지 않았다. `dense logits`, `sparse SDF`,
`final mesh`를 동일한 world bounds, axis order, resolution, BC mask로 측정해야 한다.

권장 `Phi` 구성:

- 96³ 또는 128³ truncated signed-distance field
- 48³, 24³, 12³ multi-scale field
- top/front/right accumulated density와 depth
- descriptor에서 고정 BC 영역 제외
- envelope 외부 명시적 masking

관련 코드:

- `codebase/tsdf_shape_operator.py`
- `codebase/code/shape_qd_loss.py`
- `codebase/run_shape_qd_representation_pilot.py`

### 3.3 TSDF 부호 불일치

실제 확인 결과 `pysdf`와 `trimesh.proximity.signed_distance`는 모두 내부를 양수로 반환했다.

```text
cube center:  +1.0
cube outside: -1.0
```

그러나 `codebase/tsdf_shape_operator.py:22`에서는 `pysdf` 결과만 음수화한다.

```python
query = lambda q: -field(...)
```

함수 설명은 positive-inside라고 되어 있으므로 음수화를 제거하고, 두 backend의 부호 일치를
단위 테스트로 고정해야 한다. 두 mesh 사이 절댓값 거리만 계산할 때는 부호 반전의 영향이
작지만 occupancy 변환이나 inside/outside mask와 결합하면 오류가 발생한다.

### 3.4 Prototype bank의 조건 혼합

`codebase/build_valid_shape_prototypes.py:24`는
`experiments/bracket/**/result.json`을 모두 수집한다. 현재 100개 mesh가 10개의 서로 다른
실험 계열에서 왔다.

주요 구성:

- `raqd_online`: 51개
- `rab_moqd_expanded`: 18개
- `rab_moqd_experiment_suite`: 12개
- 기타 실험: 19개

서로 다른 reference image, 생성 parameter, postprocess, FEA 조건이 섞일 수 있다. 이는 동일한
image, BC, envelope에서 가능한 형상 다양성을 정의한다는 연구 설정과 맞지 않는다.

Bank manifest에 다음 항목을 강제해야 한다.

- reference image hash
- envelope, fixed, load mesh hash
- generator 및 checkpoint revision
- dense/sparse recipe
- random seed protocol
- `mc_threshold`
- postprocess recipe
- coordinate transform 및 scale
- FEA load case

또한 기존 `result.json`의 `valid` 값을 그대로 신뢰하지 말고 containment, component count,
BC coverage, watertightness, FEA validity를 bank 생성 시 다시 측정해야 한다.

### 3.5 Sparse 단계의 debris와 분리 성분

실측 결과:

| mesh | Watertight | 연결 성분 수 |
|---|---:|---:|
| 기존 baseline final | Yes | 28 |
| scaffold final | Yes | 24 |
| scaffold dense | Yes | 1 |

Dense는 단일 성분이지만 sparse에서 작은 분리 성분이 생긴다. 이는 load 주변 debris 및
자글자글한 형상의 직접적인 후보 원인이다.

필요한 수정:

- BC-connected component를 우선하는 differentiable connectivity loss
- 일정 간격의 작은 active island 제거
- fixed/load BC에 연결되지 않은 support token pruning
- minimum component-size penalty
- final boolean 전에 `fixed ∪ load`와 연결된 모든 성분만 유지
- component 제거 전후 compliance 변화 기록

가장 큰 성분 하나만 보존하면 load peg가 떨어질 수 있으므로 BC 연결성을 기준으로 해야 한다.

## 4. DQD 방법론 수정

### 4.1 현재 구현의 정확한 위치

현재 구현은 target centroid 또는 prototype으로 생성 trajectory를 당기는 방식이다.

- frozen PCA target loss
- multi-scale occupancy prototype anchor
- dense macro scaffold BCE
- sparse prototype support expansion

아직 없는 요소:

- descriptor Jacobian을 이용하는 emitter
- objective gradient와 measure gradient의 방향 결합
- archive 상태에 따른 빈 niche/저품질 niche 선택
- parent elite에서 여러 directional proposal 생성
- final mesh 평가 후 archive를 갱신하는 반복 loop

따라서 현재 단계의 명칭은 다음이 안전하다.

> Differentiable shape-targeted guidance for QD generation

### 4.2 DQD로 확장하는 방법

1. 최종 mesh만 공통 `Phi`와 frozen encoder `E`로 측정한다.
2. archive에서 빈 niche 또는 낮은 QD-HV niche를 선택한다.
3. parent elite를 선택한다.
4. quality gradient와 descriptor gradient를 구한다.
5. 두 gradient의 선형결합 또는 orthogonalized direction으로 여러 proposal을 만든다.
6. dense와 sparse 생성 후 final postprocess를 수행한다.
7. 독립 FEA와 feasibility를 검사한다.
8. 실제 final descriptor가 해당 niche에 들어간 경우에만 archive에 삽입한다.

MEGA/CMA-MEGA 원리를 그대로 복제할 필요는 없지만, 최소한 archive feedback과 measure-gradient
proposal이 있어야 DQD라는 명칭이 설득력을 얻는다.

## 5. Descriptor 및 prototype 수정

### 5.1 PCA는 baseline으로 유지

현재 silhouette/depth PCA는 서로 다른 mesh를 같은 embedding 또는 niche로 판단하는 문제가
확인됐다. 논문 메인 descriptor보다는 비교 baseline에 적합하다.

추천 우선순위:

1. **3D TSDF autoencoder**
   - world-frame geometry를 직접 표현
   - gradient와 interpolation이 쉬움
2. **Contrastive multi-view geometry encoder**
   - top/front/right depth 및 silhouette 사용
   - 3D encoder보다 계산량이 작음
3. **Hybrid 분석 지표**
   - learned embedding을 archive 좌표로 사용
   - topology, void count, skeleton branch count는 보조 지표로 보고

### 5.2 단일 medoid prototype 개선

현재 각 niche는 KMeans centroid에 가장 가까운 단일 mesh를 prototype으로 사용한다.

개선 후보:

- niche 내부의 TSDF barycenter
- topology signature별 subcluster prototype
- skeleton consensus와 soft thickness field
- 여러 prototype에 대한 soft-min distance

서로 다른 topology를 단순 평균하면 branch가 흐려지므로 niche 내부에서도 topology family를
먼저 분리하는 것이 좋다.

## 6. FEA 수정

### 6.1 절대 weight 대신 gradient 비율 측정

`fea_w`의 숫자만 비교하면 image, BC, geometry loss에 비해 FEA가 실제로 얼마나 작동하는지
판단하기 어렵다. 매 iteration 또는 일정 간격으로 다음을 기록해야 한다.

```text
||grad L_image||
||grad L_BC||
||grad L_shape||
||grad L_FEA||
cos(grad L_shape, grad L_FEA)
cos(grad L_image, grad L_FEA)
```

권장 자동 정규화:

```text
lambda_FEA = target_ratio * ||grad L_base|| / (||grad L_FEA|| + epsilon)
```

### 6.2 권장 FEA schedule

- dense 초반: FEA off 또는 매우 약하게 적용
- dense 중후반: stiffness guidance 활성화
- sparse: 낮은 빈도로 FEA 적용
- final mesh: 독립 tetrahedral FEA 필수

현재 Shape-QD pilot은 raw `mesh.obj`에서 끝나며 final postprocess와 독립 FEA 검증이 없다.
따라서 현재 결과만으로 기계적 quality 개선을 주장할 수 없다.

## 7. 실험 계획 수정

Parameter sweep을 늘리기 전에 다음 단계별 go/no-go를 수행해야 한다.

### 7.1 Representation parity

동일 mesh를 dense, sparse, final 형식으로 표현하고 공통 `Phi`에서 거의 같은 결과가 나오는지
확인한다.

### 7.2 One-step gradient test

한 번의 latent update 후 target geometry loss가 감소하는지 확인한다. 동시에 BC와 envelope
loss가 악화되지 않아야 한다.

### 7.3 Dense-to-sparse preservation

Dense에서 만든 형상 변화량의 최소 50% 이상이 final까지 유지되는 것을 초기 목표로 둔다.
현재 측정값은 약 1% 수준이다.

### 7.4 Targetability

- 고정 image, BC, envelope, seed
- condition-compatible prototype 5개
- prototype당 최소 5회 생성
- final target distance와 target-hit rate 측정

### 7.5 Quality validation

Feasible final mesh에만 독립 FEA를 수행하고 compliance-volume Pareto front를 구성한다.

### 7.6 동일 예산 QD 비교

- random image/seed sampling
- learned descriptor post-hoc archive
- occupancy/PCA guidance
- dense-only TSDF guidance
- dense+sparse TSDF guidance
- archive-feedback DQD emitter

모든 방법에 같은 full-generation budget과 final FEA budget을 적용해야 한다.

## 8. 코드 구조 및 테스트

### 8.1 생성 코드 분리

`codebase/code/generate_with_physics_guidance.py`는 현재 4,183줄이다. BC, FEA, dense,
sparse, Shape-QD, post interface가 한 파일에 섞여 있다.

권장 구조:

```text
generation/
  dense_guidance.py
  sparse_guidance.py
  support_transfer.py
guidance/
  boundary.py
  geometry.py
  mechanics.py
  connectivity.py
qd/
  descriptor.py
  archive.py
  emitter.py
  evaluator.py
```

전체 코드에 `except Exception` 또는 광범위한 예외 처리가 51곳 있다. 일부는 예외를 출력하지
않고 넘어간다. 연구 실험에서는 실패한 guidance나 postprocess가 조용히 무시되면 결과 해석을
왜곡하므로 실패 단계와 traceback을 manifest에 기록해야 한다.

### 8.2 테스트 환경

기존 QD 테스트 21개는 통과했다. 전체 테스트는 다음 환경 문제로 한 번에 실행되지 않았다.

- 기본 Python: `pysdf`가 없어 test collection 실패
- `direct3ds2` 환경: `pytest` 미설치

`requirements_test.txt` 또는 development environment 파일이 필요하다.

새로 필요한 테스트:

- `pysdf`와 `trimesh` TSDF 부호 일치
- dense/sparse/final world-frame parity
- shape loss gradient가 finite인지 확인
- one-step update 후 target loss 감소
- descriptor BC mask와 generation BC constraint 분리
- support expansion 및 cap의 결정론
- sparse scaffold가 실제 final output에 영향을 주는지 확인

### 8.3 결정론

최근 generation log에는 cuBLAS 결정론 경고가 반복된다. 실행 환경에 다음을 설정해야 한다.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

각 run manifest에 seed, CUDA 설정, 모델 revision, package version, 입력 hash를 저장해야 한다.

## 9. 경로 의존성과 저장소 상태

### 9.1 `3D_GEN` 의존성

현재 `codebase`의 Python, shell, JSON, YAML에는 `/home/goya/SDL/3D_GEN` 직접 참조가 없다.

그러나 `/home/goya/SDL/3d_qd`와 conda interpreter를 하드코딩한 helper script가 16개 있다.
대표 파일:

- `codebase/run_gpt_cfg_study.py`
- `codebase/make_caliper_iso_scale.py`
- `codebase/make_caliper_stretch50.py`
- `codebase/run_qd_pilot.py`
- `codebase/run_raqd_online.py`
- `codebase/run_rab_moqd_smoke.py`

모두 `Path(__file__)`, `DATA_ROOT`, `D3DS2_PY`, `FENICS_PY`로 통일해야 한다.

### 9.2 GitHub clone 재현성

GitHub 복제본:

```text
/home/goya/SDL/DeepGenerativeDesign_GPT
```

현재 저장소에는 `data_real`, `data`, `external`이 없다. 반면
`codebase/README.md`에는 raw STL과 conditioning image가 추적되거나 제공된다고 적혀 있어
루트 README와 내용이 충돌한다.

해결 방법:

- 공개 가능한 최소 bracket STL과 conditioning sample 포함
- 또는 dataset 다운로드 링크와 checksum 제공
- private dataset이 필요한 경우 이를 명확히 표시하고 작은 smoke fixture 제공
- clone 직후 실행 가능한 `verify_install.py` 또는 `doctor.py` 제공

루트 `.gitignore`도 필요하다. 현재 `codebase/.gitignore`는 저장소 루트에 만들어지는
`experiments/`, `data/`, `data_real/`에는 적용되지 않는다.

### 9.3 Config validation

원본 workspace에서 대부분의 config 경로는 존재한다. 다음 cache는 현재 없다.

```text
/home/goya/SDL/3d_qd/data_real/caliper_stretch50/fea_shared.msh
```

또한 `codebase/configs/caliper_ffff.json`의 post/FEA 단계는 `caliper_nofix` geometry를
참조한다. 의도된 공유인지 config validator와 주석으로 확인해야 한다.

## 10. 권장 수정 순서

1. TSDF 부호를 수정하고 단위 테스트를 추가한다.
2. dense, sparse, final이 공유하는 geometry operator를 구현한다.
3. 동일 조건만 포함하는 prototype bank를 다시 만든다.
4. sparse macro scaffold와 dense-to-sparse consistency를 구현한다.
5. BC-connected component 기준으로 debris를 억제한다.
6. final target-hit test를 통과시킨다.
7. 독립 FEA와 Pareto archive를 연결한다.
8. archive-feedback 및 measure-gradient emitter를 구현한다.
9. 동일 예산 baseline 실험을 수행한다.
10. 경로, 환경, README, GitHub 패키징을 정리한다.

현재 바로 수행할 핵심 작업은 **sparse scaffold와 dense-to-sparse consistency를 구현하여
dense에서 만든 형상 변화가 final mesh까지 보존되는지 검증하는 것**이다. 이 조건을 통과한
뒤 learned descriptor와 대규모 QD 실험으로 진행하는 것이 계산 자원과 연구 논리 양쪽에서
가장 효율적이다.

## 11. 관련 절대 경로

- 전체 작업 원본: `/home/goya/SDL/3d_qd`
- GitHub 복제본: `/home/goya/SDL/DeepGenerativeDesign_GPT`
- 핵심 생성 코드: `/home/goya/SDL/3d_qd/codebase/code/generate_with_physics_guidance.py`
- Shape-QD loss: `/home/goya/SDL/3d_qd/codebase/code/shape_qd_loss.py`
- TSDF operator: `/home/goya/SDL/3d_qd/codebase/tsdf_shape_operator.py`
- Prototype builder: `/home/goya/SDL/3d_qd/codebase/build_valid_shape_prototypes.py`
- Shape representation pilot: `/home/goya/SDL/3d_qd/codebase/run_shape_qd_representation_pilot.py`
- 연구 구성: `/home/goya/SDL/3d_qd/codebase/docs/shape_aware_dqd_paper_plan.md`
- 해결 계획: `/home/goya/SDL/3d_qd/codebase/docs/shape_qd_resolution_plan.md`
- TSDF parity 결과: `/home/goya/SDL/3d_qd/experiments/bracket/shape_qd_tsdf_parity_2026-09-21/result.json`
- Macro scaffold 결과: `/home/goya/SDL/3d_qd/experiments/bracket/shape_dqd_scaffold_pilot_2026-09-22/niche_01`
