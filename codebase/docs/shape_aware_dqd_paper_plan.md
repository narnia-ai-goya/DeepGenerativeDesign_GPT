# Shape-Aware Differentiable Quality Diversity for Image-Conditioned Structural Mesh Generation

작성일: 2026-09-21  
상태: 논문 및 구현의 **현재 메인 구성**

## 한 문장

고정된 support/load BC와 envelope 아래에서, image-conditioned 3-D 생성기가 만드는
다양한 **형상 phenotype**을 미분가능하게 조준하고, 최종 독립 FEA로 각 형상 family의
기계 품질을 검증한다.

## 문제 정의

입력은 reference image `I`, BC `B`, envelope `D`다. 생성기는 dense/sparse 과정에서
occupancy field `rho`를 만들고 final mesh `M`을 낸다.

* 제약 `g(M)`: BC solidity, envelope containment, watertightness, 단일 component.
* Diversity `z(M)`: mesh의 geometry-only learned phenotype descriptor.
* Quality `q(M)`: independent final FEA compliance와 material volume의 Pareto quality.

따라서 목표는 하나의 최소-compliance mesh가 아니라, shape niche `k`마다 제약을
만족하고 품질이 높은 mesh를 찾는 것이다.

## Method

### 1. Frozen shape phenotype space

reference mesh bank를 동일한 world coordinate와 camera로 표현한다.

`phi(M) = [top/front/side multiscale silhouette, depth, masked SDF]`

고정 BC와 envelope에서 항상 같은 부분은 mask한다. `phi`는 이미지의 색·조명·text
style이 아니라 geometry만 담는다.

`E: phi -> z`는 pilot mesh bank로 사전학습하고 동결한다. 첫 구현은 PCA이며,
convolutional autoencoder/contrastive encoder가 제안 표현이다. encoder가 동결된 뒤에는
archive 좌표계와 모든 centroid를 바꾸지 않는다.

### 2. Feasible shape archive

pilot mesh embedding `z`에 CVT를 수행해 `K`개 niche centroid `{c_k}`를 만든다.
정사각 grid의 비어 있는 영역을 coverage 실패로 세지 않는다. 한 niche에는 하나의
scalar winner 대신 `(compliance, volume)` Pareto front를 저장한다.

### 3. Dense/sparse direct targeting

각 생성 시 target niche `c_k`를 고른다. soft occupancy로 differentiable `phi_tilde(rho)`를
계산하고, 동결 encoder를 통과시킨다.

`L_total = L_BC + L_envelope + L_connectivity + L_min-feature + lambda_FEA L_compliance + lambda_shape ||E(phi_tilde(rho)) - c_k||^2`

* dense: 큰 silhouette/layout 변화를 만들도록 `L_shape`를 먼저 적용한다.
* sparse: detail과 topology를 유지하되, 동일 target에 약한 refinement guidance를 적용한다.
* FEA: QD descriptor가 아니며, 생성 중 optional quality guidance와 final quality 검증에
  사용한다.

final mesh를 다시 `E(phi(M))`로 encode해 실제 niche를 판정한다. proxy coordinate가 아닌
final mesh coordinate로만 archive에 삽입한다.

## 선행연구와의 관계

| 선행 | 가져오는 원리 | 본 연구가 추가하는 것 |
|---|---|---|
| MAP-Elites | user-relevant phenotype space를 niche로 나누고 elite를 보관 | fixed-BC structural mesh repertoire |
| AURORA / unsupervised QD | 관측 phenotype에서 learned behavior descriptor를 만든다 | mesh geometry observation, offline-frozen representation, FEA-validated quality |
| DQD / MEGA | differentiable objective와 measure gradient로 QD 탐색 | descriptor gradient를 image-conditioned dense/sparse generation trajectory에 직접 결합 |
| QD + topology optimization | 서로 다른 topology/design detail의 고품질 구조 해를 보존 | hand-crafted void descriptor를 learned shape phenotype으로 확장하고 direct targeting |

따라서 주장하지 않는 것: learned descriptor QD의 최초 제안, DQD의 최초 제안, QD+TO의
최초 적용. 주장하는 것은 이 세 요소를 구조 mesh 생성과 최종 FEA 검증 안에 결합한
방법이다.

## 실험 설계

### RQ1. Learned geometry descriptor가 실제 형상 다양성을 나타내는가?

* `E`의 niche distance와 independent 3-D SDF/IoU distance의 상관.
* 같은/다른 niche pair를 blind visual audit으로 확인.
* topology, void-layout, silhouette distance는 분석 지표로 보고하되 QD quality와 섞지 않는다.

### RQ2. Direct shape guidance가 target niche를 실제로 조준하는가?

* 고정 `I, B, D, seed`에서 centroid별 5회 이상 생성.
* final target distance, hit rate, invalid rate를 dense only와 dense+sparse로 보고.
* 먼저 1-D PCA/encoder coordinate sweep으로 성립을 확인한 뒤 2-D CVT archive로 확장.

### RQ3. 제안법이 같은 FEA 예산에서 더 넓고 좋은 shape repertoire를 만드는가?

* coverage: occupied CVT niches / feasible niches.
* quality: niche별 Pareto HV의 합(QD-HV), 최선 compliance, volume.
* independent geometric diversity: pairwise SDF/IoU 및 topology/void-layout statistics.
* feasibility: BC, containment, watertightness, connectedness.

## Baselines

| 방법 | 분리하는 효과 |
|---|---|
| random image/seed sampling | QD archive와 direct targeting 없이 prior만 사용 |
| hand-crafted geometric MAP-Elites | 기존 QD+TO식 void 위치/크기 descriptor의 가치 |
| learned-shape post-hoc QD | learned descriptor만 쓰고 생성 중 guidance하지 않는 경우 |
| DQD with PCA descriptor | differentiable targeting은 같고 encoder 표현만 단순한 경우 |
| **proposed frozen learned-shape DQD** | representation과 direct targeting의 결합 |

모든 방법은 동일 image, BC, envelope, mesh budget, final FEA budget으로 비교한다.

## 구현 순서와 go/no-go

1. **Representation bank:** 현재 bracket mesh bank를 수집·정렬하고 geometry feature를 만든다.
2. **PCA baseline:** frozen PCA + CVT archive를 구현해 archive의 visual coherence를 검증한다.
3. **Direct targeting pilot:** PCA 첫 두 좌표를 각각 5개 target으로 guidance한다.
4. **Learned encoder:** PCA보다 shape-distance correlation/targetability가 개선될 때만 AE 또는
   contrastive encoder를 메인으로 승격한다.
5. **Final comparison:** RQ3의 모든 baseline을 동일 예산으로 실행한다.

go/no-go는 Step 3에서 final target hit와 visual shape difference가 모두 확인되는지다.
실패하면 learned descriptor를 archive-only 분석으로 한정하고, direct DQD 기여를 주장하지
않는다.

## 그림 구성

1. Input image + BC/envelope -> dense/sparse generator -> soft geometry renderer -> frozen
   encoder -> shape loss -> final mesh/FEA의 방법 그림.
2. CVT archive: 각 niche의 top/front/side mesh thumbnail, 색은 Pareto quality/HV.
3. 같은 image/BC에서 target niche를 바꾼 morphologic traversal.
4. 같은 FEA budget의 coverage-QD-HV 곡선 및 mechanical validity table.

## 제목 후보

* Shape-Aware Differentiable Quality Diversity for Image-Conditioned Structural Mesh Generation
* Differentiable Illumination of Structural Mesh Repertoires from Image Priors
