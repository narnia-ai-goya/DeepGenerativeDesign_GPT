# Shape-QD direct-targeting recovery plan

작성일: 2026-09-21

## 실측 결론

첫 direct-DQD run과 sparse-DQD run은 서로 다른 final mesh를 만들었지만, 현 48^3
archive feature에서는 동일 embedding, 동일 niche 08, 동일 target distance 11.572로
판정됐다. 따라서 현 frozen-PCA representation은 형상 archive의 exploratory plot에는
쓸 수 있어도, direct gradient target의 진실값 또는 논문 지표로 쓸 수 없다.

## 실패 원인

1. **평가 해상도 손실:** `trimesh.voxelized(...).fill()`의 48^3 hard occupancy가
   sub-voxel branch/void 변화를 지운다.
2. **proxy 불일치:** archive는 hard mesh voxelization, dense는 64^3 sigmoid occupancy,
   sparse는 active-token 평균이다. 같은 `phi`를 측정하지 않는다.
3. **silhouette saturation:** ray-wise soft OR는 한 ray에 재료가 조금만 있어도 1에
   가까워져 내부 형태에 대한 gradient가 약하다.
4. **표현 bank 오염:** 422개 bank에는 부서진/비검증 mesh와 서로 다른 postprocess
   recipe가 섞여 있다. CVT centroid가 동일 BC/image 조건에서 도달 가능한 morphology를
   보장하지 않는다.
5. **sparse support 제약:** sparse stage는 dense active tokens에서만 시작한다. 목표
   branch가 그 support 밖에 있으면 loss gradient만으로 새 topology를 만들기 어렵다.
6. **quality loop 부재:** pilot은 raw `mesh.obj`에서 끝났고 final postprocess 및
   independent FEA로 elite quality를 검증하지 않았다.

## 권장 재설계

### A. Valid, condition-compatible bank

동일 image/BC/envelope/low-resolution generator recipe로 100--200개 후보를 만들고,
watertightness, connectedness, containment를 통과한 mesh만 bank에 넣는다. representative
subset에는 final postprocess와 tet FEA를 수행한다. 기존 RAB expanded의 18 valid case는
bootstrap으로만 쓰며, heterogeneous historical mesh 전체를 encoder training set으로 쓰지
않는다.

### B. 하나의 geometry measurement operator

`Phi(M)`를 fixed design-domain lattice의 96^3 또는 128^3 signed-distance/occupancy
field로 정의한다. final mesh, dense occupancy, sparse occupancy가 같은 world frame과
same design mask에서 이 operator를 사용한다.

multi-scale average pools (128, 64, 32, 16)와 top/front/side accumulated density/depth를
함께 저장한다. hard voxel fill과 saturated soft OR는 archive 지표에서 제거한다.

### C. Descriptor와 control loss를 분리

archive descriptor는 frozen geometry encoder `E(Phi(M))`와 feasible-CVT niche로 둔다.
그러나 direct loss는 centroid embedding 하나에 맞추지 않는다. 각 niche의 valid
prototype/barycenter field `Phi_k`를 만들고 다음 proximity loss를 쓴다.

`L_anchor = sum_s w_s ||Pool_s(Phi_tilde(rho)) - Pool_s(Phi_k)||_1`

이는 "어떤 형상 family로 갈 것인가"를 low-frequency부터 실제 공간 형상으로 전달한다.
`E`는 archive를 의미 있게 분류하고, `L_anchor`는 생성기가 따라갈 수 있는 gradient를
제공한다.

### D. Dense-to-sparse support 전달

target niche prototype의 coarse support를 64^3에서 dilate해 dense active mask와 union한다.
이 union을 512 sparse tokens까지 올린 뒤 sparse refinement를 시작한다. 목표 morphology가
초기 dense support 밖에 있어도 branch/void가 자랄 공간을 확보한다.

`active_sparse = upsample(dilate(active_dense union support(Phi_k)))`

BC/outside constraints는 target support보다 우선한다.

### E. DQD emitter

직접 centroid를 무조건 당기는 방식 대신, archive의 빈/저밀도 niche 근처 prototype을
고르고 `L_anchor`로 local proposal을 만든다. 같은 parent에서 shape loss weight와
orthogonal geometry-Jacobian direction을 여러 개 만들어 QD emitter를 구성한다. final
measurement가 실제 niche에 들어간 proposal만 archive에 삽입한다.

## 단계별 go/no-go

1. **Measurement test:** 의도적으로 서로 다른 20 mesh가 high-res `Phi`에서 distinct하고,
   같은 mesh의 dense/sparse/final representation distance가 작아야 한다.
2. **Gradient test:** 한 target prototype에 대해 one-step latent update 뒤 `L_anchor`가
   감소하는지, gradient norm이 BCE/regularization과 같은 order인지 측정한다.
3. **1-D targetability:** compatible prototype 5개를 골라 final `Phi` distance와 target-hit
   rate를 측정한다. 이 단계에서 hit가 없으면 2-D archive를 열지 않는다.
4. **QD run:** coverage, per-niche Pareto HV, FEA validity, independent SDF diversity를
   같은 final mesh budget에서 baseline과 비교한다.

## 필수 baseline

* random seed/image sampling
* existing post-hoc learned-shape archive
* hand-crafted void-layout MAP-Elites
* dense-only anchor guidance
* dense+sparse anchor guidance with target-support expansion

## 논문 주장 조건

"direct DQD"는 final high-resolution shape descriptor target hit와 independent FEA quality가
모두 보여질 때만 주장한다. embedding coverage만 늘어난 결과, 또는 raw mesh에서만
측정한 결과는 direct-QD evidence가 아니다.
