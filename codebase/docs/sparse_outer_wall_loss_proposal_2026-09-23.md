# Sparse 단계 외벽 패임: 손실과 표현 영역 검토

## 관찰

대상은 bracket의 고정단 근처 **바깥 옆벽에 열린 패임**이다. 관통 구멍이나 분리된 컴포넌트와 같지 않다. 같은 dense cache, seed, BC, FEA-on 조건에서 dense-core 평균 손실, 최악 0.1/0.5% 손실, 강제 SDF 투영은 이를 자연스럽게 없애지 못했다. 비교 결과: `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/shape_language_angular_2026-09-23/sparse_hole_prevention_2026-09-23/index.html`.

현재 `topology_preservation_loss`는 64³ active-cell의 안팎과 cone 방향에서 부호를 감독한다. `sp_dense_core`는 dense 표면에서 0.5 mm 이상 들어간 내부만 감독한다. 표면에 열려 있으나 관통하지 않은 국소 패임은 Betti 수나 연결 컴포넌트를 바꾸지 않을 수 있고, 내부 평균 손실에서는 희석된다. 최악값에 집중한 CVaR 손실도 late steps에서 0.30486 부근으로 포화했으며 결과 패임은 남았다. 이 수치는 `softplus(3)/10`과 일치하여 SDF 값 1.0의 빈 공간이 최악 샘플을 지배할 가능성이 있다. 실제 gradient 소실 여부는 추가 계측 전까지 추정이다.

64³ dense cache에서 active token은 13,539개다. dense 표면 SDF를 64³ voxel center에서 샘플하면 내부 0.5 mm 이상인 13,539개는 모두 active지만, 표면 바깥 1–2 mm에 있는 5,838개 voxel center는 모두 inactive였다. 이는 coarse grid의 half-cell 표면 때문에 예상되는 결과이지만, sparse decoder가 벽을 바깥으로 복원할 수 있는 표현 영역이 제한됨을 보여 준다. 기존 `thin_expand_vox`는 dense cache 로드 시 실행되지 않는 branch 안에 있다. 그래서 캐시 기반 대조군에는 효과가 없다.

## 1순위: 옆면 front-surface loss, 필요할 때만 sparse support 확장

표면 손실은 먼저 **기존 support에서** 시험한다. 그 loss가 지적한 ray에 sparse 출력 좌표가 없어 gradient가 흐르지 않을 때에만 64³ dense active set의 바깥에 한 voxel halo를 추가한다. 원래 dense mask, dense mesh, BC와 topology target은 그대로 둔다. 새 옵션 `sp_support_halo_vox`는 기본 0이며 캐시 로드와 정상 생성 양쪽에서 적용되도록 구현했다.

한 층 전체 halo의 FEA-on 대조군은 13,539 → 19,377 tokens(+43.1%), raw mesh 16 components, 699.2 cm³였다. 기준 546.2 cm³보다 28.0% 크고 외벽에 반복되는 줄무늬/계단 아티팩트가 생겼다. 따라서 **전역 halo 단독은 기각**한다. halo를 다시 쓴다면 `new_token` 부분의 기본 점유를 낮추는 exterior-empty prior와 side-ray 손실을 함께 쓰거나, gradient가 실제로 부족한 ray 인근에서만 활성 토큰을 추가해야 한다. 그 전에는 기존 support를 유지하는 편이 안전하다. 결과 mesh: `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/shape_language_angular_2026-09-23/outer_wall_sparse_support_2026-09-23/halo1_actual/generation/mesh.obj`.

새 손실은 측면 카메라의 기준 깊이 `d_ref(p)`와 sparse SDF의 soft first-hit 깊이 `d_sp(p)`를 비교한다. sparse SDF는 값이 `tau`보다 작을 때 재료다. 각 측면 ray의 샘플 점 `x_i`에 대해 `rho_i = sigmoid(k*(tau-f(x_i)))`, `alpha_i = rho_i * product_{j<i}(1-rho_j)`, `d_sp = sum_i(alpha_i*z_i)/(sum_i alpha_i+eps)`로 잡는다. 기준 dense 메쉬에서 실제 벽으로 보이는 픽셀에만 `L_recess = mean softplus((d_sp-d_ref-delta)/T)*T`를 적용하고, 적중 확률 `sum_i alpha_i`가 너무 낮은 픽셀에도 penalty를 준다. `delta`는 허용할 표면 이동 폭이다. 의도된 관통 공간의 픽셀에는 `L_void = mean sum_i rho_i`를 적용하여 벽을 무작정 채우지 못하게 한다. 반대 방향 측면 두 장을 사용한다.

이 손실은 전체 체적이나 모든 내부 voxel을 누르는 대신 **실제로 보이는 외벽의 과도한 후퇴**만 지적한다. 따라서 QD가 요구하는 개구부와 다른 형상 변형에는 여지를 남길 수 있다. 기준 깊이는 카메라 자세가 알려진 동일 dense mesh 렌더에서 얻는다. GPT가 만든 side image는 시점/형상 일관성이 검증되지 않으면 깊이 정답으로 사용하지 않는다. 옆면 이미지 생성 token을 단순 연결하는 현재 multi-view encoder에는 카메라 자세가 전달되지 않으므로, 이 손실은 그 인코더와 별도로 sparse SDF에 직접 적용해야 한다.

512³ 전체 grid에 differentiable rendering을 걸지 않는다. sparse decoder가 출력한 좌표에서 측면의 벽 ray만 선택하여 depth/opacity를 계산한다. 샘플 위치에 좌표가 없으면 `rho=0`이고 gradient도 없으므로, halo의 이득을 loss 이전에 검증한다. dense 주변 surface sample별 normal-line 부호/단일 crossing 손실은 카메라가 없는 경우의 대안이다. 다만 dense의 64³ 계단 형상을 복사하지 않도록 수 mm tolerance를 둔다.

## 우선순위가 낮은 대안

Persistent-homology/Betti 손실은 실제 관통 구멍이나 연결 단절에는 적합하지만, 표면에 열린 작은 패임은 topology가 같을 수 있으므로 이번 결함의 주 손실로 쓰지 않는다. soft-clDice는 가느다란 튜브의 연결성 보존에 강점이 있지만 bracket의 넓은 판형 외벽에는 직접적인 감독이 약하다. Eikonal이나 Laplacian은 SDF/곡면의 규칙성에는 도움을 줄 수 있으나 패임 위치를 알려 주지 않고, 기존 smoothing 과다 문제를 악화시킬 수 있다.

근거: [Soft Rasterizer, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Soft_Rasterizer_A_Differentiable_Renderer_for_Image-Based_3D_Reasoning_ICCV_2019_paper.html), [NeuS, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/e41e164f7485ec4a28741a2d0ea41c74-Abstract.html), [Topology-Preserving Deep Image Segmentation, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/2d95666e2649fcfc6e3af75e09f5adb9-Abstract.html), [clDice, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf). 위 ray 손실의 구체식과 bracket 적용은 이 프로젝트를 위한 제안이지 이 논문들의 구현을 그대로 옮긴 것이 아니다.

## 실험 순서와 후속 검증

1. 완료: halo만 켠 FEA-on 대조군은 표현 공간만 키웠고 외벽 품질과 체적을 악화시켰다.
2. 완료: 기존 support에서 두 종류의 ray 손실을 구현·실행했다. ray 좌표 누락은 없었지만 양측 패임 제거에는 실패했다.
3. 후속: 동일 run의 sparse decoder field, pre-refiner MC, refiner 후 MC의 깊이를 같은 좌표계에서 계측한다. 이 결과가 가리키는 실제 손실 발생 단계에만 새로운 제약을 건다.
4. 좋은 조건이 발견된 뒤 여러 seed/이미지로 확대하고 final post, mass, 독립 FEA를 검증한다. 분리 컴포넌트 수만으로 성공을 선언하지 않는다.

## 구현 및 1차 실험 결과

두 종류의 손실을 기본 비활성 옵션으로 구현했다. `sp_wall_ray_w`는 dense 64³ 토큰의 첫 활성 칸 안쪽 512³ 샘플을 감독한다. `sp_side_depth_w`는 실제 dense mesh SDF를 카메라 방향으로 샘플해 좌·우 직교 측면의 첫 표면 깊이를 구한 다음, sparse SDF의 soft first-hit depth가 허용 오차보다 뒤로 물러나면 벌점을 준다. 모두 sparse denoising의 `sp_guide` 안에서 latent에 gradient를 준다. 구현은 `/home/goya/SDL/3d_qd/codebase/code/sparse_wall_ray_loss.py`, `/home/goya/SDL/3d_qd/codebase/code/sparse_side_depth_loss.py`, 연결 지점은 `/home/goya/SDL/3d_qd/codebase/code/generate_with_physics_guidance.py`이다. 손실의 gradient 및 의도된 빈 ray 무감독을 검증하는 단위 테스트 3개가 통과했다.

동일 dense cache·seed·BC·FEA ON으로 평균 front-band 가중치 2/5, worst-ray 5/20%, 실제 dense 깊이 평균/5% tail(허용 오차 3 mm)을 돌렸다. 모든 실행에서 참조 측면 ray 약 4.6만 개/방향에 sparse 좌표가 존재했다. 그렇지만 고정단 옆벽 패임은 양쪽에서 제거되지 않았다. 실제 깊이 평균형의 x-minus 후퇴 P95는 baseline 13.13 mm에서 15.33 mm로 악화되었고, x-plus는 7.91→7.92 mm로 거의 동일했다. 5% tail형도 x-minus 15.33 mm, x-plus 7.85 mm였다. 앞면 band 손실은 조각 수가 줄기도 했지만 3D 표면 품질을 개선하지 못했다. 결과와 옆면 렌더: `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/shape_language_angular_2026-09-23/outer_wall_sparse_loss_study_2026-09-23/index.html`.

**이 손실들은 실험용으로 남겨 두고 기본값 0을 유지한다.** Sparse guidance의 측면 깊이 손실은 중간 step에서 작아졌다가 마지막 denoising step에서 다시 커졌고, 최종 refiner 메쉬에서는 패임이 남았다. 따라서 단순히 weight를 올리기보다, 같은 run의 (1) sparse decoder field, (2) pre-refiner MC, (3) refiner 후 MC를 **같은 월드 프레임의 ray-depth**로 비교해야 한다. 이 측정 전에는 결함이 decoder prior의 포화인지 refiner에 의한 변화인지 단정할 수 없다. 어느 단계가 재료를 빼는지 찾은 다음 그 단계의 출력에 맞는 손실 또는 제약을 적용한다. 현재 독립 최종 FEA와 post-processed mesh 평가는 하지 않았다.

이를 위해 평균 실제-깊이 손실 조건을 `D3DS2_SAVE_PRE_REFINER=1`로 재실행했다. 월드 프레임으로 변환한 pre-refiner OBJ는 `/home/goya/SDL/3d_qd/experiments/bracket/designer_steered_semantic_bo_qd_pilot_2026-09-22/shape_language_angular_2026-09-23/outer_wall_sparse_side_depth_2026-09-23/depth_mean_w05_tol3_diagnostic/generation/mesh_pre_refiner_world.obj`이다. 동일 실행에서 x-minus 후퇴 P95는 pre 14.53 mm → post 13.39 mm, x-plus는 8.28 → 8.00 mm였다. 따라서 refiner 전에도 결함이 있고 refiner는 평균적으로 완화했다. **이번 조건의 주 원인은 sparse decoder/denoising 쪽**이라는 이전 관찰과 일치한다. 다만 같은 설정의 앞선 실행에서는 x-minus 최종 P95가 15.33 mm였다. 설정상 같은 seed라도 실행 간 결과가 달라졌으므로, 원인을 찾기 전까지 작은 수치 차이를 방법의 우열로 해석하지 않는다. 다음 계측은 refiner 전 SDF에서 결함 ray별 `sdf`, `dL/dsdf`, `dL/dlatent` 및 마지막 denoising step에서 모델 update가 guidance update를 얼마나 되돌리는지 기록하는 것이다.
