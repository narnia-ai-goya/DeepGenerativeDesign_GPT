# 의자 얇은 부재·연결성: 후속 연구 검토 (2026-10-02)

## 관찰된 실패 모드

`complex_truss_armchair`에서 모든 직선 support corridor를 제거하면 prototype은 한 성분이지만 dense에서 앞쪽 두 고정 발이 좌석에서 분리된다. 앞다리 경로만 복원하면 sparse는 한 성분이고 BC는 통과한다. 그러나 입력의 팔걸이·대각 보강재는 dense부터 소실되고 sparse에서 복원되지 않는다. 앞다리만 복원한 sparse의 front/right/top silhouette IoU는 0.5650/0.5819/0.7582다. 과거 sparse 64³ 이미지 투영 가중치 0.5·2의 변화도 외형을 거의 바꾸지 못했다(`sparse_style_image_guidance_2026-09-29/REPORT.md`).

현재 `ImageProjectionLoss`는 64³ occupancy를 front/right/top으로 투영해 foreground BCE를 계산한다. 이는 전체 실루엣을 맞추지만 실루엣 내부의 대각 부재·접합점·팔걸이 개구부를 각각 구분하지 않는다. dense 출력 후 64³ active-token 필터와 envelope/BC 합집합이 있고, sparse decoder는 이 active support에 크게 의존한다. 따라서 우선 타깃은 후반 sparse 가중치가 아니라 **dense의 얇은 부재 유지와 BC 간 연결**이다.

## 가장 관련된 연구

| 연구 | 논문의 사실 | 이 프로젝트에 대한 적용 추론 | 한계 |
|---|---|---|---|
| [SketchSplat, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Ying_SketchSplat_3D_Edge_Reconstruction_via_Differentiable_Multi-view_Sketch_Splatting_ICCV_2025_paper.html) | 정합된 여러 이미지에서 parametric 3D edge를 differentiable sketch splatting으로 직접 최적화하고, edge topology도 정돈한다. | 입력 의자의 대각재·팔걸이·접합점을 **2D edge/junction 타깃 → 3D graph 후보 → 재투영 손실**로 만들자. 고정 원기둥을 합집합하는 대신 graph 주변의 기존 dense 점유를 보존하도록 유도한다. | 원래는 CAD edge 재구성. 금속 사진의 하이라이트와 진짜 부재 경계를 분리해야 하며 구조 성능은 보장하지 않는다. |
| [clDice, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html) | 2D/3D 관형 구조의 skeleton과 mask 교차를 이용해 연결성을 평가하고 soft-clDice를 제안한다. | 검증된 3D guide graph가 있을 때만 국소 centerline continuity를 측정하거나 손실로 사용한다. | 잘못 정한 skeleton을 그대로 보존하므로 과거 직선 corridor의 막대를 되살릴 위험이 있다. whole-chair 전역 손실로 쓰지 않는다. |
| [Connectivity constraints via level set and spectral graph, SMO 2026](https://link.springer.com/article/10.1007/s00158-026-04327-5) | 고유값 기반 연결 제약으로 2D/3D TO를 수행한다. 논문은 연결성 제약을 만족해도 가는 막대나 균열 같은 실패가 생길 수 있고 필터·dilation·perimeter regularization 선택이 중요하다고 보고한다. | 지금의 'BC 통과 + 막대 같은 가짜 경로'가 연결성만의 지표로 부족한 이유를 설명한다. BC 연결 외에 최소 단면, 경로 길이, 이미지 부재 일치도를 함께 검사한다. | 512³ 전체 고유값 최적화는 비싸고 직접 도입 우선순위가 낮다. |
| [CrossSDF, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Walker_CrossSDF_3D_Reconstruction_of_Thin_Structures_From_Cross-Sections_CVPR_2025_paper.pdf) | 얇은 구조를 위한 SDF 손실과 적응적 표면 샘플링을 제안한다. | 전체 격자를 균등하게 다루지 말고 의도된 얇은 부재 주변 active tokens와 sparse SDF 샘플을 국소적으로 늘려보자. | 원래 입력은 2D cross-section이며 여기의 세 뷰 사진과 다르다. 방법 원리의 전용은 별도 검증이 필요하다. |
| [Direct3D-S2, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f9666a092153f281b5116cd1f64b5c91-Abstract-Conference.html) | sparse volume 기반 3D latent/VAE와 sparse attention으로 고해상도 생성을 효율화한다. | 현재 decoder의 세부 해상도가 높더라도 coarse active support에서 사라진 부재는 복원이 어렵다. dense active set·후처리 단계의 보존 검증이 선행돼야 한다. | 원 논문이 우리 BC-aware 변형의 재현성을 보증하지는 않는다. |

## 추천하는 학습 없는 pilot

1. **3D guide는 입력 이미지에서 얻는다.** front/right/top의 실제 부재와 교차 접합점만 표시하거나 추출한다. 하이라이트·그림자 edge는 제외한다. SketchSplat 원리처럼 각 3D edge 후보의 재투영이 세 뷰에서 맞는지 검사한다. 디자이너가 모호한 대응을 승인한다.
2. **고정 막대 대신 국소 soft target.** graph에서 최소 폭이 있는 *허용 튜브*를 만들되, 이를 mesh에 무조건 union하지 않는다. dense에서 해당 영역의 occupancy가 끊기지 않도록 손실과 active-token support를 조정하고, 매 단계에서 의도되지 않은 외부 점유를 벌점화한다. BC voxel만 경성 보존한다.
3. **dense gate를 세분화한다.** 기존 BC coverage·component 외에 앞다리 4개↔좌석, 좌석↔등받이의 경로와 최소 단면, 팔걸이/대각재의 3D guide coverage, front/right/top **edge reprojection**을 모두 기록한다. 실루엣 IoU만 좋고 얇은 부재가 사라진 후보는 sparse에 넘기지 않는다.
4. **sparse는 성공한 dense 후보만 세부화한다.** sparse 후 같은 지표를 다시 평가한다. 기존 64³ silhouette guidance 가중치만 반복 조정하지 않는다. FEA는 geometry gate를 통과한 후보에서만 재개한다.

첫 비교는 `front_support_only` 기준에 (A) edge/junction 재투영 손실, (B) 그 손실+graph 부근 active support 확장, (C) BC 연결·최소 단면 gate를 순차 적용하는 것이다. 각 실험은 같은 세 뷰·BC·envelope·seed를 유지한다. 성공은 가짜 막대 감소와 앞다리 BC 연결을 **동시에** 만족하면서 대각재와 팔걸이의 재투영/3D 연결성이 개선되는 경우로 정의한다. 이는 논문들을 조합한 제안이며 아직 검증된 결과가 아니다.

현재 비교 렌더: `/home/goya/SDL/3d_qd/experiments/chair/sofa_style_2026-09-28/front_support_only_2026-10-02/dense_sparse_comparison.png`
