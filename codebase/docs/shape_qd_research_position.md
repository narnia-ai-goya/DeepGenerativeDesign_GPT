# Shape-QD research position

작성일: 2026-09-21

## 결론

형상 자체를 QD의 diversity로 쓰는 것은 임의의 아이디어가 아니다. QD에서 behavior
descriptor는 genotype이 아니라 **solution의 phenotype에서 측정한 표현**이어야 하며,
그 표현을 자동으로 학습하는 unsupervised-QD 계열이 이미 있다. 우리 문제의 새로움은
이를 fixed-BC 구조 mesh 생성 및 in-generation differentiable guidance에 연결하는 데
있다.

## 직접적으로 연결되는 선행연구

| 연구 | 핵심 | 본 연구에서의 의미 |
|---|---|---|
| Mouret & Clune, MAP-Elites (2015) | 사용자가 고른 phenotype feature space의 각 niche에서 최고 quality 해를 보관 | "형상 family별 좋은 bracket"이라는 archive의 기본 정의 |
| Cully, AURORA (GECCO 2019); Grillotti & Cully (2021) | raw observation을 PCA/autoencoder로 저차원 behavior descriptor로 학습 | mesh의 multi-view silhouette/depth 또는 occupancy를 observation으로 놓는 근거 |
| Fontaine & Nikolaidis, DQD/MEGA (NeurIPS 2021) | objective와 measure의 gradient가 있을 때 QD exploration에 활용 | soft mesh representation에서 shape descriptor gradient를 dense/sparse guidance로 전달하는 근거 |
| Xiong et al., Evolutionary Seeding of Diverse Structural Design Solutions via TO (2024) | 같은 stiffness라도 다른 shape/detail은 다른 structural-design descriptor여야 함 | 구조성능을 quality로, 형상을 diversity로 분리하는 직접적 engineering precedent |

## 권장 방법: frozen shape manifold + CVT archive

각 mesh `M`을 BC와 envelope이 정렬된 공통 좌표계에서 다음 phenotype으로 바꾼다.

`phi(M) = concat(multiscale top/front/side soft silhouette, depth, masked signed-distance)`

* 렌더 색, 조명, prompt가 아니라 geometry만 사용한다.
* 모든 후보에서 동일한 BC/envelope pixel은 제외해 형상이 아닌 조건이 거리를 지배하지
  않게 한다.
* low-resolution view는 전체 outline/branching을, higher-resolution distance map은
  hole/ligament/void placement를 보존한다.

reference mesh collection으로 encoder `E`를 **사전 학습하고 동결**한다.

`z(M) = E(phi(M))`

2-D visualization은 PCA/UMAP projection으로 만들 수 있지만, 실제 archive는 2-D
regular grid보다 `z` 공간의 feasible samples에서 만든 **CVT-MAP-Elites centroids**를
사용한다. 이렇게 하면 빈 사각 격자가 "생성 불가능한 형상"을 뜻하는 일을 줄인다.
각 centroid는 한 shape niche이고, 그 niche에는 compliance와 volume Pareto elite를
보관한다.

## 왜 online AURORA를 그대로 쓰지 않는가

AURORA는 encoder를 계속 다시 학습하고 모든 descriptor를 재계산한다. open-ended
behavior discovery에는 장점이지만, 이 연구에서는 run 도중 archive 좌표계가 움직이면
coverage·target hit·baseline 비교가 바뀐다. 따라서 pilot reference set으로 encoder와
CVT centroids를 동결한다. 이는 "무엇을 diverse하다고 볼 것인가"를 실험 전에
고정하는 재현성 조건이다.

## in-generation QD

soft occupancy `rho`에서 differentiable render를 만들어 같은 encoder에 넣는다.

`L_shape = lambda_shape * ||E(phi_tilde(rho)) - c_k||^2`

여기서 `c_k`는 선택된 CVT niche centroid다. 마지막에는 marching cubes final mesh를
독립적으로 encode해 실제 niche를 판정한다. 따라서 training/guidance proxy가 archive
정의를 바꾸지 않는다.

## 논문상 검증할 주장

주장은 "embedding coverage가 커졌다" 하나로 끝나면 약하다. 다음을 함께 보인다.

1. **형상 coverage:** 고정 encoder/CVT niche coverage 및 nearest-neighbor phenotype
   distance.
2. **독립 형상 검증:** pairwise 3-D SDF/IoU distance, topology 및 void-layout 통계.
   encoder가 만든 가짜 다양성이 아님을 확인한다.
3. **기계 quality:** niche별 final FEA compliance와 volume Pareto front.
4. **제약 준수:** BC solidity, envelope containment, connectedness, watertightness.
5. **조준 가능성:** 목표 centroid와 final mesh embedding의 거리 및 target-hit rate.

필수 baseline은 random seed/prompt sampling, post-hoc latent-space QD, hand-crafted
shape descriptor QD(예: void 위치·크기), 그리고 제안하는 frozen learned-shape QD다.

## 차별점의 정확한 범위

"learned descriptor QD" 자체는 AURORA가 선행한다. "differentiable QD" 자체는
MEGA가 선행한다. 본 연구의 기여 후보는 **고정된 boundary condition을 가진
image-conditioned 3-D structural mesh generator에서, geometry-only learned phenotype
descriptor를 dense/sparse 생성 궤적에 직접 guidance하고, 최종 independent FEA로
quality를 검증하는 것**이다.

## 참고 문헌

* Mouret, J.-B. and Clune, J. *Illuminating search spaces by mapping elites*, 2015.
  https://arxiv.org/abs/1504.04909
* Cully, A. *Autonomous skill discovery with Quality-Diversity and Unsupervised
  Descriptors*, GECCO 2019. https://doi.org/10.1145/3321707.3321804
* Grillotti, L. and Cully, A. *Unsupervised Behaviour Discovery with Quality-Diversity
  Optimisation*, 2021. https://arxiv.org/abs/2106.05648
* Fontaine, M. C. and Nikolaidis, S. *Differentiable Quality Diversity*, NeurIPS 2021.
  https://papers.nips.cc/paper/2021/hash/532923f11ac97d3e7cb0130315b067dc-Abstract.html
* Xiong, X. et al. *Evolutionary Seeding of Diverse Structural Design Solutions via
  Topology Optimization*, ACM TELO, 2024. https://doi.org/10.1145/3670693
