# QD-LLMs와 의자 image-conditioned 구조 QD의 관계

검토 원문: `/home/goya/SDL/3d_qd/references/QD_LLM.pdf` (Koh et al., GECCO Companion 2026, pp. 1408–1415, DOI: 10.1145/3795101.3814651).

## 논문의 실제 기여

QD-LLMs는 자연어 prompt를 genotype, Shap-E로 만든 자동차 3D 형상을 phenotype으로 쓰는 MAP-Elites 계열 폐루프다. LLM을 prompt 변이 연산자로 사용한다. Uniform emitter는 점유 셀에서 균일하게 parent를 뽑고, sparse emitter는 이웃이 적은 셀을 선호한다. Directional emitter는 빈 셀을 목표로 source/guide prompt를 골라 목표 descriptor 방향으로 crossover한다. 별도의 LLM allocator가 archive 및 emitter 성공 이력을 보고 세 emitter의 예산을 조정한다 (원문 §3, Fig. 1–3).

자동차 실험은 *volume*과 *vertical center of gravity*를 20×20 QD descriptor로 사용한다. 품질은 vision-language domain alignment와 drag surrogate를 가중합한 scalar fitness다. GPT-4o-mini로 prompt를 만들고 Shap-E로 3D를 생성하며, 각 방법을 2,000회 × 5 offspring = 10,000형상 규모로 비교했다 (원문 §4). Table 2에는 QD-LLMs의 평균 QD-score 0.8828, domain score 0.9771, DPAR 0.9200이 제시된다. 이 값은 자동차 형상·drag 대리모델의 결과이며 의자 강성으로 전이되는 증거는 아니다.

## 겹침과 차별화 경계

| 요소 | QD-LLMs | 현재 의자 연구 |
|---|---|---|
| 탐색 genotype | 자유 자연어 prompt | designer가 정한 불변 specification + 편집 가능한 형상 문구 |
| 생성 경로 | prompt → Shap-E 3D | text + reference image → 이미지 편집 → Direct3D-S2 3D; BC는 별도 3D specification으로 검사 |
| QD descriptor | 자동차 volume, vertical center of gravity | 3D에서 측정한 side opening, backrest taper 등 형상 phenotype |
| 구조 유효성 | 자동차 domain alignment의 soft score/threshold | 고정·하중 BC 접촉, envelope, 수선량, 연결성의 hard gate |
| 품질 | domain/drag scalar fitness | 질량·좌석/등받이 compliance의 다목적 비교를 목표로 함 |
| designer | 일반적인 목표 영역 지정 | 현재 style/BC specification을 조정하며, 목표 niche 직접 선택은 제안 단계 |

따라서 **LLM을 QD prompt emitter로 사용한다**, **빈 archive 셀을 prompt로 겨냥한다**, **자연어 변이로 latent perturbation을 대체한다**는 주장은 신규성이 없다. 우리 논문의 기여는 image-conditioned 3D 구조 설계에서 specification과 기능 인터페이스를 보존하는 방법, 3D 형상 descriptor와 구조 품질의 결합, designer가 실패 원인을 보며 다음 후보를 조정하는 절차로 좁혀야 한다.

현재 구현 상태를 과장해서는 안 된다. 의자 9개 고유 후보의 3×3 archive에 기존 후보 3셀이 들어갔지만 새 tapered 5개는 BC/수선량 gate에서 모두 탈락했다. 기존 FEA도 64³ 수선 voxel과 공통 35 mm tetra mesh에 대한 proxy이므로, 아직 고해상도 원본 OBJ의 독립 검증이라고 쓰면 안 된다. QD-LLMs와의 성능 우위는 아직 측정하지 않았다.

## 바로 적용 가능한 방법론

1. QD-LLM/UE·DE를 **선행 baseline**으로 재구현하되 동일한 이미지 모델, 3D generator, 총 호출 수로 비교한다. 자동차 논문의 allocator나 directional emitter 자체를 신규 기여로 포장하지 않는다.
2. Prompt를 불변 `specification`(4 feet, seat/back load contacts, envelope, camera)과 변이 가능한 `shape clause`로 분리한다. Parent는 BC를 통과한 elite에서만 선택한다.
3. Directional prompt에 목표 형상 셀뿐 아니라 실패 진단(예: seat 16–32 mm below load; back BC 12%)을 제공한다. 이미지 단계의 싼 검사 후 3D로 승격시키고, 3D BC gate 통과 뒤 FEM을 실행한다. 이는 이 논문의 soft domain threshold보다 강한 구조 조건이지만 유효성 향상은 실험으로 입증해야 한다.
4. Archive는 *realized mesh*의 형상 descriptor로 채우고, 셀 내부에서는 질량과 compliance의 Pareto 집합을 보존한다. Text embedding은 후보 제안에만 쓰며, 실현 형상 다양성과 혼동하지 않는다.
5. 동일 예산에서 random prompt, fixed text novelty, QD-LLM/UE, QD-LLM/DE, feasibility-aware variant를 비교한다. 지표는 유효 셀 수, 유효 후보 비율, 질량–compliance hypervolume, 디자인 의도 충실도, 유효 elite 1개당 계산 비용이다.

Figure 3의 일반적인 `prompt → generator → evaluation → archive → prompt` 순환 구조는 이미 이 논문에 있다. 우리 overall figure는 이미지 중간 단계, 명시적인 BC/envelope gate, 질량–구조성능 Pareto archive, designer feedback을 중심으로 설명해야 한다.
