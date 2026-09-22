# QD 별도 논문 기획 — Quality-Diversity가 무엇이고, 어디서 이 연구와 맞물리는가

> **2026-09-21 상태:** 이 문서는 초기 `질량비 × 두께` descriptor 기획을 보존한다.
> 현재 메인 구성은 geometry-only learned phenotype descriptor와 direct DQD guidance를
> 사용하는 [`shape_aware_dqd_paper_plan.md`](shape_aware_dqd_paper_plan.md)다. 이 문서의
> 축·claim·실험 순서를 현재 논문 계획으로 인용하지 않는다.

**전제**: 별도 논문으로 간다. 따라서 "QD를 적용했다"는 기여가 아니다.
**선행 문서**: [`qd_extension.md`](qd_extension.md) — 기존 논문에 QD를 *얹는* 시나리오. 예산 전략(Tier 0/1/2)은 여전히 유효하지만, 별도 논문에서는 §4의 **조준(targeting)** 전략이 더 근본적인 해법이다.

작성일 2026-09-01.

---

## 1. QD란 무엇인가

### 정의

Quality-Diversity는 **하나의 최적해가 아니라, 사용자가 정의한 특징 공간을 고르게 덮는 고성능 해집합을 한 번의 실행으로 찾는** 최적화 패러다임이다 (`00번 논문`).

세 요소로 구성된다:

| 요소 | 기호 | 이 연구에서 |
|---|---|---|
| **Genotype** | x | 탐색 변수 (prompt, guidance weight, seed) |
| **Descriptor** (= behavior/measure) | b(x) ∈ B | 다양성을 정의하는 축 (질량비, 두께, 구멍수) |
| **Objective** (= fitness/quality) | f(x) | 품질 (−compliance) |

descriptor 공간을 격자로 나누고 (셀 = niche), **각 셀마다 그 셀에 속하는 최고 성능 해 하나**를 보관한다. 이 격자를 archive라 하고, MAP-Elites가 표준 알고리즘이다:

```
반복:
  archive에서 부모를 뽑아 변이 → x'
  x' 평가 → f(x'), b(x')
  b(x')가 속한 셀의 기존 elite보다 f(x')가 높으면 교체
```

### 무엇이 아닌가 — 구분이 중요하다

| 혼동 대상 | 차이 |
|---|---|
| **다목적 최적화 (MOO)** | MOO는 목적 간 **trade-off**(Pareto front)를 찾는다. QD의 descriptor는 목적이 아니다 — 최대화 대상이 아니라 **분류 축**이다. 질량비 0.3인 해가 0.6인 해보다 "낫지" 않다. 그냥 다른 칸이다. |
| **다양성 정규화(diversity regularization)** | 해집합의 분산을 키우는 페널티가 아니라, **명시적으로 정의된 축**을 덮는다. 어디가 비었는지 알 수 있다. |
| **여러 번 재시작** | 재시작은 같은 최적해 근방으로 반복 수렴한다. QD는 archive가 부모 풀이 되어 **채워진 셀이 빈 셀로 가는 발판(stepping stone)**이 된다. |
| **샘플링 다양성** | 생성 모델의 온도/seed 다양성은 **측정도 제어도 안 된다**. QD는 축을 지정하고 커버리지를 잰다. |

### QD가 엔지니어링 설계와 맞는 이유

`00번 논문` 서두가 정확히 이 지점을 말한다:

> "optimization is most often used at the beginning of the design process to explore various options and examine the trade-offs that are inherent to the domain. This calls for algorithms that are designed as **exploration tools more than as pure optimization tools**."

그리고 본 연구 논문의 Discussion이 같은 말을 한다:

> "Structural-part exploration thus becomes an **interactive loop that yields many mechanically valid, stylistically distinct candidates**, rather than the single re-solved solution of classical topology optimization."

두 문장이 같은 것을 서술하고 있다. **논문은 이미 QD의 문제의식에 도달해 있으면서 QD의 도구를 쓰지 않고 있다.** 이것이 fit의 출발점이다.

---

## 2. 표면적 fit — 필요하지만 논문이 되지는 않는 것

이 연구에 QD를 붙이면 자연스럽게 얻어지는 것들:

- 다양성이 **측정된다** (coverage, QD-score). 현 논문 Table 1의 `Style` 열이 ✓/× 정성 표기에 머무는 문제가 해소된다.
- SIMP와의 격차가 **정량화된다**. Fig. 2의 "single smooth branching family" 주장이 `1/N 셀 vs M/N 셀`이 된다.
- 큐레이션 병목이 **탐색으로 대체된다** (Limitations 첫 항목).
- 실패 샘플이 **제약 학습 신호**가 된다 (현재는 "automatic filtering step"으로 폐기).

전부 타당하지만, **이것만으로는 별도 논문이 안 된다.** "기존 QD 알고리즘을 새 도메인에 적용" 이상이 아니기 때문이다. 진짜 기여는 다음 절에 있다.

---

## 3. 진짜 fit — 이 연구가 QD의 전제를 깨는 지점

QD 알고리즘은 특정 전제 위에 설계돼 있다. 이 연구는 그 전제를 **네 군데에서 위반한다**. 위반은 문제가 아니라 기여의 원천이다.

| # | QD의 표준 전제 | 이 연구의 실제 | 함의 |
|---|---|---|---|
| 1 | 평가가 **블랙박스**다 (gradient 없음) | **미분가능 FEM**이 이미 있다. ∂C/∂ρ가 latent까지 역전파된다 — 논문의 핵심 기여가 정확히 이것 | 목적함수 gradient 사용 가능 → **DQD 계열** |
| 2 | descriptor는 **평가 후에만** 알 수 있다 | 질량비·두께는 생성 중간(dense 64³)에 알 수 있고, **미분가능한 loss로 이미 구현돼 있다** (Table 2의 per-voxel thickness) | descriptor를 **유도**할 수 있다 |
| 3 | 변이는 genotype 공간의 **무작위 교란** | 생성 과정 **내부**(guidance loss)에 개입할 수 있다 | 변이가 방향성을 가진다 |
| 4 | 평가 비용이 **일정**하다 | `skip_sparse`로 충실도 조절 가능, 논문이 dense가 매크로 레이아웃을 확정한다고 명시 | 다중충실도(multi-fidelity) |

**1번과 2번의 조합이 결정적이다.** objective와 measure가 **둘 다 미분가능**하면 QD는 완전히 다른 알고리즘이 된다. 이게 Differentiable Quality-Diversity(DQD) 계열의 전제이고, 이 연구는 그 전제를 **이미 절반 이상 충족한 상태로 시작한다** — 미분가능 FEM을 만드는 것이 논문 1의 기여였기 때문이다.

---

## 4. 핵심 아이디어 — archive를 탐색하지 말고 조준하라

### 문제: 예산

파트당 11분, 4090 한 대로 하루 130회. 표준 MAP-Elites는 10⁴~10⁶ 평가를 쓴다. **2~4자릿수가 부족하다.** [`qd_extension.md`](qd_extension.md)의 Tier 전략은 이걸 10배 정도 완화하지만, 근본 해결은 아니다.

### 해법: 셀 배정을 사후에서 사전으로 옮긴다

```
표준 MAP-Elites:   변이 → 생성 → 평가 → "어느 셀에 떨어졌나?" 확인   (셀 배정이 사후적)
                   ⇒ 빈 셀을 채우려면 운에 기댄다. 평가 횟수 ≫ 셀 수

제안:              목표 셀 (v*, t*) 지정 → descriptor loss를 guidance에 추가
                   → 그 셀을 겨냥해 생성                            (셀 배정이 사전적)
                   ⇒ 평가 횟수 ≈ O(셀 수)
```

논문 1이 이미 증명한 것이 바로 **"guidance loss를 추가하면 형상이 그 방향으로 움직인다"**이다. compliance에 대해 98/100으로 작동했다. 같은 메커니즘을 descriptor에 적용하면 된다:

```
L_total = L_BCE + L_reach + λ_FEM·L_compliance + λ_D·L_descriptor
                                                  ↑ 신규
L_descriptor = (v(x̂₀) − v*)²  +  (t(x̂₀) − t*)²
               질량비 조준        두께 조준
```

**400셀 × 11분 = 73시간.** 예산 문제가 원리적으로 풀린다. Tier 전략을 얹으면 더 줄어든다.

### 어느 축을 조준할 수 있는가

| Descriptor | 미분가능? | 근거 | 역할 |
|---|---|---|---|
| **질량비 V/V_domain** | ✓ | `Σσ(o)/|D|` — 자명 | **조준** |
| **중앙 부재 두께** | ✓ | Table 2의 per-voxel thickness loss가 **이미 구현돼 있다** | **조준** |
| 등주비 (표면적/부피^⅔) | ✓ | 미분가능, openwork 정도의 연속 대리값 | 조준 (선택) |
| **genus (관통 구멍 수)** | ✗ | 위상 불변량, 이산 | **탐색** |
| 스타일 임베딩 | △ | CLIP은 미분가능하나 축 해석이 약함 | 탐색 |

여기서 논문적으로 흥미로운 구조가 나온다:

> **미분가능한 축은 조준하고, 미분 불가능한 축은 탐색한다.**

2D archive `(질량비 × 두께)`는 **양축 조준 가능** → O(셀 수)로 채운다. genus는 3번째 축 또는 사후 분석으로 두고, 같은 셀 안에서 랜덤 탐색이 담당한다. 하이브리드 QD 알고리즘이 그 자체로 기여가 된다.

### 서사의 평행 구조 — 여기가 가장 강하다

```
논문 1:  물리는 post-hoc latent search보다 in-generation guidance가 낫다.
         (98/100, −63.8% vs 35/100, +0.8%)

논문 2:  다양성도 마찬가지다.
         고정 latent를 QD로 탐색하는 것보다, in-generation descriptor guidance로 조준하는 게 낫다.
```

**같은 논지의 두 번째 축이다.** 논문 1의 실험 구조(Table 3, Fig. 5)를 그대로 재사용해서 diversity 축으로 반복할 수 있고, 두 논문이 서로를 강화한다. 별도 논문으로 내되 시리즈로 읽히는 구성이다.

그리고 이 평행이 우연이 아니다 — 둘 다 **"생성이 끝난 뒤에 건드리면 이미 커밋된 형상에 묶인다"**는 같은 원리다. 논문 1의 표현으로:

> "the post-hoc search only nudges an already-committed sparse latent, it stays anchored to that shape and cannot redistribute mass along the load path."

latent space를 QD로 탐색하는 방법도 정확히 같은 한계를 갖는다. 이게 baseline 대비 우위의 **이론적 근거**다.

---

## 5. 별도 논문 구성안

### 기여 (claim)

1. **생성 prior 위의 미분가능 QD.** objective(compliance)와 measure(질량비·두께) 양쪽 gradient를 flow latent에 역전파해 archive 셀을 직접 조준한다. 평가 예산이 O(#cells)로 떨어진다.
2. **혼합 조준-탐색 QD.** 미분가능 축은 guidance로 조준, 위상 축(genus)은 변이로 탐색하는 하이브리드.
3. **제약 하 QD.** containment ≥ 0.99, BC solidity ≥ 0.99, tet mesh 생존을 hard constraint로 다룬다 (`06번 논문`). 실패 샘플을 폐기하지 않고 학습한다.
4. **실제 CAD 부품 도메인의 QD 벤치마크.** QD 문헌의 표준 도메인(로봇 걸음걸이, arm repertoire)이 아닌, 검증 FEM이 붙은 구조 부품 3종.

### 주요 실험 — baseline이 기여를 만든다

| # | Baseline | 무엇을 증명하나 |
|---|---|---|
| 1 | **Latent-space QD** (생성 완료 후 sparse latent를 MAP-Elites로 탐색) | **가장 중요.** 논문 1의 post-hoc 대응물. §4의 평행 구조를 실증 |
| 2 | **큐레이션 프롬프트 열거** (= 현 논문의 100회 실행) | 동일 예산에서 coverage 비교. 탐색의 가치 |
| 3 | **표준 MAP-Elites** (조준 없이 랜덤 변이) | 조준의 기여를 분리 |
| 4 | **SIMP topology optimization** | 단일 해 → 1개 셀. Fig. 2 주장의 정량화 |
| 5 | **FEM guidance off** | 다양성은 넓되 품질이 낮음 → quality-diversity 동시 달성 주장 |

지표: coverage, QD-score, archive profile, max fitness (논문 1의 Table 3과 직접 비교 가능하게 유지).

Ablation: 조준 축 개수(0/1/2), λ_D 스윕(품질-커버리지 trade-off), 제약 처리 방식(폐기 vs 학습).

### 투고처

- **GECCO / ACM TELO** — QD 알고리즘 기여를 정면으로 평가. `02번`(GECCO), `00번`이 이 커뮤니티
- **NeurIPS / ICLR** — DQD + 생성모델 축을 강조할 경우
- **JMD / EAAI / CAD** — 엔지니어링 응용 강조. `06번`이 EAAI

**GECCO를 1순위로 권한다.** 기여 1·2가 알고리즘 기여이고, QD 커뮤니티는 "expensive evaluation"을 오래된 난제로 인식하고 있어서 O(#cells) 주장이 바로 읽힌다.

### 양방향 기여로 프레이밍하기

| 커뮤니티 | 그들의 난제 | 이 논문이 주는 것 |
|---|---|---|
| **QD** | 평가가 비싸다 (시뮬레이션 수만 회) | 생성 prior + 미분가능 measure로 O(#cells) |
| **3D 생성** | 다양성을 제어·측정할 방법이 없다 | archive라는 명시적 다양성 좌표계 |

한쪽 응용이 아니라 양방향으로 쓰면 심사 저항이 줄어든다.

---

## 6. 반드시 먼저 확인할 선행연구

아래는 내 사전 지식 기준이며 `references/`에 없다. **차별점을 세워야 하므로 착수 전 확인이 필수다.**

| 선행연구 | 확인할 것 | 예상 차별점 |
|---|---|---|
| **DQD / CMA-MEGA** (Fontaine & Nikolaidis, NeurIPS 2021로 기억) | objective·measure 양쪽 gradient를 쓰는 QD의 정식화 | 이들은 **직접 파라미터 공간**. 우리는 **생성 prior의 sampling 궤적**에 개입 |
| **Latent Space Illumination** (Fontaine et al., GECCO 2021 계열) | GAN latent를 QD로 탐색 | 정확히 **baseline 1**. 고정 latent 탐색 = post-hoc의 한계를 그대로 승계 |
| **pyribs** | QD 표준 구현체 | 재구현 대신 채택 여부 결정 |
| 물리 제약 + QD 설계 사례 | 구조/기계 도메인 QD 적용 | 검증 FEM + hard spec 결합 사례 유무 |

특히 두 번째가 결정적이다. **latent space illumination이 이미 있다면, 우리 주장은 "그것보다 in-generation이 낫다"가 되어야 하고 — 이건 논문 1이 이미 같은 구조로 증명한 논지다.** 오히려 유리하다.

---

## 7. 리스크

| 리스크 | 심각도 | 대응 |
|---|---|---|
| **조준이 실패한다** — λ_D를 걸어도 descriptor가 목표값으로 안 감 | **치명** | 논문 1이 compliance·BCE·reach 모두에서 작동을 실증했으므로 사전 확률은 높다. 그래도 **가장 먼저** 파일럿 (§8) |
| 조준이 품질을 희생 — 셀은 채우나 compliance가 나쁨 | 높음 | λ_D 스윕으로 trade-off 곡선 제시. 이 곡선 자체가 결과물 |
| descriptor가 genotype/guidance에 반응 안 함 | 높음 | 기존 140회 실행으로 산포 먼저 확인 (비용 0) |
| genus 계산이 메시 노이즈에 과민 | 중 | mesh `euler_number` 대신 64³ occupancy에서 카운트 |
| 조준 loss가 기존 guidance와 충돌 (BCE/reach와 경쟁) | 중 | 논문 1의 warmup·스케줄 관례 재사용. dense에만 걸고 sparse는 자유 |

---

## 8. 착수 순서

**Step 0과 1은 GPU 비용이 0이거나 극소이며, 논문 성립 여부를 가른다. 이것부터 한다.**

```
Step 0  기존 140회 실행에서 descriptor 산포 확인            비용 0, CPU 수 분
        → 질량비·두께·genus가 퍼져 있나? 뭉쳐 있으면 축 교체
        → 이 산포가 곧 baseline 2(큐레이션 열거)의 coverage

Step 1  조준 파일럿: L_descriptor 추가, 목표 질량비 3~5점    ~1시간 × 5
        → 목표값을 따라 실제로 움직이는가? 논문 성립의 관문
        → 실패 시 전체 기획 재검토

Step 2  archive/descriptor 구현 + Step 0 재인덱싱

Step 3  2D archive 조준 채우기 (질량비 × 두께)              ~73시간
        → 논문의 메인 그림

Step 4  baseline 1(latent-space QD) 구현 — 핵심 비교

Step 5  제약 처리(06번) + 나머지 baseline + ablation
```

Step 1이 관문이다. **여기서 조준이 작동하면 논문의 뼈대가 그 자리에서 선다.** 작동하지 않으면 [`qd_extension.md`](qd_extension.md)의 Tier 기반 표준 MAP-Elites로 후퇴하고, 기여는 응용 쪽으로 재조정한다.
