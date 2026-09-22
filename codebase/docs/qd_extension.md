# QD 확장 설계안 — Form Embodies Mechanics에 Quality-Diversity 얹기

**대상 논문**: `references/21210_Form_Embodies_Mechanics_.pdf`
**참고 문헌**: `references/00. Quality-Diversity Optimization.pdf` (Chatzilygeroudis et al., MAP-Elites 계열 개관),
`02. Multi-Objective Quality Diversity Optimization.pdf` (MOME),
`06. Bayesian QD ... mixed continuous, discrete and categorical variables.pdf` (Brevault & Balesdent, 제약·혼합변수 Bayesian QD)

작성일 2026-09-01.

> **후속**: 이 문서는 QD를 기존 논문에 *얹는* 시나리오다. 이후 **별도 논문**으로 방향이 확정되었으므로
> [`qd_paper_concept.md`](qd_paper_concept.md)를 먼저 읽을 것. 예산 전략(§3의 Tier 0/1/2)과
> descriptor 후보(§2.2)는 그대로 유효하지만, 별도 논문에서는 표준 MAP-Elites 대신
> **미분가능 descriptor로 archive 셀을 조준하는** 접근이 중심이며 예산 문제를 더 근본적으로 해결한다.

---

## 1. 왜 QD인가 — 논문에 이미 뚫려 있는 QD 모양의 구멍

QD를 "새 기능"으로 덧붙이자는 제안이 아니다. 현재 논문이 **주장하지만 측정하지 않는 축**이 정확히 QD가 다루는 축이다.

| 논문이 말하는 것 | 현재 근거 | 공백 |
|---|---|---|
| Table 1의 `Style` 열: 우리만 ✓ | 정성 표기 (✓/×/△) | **정량 지표 없음** |
| Fig. 2: SIMP는 "single smooth, branching family" | 그림 한 장 | 다양성 격차가 **측정되지 않음** |
| Discussion: "consistent sign shift rather than a collapse onto a canonical shape" | compliance 분산 | 형상 공간 **커버리지 미측정** |
| "many mechanically valid, stylistically distinct candidates in minutes" | 100개 프롬프트 실행 | 프롬프트는 **사람이 큐레이션한 고정 리스트** |

Table 3은 전부 quality 지표(C, u_max, σ_vM)다. **diversity 지표가 하나도 없다.** 논문의 차별점 절반이 정성 서술로 남아 있다.

구조적으로 정리하면 현재 파이프라인은 이렇다:

- 탐색이 아니라 **열거**다. 큐레이션된 100개 프롬프트를 한 번씩 돌린다. 선택도, 재조합도, 세대도 없다.
- 실패가 **버려진다**. Limitations의 "a small fraction of styles produce members too thin or fragmented to survive tetrahedral verification ... removed by an automatic filtering step" — 이 정보가 루프로 돌아오지 않는다. 제약 위반 샘플은 QD에서는 학습 신호다.
- 커버리지 상한이 **2D 조건화 단계에 묶여 있다**. 논문이 Limitations에서 자인한 첫 번째 항목이다.
- guidance weight가 **도메인당 수동 튜닝**이다("requiring only re-tuned guidance weights"). 이건 자동 탐색이 대신할 수 있는 축이다.

QD는 이 네 가지를 동시에 건드린다. 그리고 필요한 재료가 이미 코드베이스에 다 있다 — 미분가능 FEM, 검증용 tet FEM, 프롬프트 세트, 변형(variant) 실행 엔진.

> **한 문장 요약**: QD는 논문의 "다양성" 주장을 서술에서 **측정 가능한 기여**로 바꾸고, 동시에 큐레이션 병목을 탐색으로 대체한다.

---

## 2. 문제 정식화

### 2.1 Genotype (탐색 변수)

`06번 논문`이 다루는 **혼합 연속·이산·범주형** 세팅과 정확히 일치한다. 이게 참고문헌 선택의 근거다.

| 종류 | 변수 | 근거 |
|---|---|---|
| **범주형** `x_q` | style prompt의 슬롯 값 | `configs/bracket.json`의 `cond_img2img.template`이 이미 `"metal bracket, {form} openwork, {descriptor}, monochrome metal, white background"` — **슬롯 구조가 이미 존재한다** |
| **연속** `x_c` | `fea_w`, `sp_fea_w`, `cw`(reach), `out_w`, `bc_w`, `tw_rmin`, `sp_thick_target`, `vol_target`, `sp_guide_w`, `cfg`/`sp_cfg`, `mc_threshold` | 도메인별 수동 튜닝 대상이던 값들 |
| **이산** `x_d` | `seed`, `dense_steps`/`sparse_steps`, `rmin_radius`, `sp_r_min_voxels`, `n_views` | 이미 config 노브 |

**프롬프트 슬롯 조합이 핵심이다.** 현재 100개 프롬프트는 접두사로 이미 taxonomy를 이룬다 (`de_` 90, `grid_` 4, `bionic_` 3, `organic_` 3). 이걸 (form × descriptor) 곱집합으로 풀면 큐레이션 리스트를 **조합적으로 확장**할 수 있다 — Limitations 첫 항목("coverage is bounded by the 2D conditioning stage")에 대한 직접적 대응이다.

교배 연산은 종류별로 나눈다: 범주형은 슬롯 단위 uniform crossover, 연속은 isotropic Gaussian 또는 SBX, 이산은 정수 격자 위 랜덤워크. (`00번 논문`의 표준 MAP-Elites variation 연산자 구성)

### 2.2 Descriptor (behavior/measure space) — 다양성의 정의

**엔지니어가 실제로 읽는 축**이어야 하고, **싸게 계산**돼야 한다. 후보와 계산 경로:

| Descriptor | 의미 | 계산 | 비용 |
|---|---|---|---|
| **V / V_domain** | 질량 예산 | `trimesh` 부피 / 설계영역 부피 (post 단계가 이미 도메인과 intersect하므로 분모 고정) | 무료 |
| **genus (관통 구멍 수)** | openwork 정도 | 64³ occupancy에서 Betti₁ 카운트 (mesh의 `euler_number`는 작은 handle에 과민하므로 **dense 그리드에서 세는 쪽을 권장**) | 거의 무료 |
| **중앙 부재 두께** | 제조성 | sparse 단계의 `sp_thick` 로직 재사용, 또는 mesh 내부 SDF 최대값 | 저 |
| **하중경로 집중도** | 구조 거동 | per-cell strain energy W_e의 Gini 계수. `fenics_fea_bracket.py`가 이미 `mean_W_e`/`max_W_e`와 sensitivity `.npy`를 저장 | 무료 (FEA 부산물) |
| 스타일 임베딩 2D 투영 | 시각적 다양성 | CLIP/DINO + PCA | 저, 단 **공학적 해석력 약함** |

**권장 시작점: 2D archive `(V/V_domain) × genus`.**

- 두 축 모두 해석 가능하고 논문 서사와 직결된다 (질량-강성 트레이드오프, openwork limitation).
- 2D인 이유는 순수하게 **예산**이다. 20×20 = 400 셀, 평가당 11분이면 셀당 1회만 채워도 73시간이다. 3D 축은 Tier-1 대리 평가가 검증된 뒤에 열어야 한다.
- 3D로 확장할 때 세 번째 축은 두께(제조성)를 권장한다 — 심사자가 묻는 "만들 수 있는가"에 답한다.

### 2.3 Objective와 제약

```
maximize   f(x) = −C(x)                    # 검증 FEM 컴플라이언스 (mJ)
subject to g₁: containment      ≥ 0.99     # 논문 보고값 99.4%
           g₂: BC solidity      ≥ 0.99     # 논문 보고값 99.9%
           g₃: 0 < C < sane_c             # tet 메시 생존 (compliance_of가 이미 구현)
```

`g₃`는 `run_from_image.py:313`의 `compliance_of()`가 그대로다 — 논문의 "automatic filtering step"이 이미 코드에 있다. **차이는 처리 방식이다**: 지금은 버리고, QD에서는 제약 위반으로 surrogate에 학습시킨다. 이게 `06번 논문`이 다루는 constrained QD의 핵심이고, 얇은 부재 실패가 반복 생성되는 걸 막는다.

---

## 3. 지배적 제약: 예산

이 섹션이 설계 전체를 결정한다.

- 논문 기준 **bracket 1개당 ≈11분** (FEM guidance on), geometry-only는 ≈5.5분. RTX 4090 1대.
- 하루 24시간 = **약 130회 평가**.
- 표준 MAP-Elites는 통상 10⁴~10⁶ 평가를 쓴다. **2~4자릿수가 모자란다.**

그래서 순진한 MAP-Elites는 불가능하고, 3단 평가로 간다.

### Tier 0 — 이미 지불한 비용 재활용 (추가 비용 0)

논문 실험이 그대로 QD의 초기 DoE다: bracket 100 + motor_mount 20 + link 20, 게다가 ablation의 on/off/posthoc 3변형까지. `experiments/<dom>/<style>/`에 이미 디스크에 있다.

**이걸 먼저 archive에 인덱싱해서 descriptor 산포도를 그리는 것이 첫 액션이다** (§8).

### Tier 1 — 저비용 대리 평가 (수십 초)

`configs/bracket.json`에 **`"skip_sparse": false`가 이미 있다.** 이걸 켜면 dense 64³ cascade만 돈다:

- in-loop dense FEM이 compliance 대리값을 준다 (정규화 모듈러스 E₀=1이지만 **순위만 필요**하다)
- descriptor(V/V_dom, genus)는 64³ occupancy에서 직접 계산 — sparse 단계가 필요 없다
- 논문 §Method가 명시하듯 dense cascade가 **매크로 레이아웃을 확정**한다. 즉 archive 축(질량·구멍수)은 dense 단계에서 이미 결정된다. **대리 평가가 원리적으로 타당한 이유다.**

거칠게 11분 → 1분 미만. 예산이 10배 이상 늘어난다.

### Tier 2 — 정밀 평가 (11분)

Tier-1에서 유망한 후보만 full sparse + post + tet FEM. archive에 최종 등재되는 elite는 반드시 Tier-2를 통과해야 한다 (논문 보고값이 전부 검증 FEM 기준이므로 일관성 유지).

**Tier1 → Tier2 보정**을 GP로 학습하면 그게 §4의 안 B다.

---

## 4. 알고리즘 3안

### 안 A — MAP-Elites + 3단 평가 **(권장 시작점)**

표준 batch MAP-Elites (`00번 논문`). 변이는 §2.1, 평가는 Tier1 선별 → Tier2 확정.

- 구현 부담이 가장 낮다. `run_ablation.py`의 `extra=[...]` → CLI 인자 패턴을 그대로 genotype 주입 통로로 쓴다.
- 산출물이 바로 논문 자산이다: archive heatmap 그림 + coverage/QD-score 표.
- 위험: descriptor가 genotype에 반응하지 않으면 archive가 안 채워진다 → §8의 파일럿으로 먼저 확인.

### 안 B — Bayesian QD (`06번`)

(C, descriptors, 제약)에 GP surrogate를 얹고 infill criterion으로 다음 평가점을 고른다. 혼합변수 공분산(범주형 프롬프트 처리)이 이 논문의 기여다.

- 06번이 주장하는 절감폭이 **최대 2자릿수**다. §3의 예산 문제에 정면으로 답한다.
- 제약을 GP로 학습하므로 "얇아서 죽는" 영역을 회피하며 탐색한다.
- 위험: 혼합변수 커널 구현 복잡도가 높다. **안 A의 Tier-1을 GP로 대체/보강하는 형태**로 점진 도입하는 게 안전하다.

### 안 C — MOME (`02번`)

셀마다 단일 elite 대신 **local Pareto front**를 유지한다. 목적은 (−C, −mass, −σ_vM_max).

- Table 3이 **이미 다목적 데이터를 보고 중이다** — 자연스러운 확장이고 기여로는 가장 강하다.
- 위험: 셀당 여러 해를 유지하므로 예산이 더 든다. 안 A/B로 예산 효율이 확보된 뒤가 맞다.

### 권장 경로

```
A (baseline archive 확보)  →  B (예산 효율 주장)  →  C (여력 시)
```

A만으로도 논문에 넣을 그림과 표가 나온다. B는 "이걸 어떻게 감당하는가"라는 심사자 질문에 대한 답이다.

---

## 5. 코드베이스 접점

기존 구조를 거의 건드리지 않는다. `run_ablation.py`가 이미 "genotype을 CLI 플래그로 바꿔 STAGES 엔진에 넘기는" 패턴이므로, QD 루프는 그 위층이다.

```
codebase/
├── run_qd.py                 # 신규. run_ablation.py와 같은 층위
├── code/qd/
│   ├── archive.py            # MAP-Elites grid, add/replace, coverage/QD-score
│   ├── descriptors.py        # V/V_dom, genus, thickness, W_e Gini
│   ├── genotype.py           # 인코딩 ↔ CLI 플래그 + 프롬프트 슬롯 조합
│   ├── mutate.py             # 범주형/연속/이산 변이 연산자
│   └── surrogate.py          # (안 B) 혼합변수 GP + 제약 + infill
└── configs/qd_bracket.json   # archive 축·bin·범위, 탐색 변수 범위, 예산
```

재사용하는 것:

| 필요한 것 | 이미 있는 것 |
|---|---|
| genotype → 실행 | `run_ablation.py`의 `VARIANTS` / `extra=[...]` |
| 스테이지 실행 | `run_from_image.STAGES['gen'/'post'/'fea']`, `wd=` 로 출력 경로 지정 |
| 제약 g₃ | `run_from_image.compliance_of()` |
| 재개(resume) | 모든 스테이지가 이미 skip-if-exists |
| 대리 평가 | `stages.mesh.skip_sparse` |
| W_e 통계 | `fenics_fea_bracket.py`의 `mean_W_e`/`max_W_e` + sensitivity `.npy` |
| 검증 지표 | `fea_tet_summary.json` (`compliance`, `u_max`, `vm_max`, `vm_mean`) |

출력은 `experiments/<dom>/qd/<cell_id>/<genome_hash>/`로 두면 기존 `variant_dir` 관례와 어긋나지 않고, archive는 디스크를 인덱싱해서 복원 가능하다 (중단 내성).

---

## 6. 평가 프로토콜

**QD 지표** (`00번 논문`의 표준):

| 지표 | 정의 |
|---|---|
| **Coverage** | 채워진 셀 수 / 전체 셀 수 |
| **QD-score** | Σ_cells f(elite) — 품질과 다양성의 결합 |
| **Max fitness** | 최소 compliance. **논문 Table 3과 직접 비교 가능해야 한다** |
| Archive profile | threshold별 채워진 셀 수 (품질 하한을 올려가며) |

**비교 대상** — 여기가 논문 기여를 만든다:

1. **SIMP topology optimization**: 단일 해이므로 archive의 **1개 셀**만 채운다. Fig. 2의 정성 주장("single family")을 **coverage 1/400 대 N/400**으로 정량화한다.
2. **큐레이션 프롬프트 열거 (= 현재 논문)**: 100회 평가로 몇 셀을 채우는가. QD가 **같은 예산으로 더 많은 셀**을 채우면 탐색의 가치가 증명된다. 이게 가장 중요한 비교다.
3. **랜덤 탐색**: 같은 예산, 변이 없이.
4. **FEM guidance off**: 다양성은 넓지만 품질이 낮음을 보이면 Table 1의 `Mech.`+`Style` 동시 만족 주장이 정량적으로 지지된다.

비교 2가 핵심이다. **동일 평가 예산 하에서** QD가 큐레이션 열거를 커버리지로 이기는 그림 하나면 기여가 선다.

---

## 7. 논문 서사에서의 위치

현재 논문의 논지는 quality 축이다: *"in-generation guidance가 post-hoc보다 강성이 낫다"* (98/100, −63.8%).

QD를 얹으면 diversity 축이 붙는다: *"그리고 그 스타일 다양성을 illuminate하고 측정할 수 있다."*

구체적으로 바뀌는 것:

- **Table 1의 `Style` 열**이 ✓/× 정성 표기에서 **QD-score/coverage 수치**로 바뀐다. 표를 실제로 지탱하는 근거가 생긴다.
- **Fig. 2의 SIMP 비판**이 그림 한 장에서 **측정된 격차**가 된다.
- **Limitations 첫 항목**("coverage is bounded by the 2D conditioning stage")이 한계 서술에서 **해결 시도**로 전환된다 — 프롬프트 슬롯 조합 탐색이 그 답이다.
- **Future work**의 "extend beyond linear-static stiffness to thermal, modal, or manufacturing objectives"가 MOME(안 C)의 목적 벡터로 자연스럽게 이어진다.

투고 형태로는 두 갈래다. 별도 논문(QD가 주인공)으로 가거나, 현 논문의 확장 섹션으로 넣거나. **별도 논문 쪽을 권장한다** — 현 논문은 이미 quality 축으로 서사가 완결돼 있고, QD는 알고리즘 기여(제약·혼합변수·다중충실도)를 따로 주장할 분량이 나온다.

---

## 8. 리스크와 첫 액션

### 최대 리스크: descriptor가 genotype에 반응하지 않을 가능성

프롬프트를 바꿔도 V/V_dom과 genus가 좁은 범위에 뭉쳐 있으면 archive가 안 채워지고, QD 전체가 성립하지 않는다. 이건 **돈을 쓰기 전에** 확인해야 한다.

### Step 0 (추가 GPU 비용 0) — 반드시 먼저

이미 실행된 bracket 100 + motor_mount 20 + link 20 결과에서 descriptor를 계산하고 산포도를 그린다.

- 뭉쳐 있으면 → 축을 바꾼다 (두께, W_e Gini, 스타일 임베딩 순으로 후보 이동)
- 퍼져 있으면 → 그 산포가 **초기 archive**이자 논문의 첫 그림이 된다

`experiments/`를 읽고 descriptor만 계산하면 되므로 CPU 몇 분이다. **이 결과가 나오기 전에 QD 루프를 구현하는 건 순서가 틀렸다.**

### 나머지 리스크

| 리스크 | 대응 |
|---|---|
| Tier-1 대리 평가가 Tier-2와 순위 불일치 | Step 0 데이터로 dense-FEM 값 vs 검증 C의 순위상관 먼저 측정 |
| genus 계산이 메시 노이즈에 과민 | mesh `euler_number` 대신 64³ occupancy에서 카운트 |
| 예산 초과 | 2D archive·bin 수 축소부터. 축 하나 늘리면 비용이 곱으로 는다 |
| 프롬프트 조합이 무의미한 문장 생성 | 슬롯 값을 검증된 어휘로 제한, `cond_ref.negative`는 고정 유지 |
| 혼합변수 GP 구현 부담 (안 B) | 안 A로 baseline 확보 후 착수. A만으로도 논문 성립 |

### 실행 순서

```
Step 0  기존 실행 결과로 descriptor 산포 확인          ← 지금 할 것, 비용 0
Step 1  archive.py + descriptors.py + Step 0 재인덱싱
Step 2  Tier-1 대리 평가 검증 (skip_sparse, 순위상관)
Step 3  안 A 루프 (run_qd.py), 소규모 예산으로 파일럿
Step 4  비교 실험 (§6) — 특히 "큐레이션 열거 vs QD, 동일 예산"
Step 5  안 B (Bayesian QD) 또는 안 C (MOME)
```

---

## 부록: 참고문헌이 각각 답하는 질문

| 문헌 | 이 설계에서 답하는 질문 |
|---|---|
| `00. Quality-Diversity Optimization` | MAP-Elites 골격, 변이 연산자, coverage/QD-score 지표 정의 → §2, §4-A, §6 |
| `02. Multi-Objective QD (MOME)` | Table 3의 다목적 지표를 셀별 Pareto front로 → §4-C |
| `06. Bayesian QD (mixed vars, constrained)` | **평가당 11분** 예산 문제 + 범주형 프롬프트 + 제약 g₁~g₃ → §3, §4-B |

06번이 이 설계에서 가장 중요하다. 예산이 지배적 제약이고(§3), genotype이 혼합변수이며(§2.1), 제약 위반 샘플이 실제로 발생하기(논문 Limitations) 때문에 — 세 조건이 그 논문의 세팅과 그대로 일치한다.
