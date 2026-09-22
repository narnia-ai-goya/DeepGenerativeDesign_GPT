# 의자 사례를 이용한 3D QD 실험안

상태: 2026-09-13 계획. 의자 CAD, conditioning 이미지, 생성·FEA 결과는 아직 만들지 않았다. 아래는 실행 가능한 설정 파일이 아니라 구현·평가 명세다.

## 연구에서의 역할

의자를 주요 시각적 사례로 채택한다. 산업 부품은 구조 성능과 제약 준수를 검증하고, 의자는 사용자가 원하는 의미와 형태 다양성이 최종 3D에 남는지를 검증한다. 심사 중인 기존 물리 가이드 논문과 구분되는 중심 질문은 '다양한 유효 설계 집합의 탐색과 제어'다.

LMTO의 §5.4/Fig. 9와 Appendix C를 출발점으로 삼는다. 원문의 의자 실험은 80³ 요소, 체적비 0.2, E=1, ν=0.3의 설정을 제시한다. 이 수치는 원문 설정이며 현재 bracket 설정이 아니다. [저자 원문](https://structoptlab.github.io/files/2025-Integrating%20large%20models%20with%20topology%20optimization%20for%20conceptual%20design.pdf)

참고 PDF: `/home/goya/SDL/3d_qd/references/LMTO_2025_Liang_3D_topology_optimization.pdf`

## 먼저 만들 하나의 개발 문제

중립적인 의자 설계 공간을 만들고, 그 안에서 표면 장식이 아니라 구조 형태를 바꾸게 한다.

- 고정 사양: 외곽 설계 영역, 좌판 높이·사용 면적, 등받이 영역, 지지 영역, 재료, 하중 패치.
- 반드시 비워 둘 영역: 사람이 앉는 공간과 좌판 위 진입 공간. 외곽 box 안을 모두 채운 구조가 좋은 의자로 판정되지 않게 한다.
- 보존 영역: 좌판의 지정된 하중 전달 패치와 바닥 지지 패치. 좌판 전체를 처음부터 두꺼운 고정 고체로 채워 표현을 지나치게 제한하지 않는다.
- 변경 가능 영역: 좌판 아래 프레임, 다리/기둥 연결, 등받이의 구조와 외형, 팔걸이 부근의 허용 영역.
- 초기 물리 문제: 좌판 분포 하중 하나로 평가기 연결을 확인한다. 등받이 하중과 다중 load case는 별도 검증 후 추가한다.

원논문의 정규화 문제를 재현하는 실험과 실제 길이·재료를 정한 응용 실험을 구분한다. 원문 그림에서 하중·지지 patch의 위치와 방향을 명시적으로 전사하기 전에는 exact reproduction이라고 부르지 않는다.

## 입력 이미지와 스타일

첫 bank는 neutral chair / penguin chair / avocado chair의 3개 brief, 각 6개 view다. 두 번째 단계에서 branch/coral, ribbon, coarse lattice 등으로 넓힐 수 있다.

모든 스타일은 같은 중립 CAD를 참조하고 좌판 위치, 바닥 접촉부, 카메라를 유지한다. 같은 스타일의 6개 view가 다른 의자 6개를 묘사하지 않도록 생성 후 cross-view 구조를 검사한다. 원논문 그림을 출력 목표로 그대로 사용하지 않고 새로운 입력 자산을 만든다.

프롬프트 초안:

> Refer to the supplied neutral chair geometry. Preserve the seat height, usable seat area, support contact regions, camera, and empty sitting space. Redesign the permitted structure as a [penguin-inspired / avocado-inspired] chair. Express the concept through the actual three-dimensional backrest, supports, and openings rather than color or painted decoration. Use a single monochrome material on a white background, with no people or text.

이미지 생성 도구의 정확한 backend가 확인되지 않으면 임의의 GPT 버전명을 사용하지 않는다. 최종 PNG와 prompt·참조 CAD·카메라·해시를 고정해 모든 탐색 방법이 공유한다.

## QD와 시각적 의미를 연결하는 방법

주 비교에서는 산업 부품과 동일한 물리 archive를 사용한다: 재료 체적비 × 사전 정의한 재료 배치 특징. compliance가 품질이고, 실제 좌판/지지/빈 공간의 유지가 feasibility다.

추가로 최종 렌더에서 의미 보존과 시각적 다양성을 측정한다. 프롬프트에 penguin이라고 적혀 있었다는 이유만으로 penguin 셀에 넣지 않는다. 입력 이미지의 의미 점수도 최종 형상의 의미 점수와 분리한다.

물리 archive가 모든 semantic style을 보관한다는 보장은 없다. 따라서 다음을 구분해 보고한다.

1. 물리적으로 서로 다른 chair 대안을 얼마나 확보했는가?
2. 생성된 후보와 최종 elite 각각에서 요청한 개념이 얼마나 남았는가?
3. 구조 성능만 보고 선택할 때 의미 있는 스타일이 사라지는가?
4. 제안한 최종 형상 피드백이 그 손실을 줄이는가?

시각적 특징을 직접 archive 축으로 쓰는 확장은 별도 프로토콜로 둔다. 최종 렌더 기반 분류/embedding을 개발 데이터에서 검증한 뒤에만 사용하고, 기존 물리 archive의 coverage와 같은 수치로 섞지 않는다.

## 비교와 논문 그림

| 비교 | 고정할 조건 | 보고할 결과 |
|---|---|---|
| 생성 prior만 / 물리 가이드 | 같은 CAD·이미지·seed block·최종 verifier | 의미 보존과 feasibility 변화 |
| Sobol / MAP-Elites / Bayesian QD / 제안 방법 | 같은 이미지 bank·archive·비용 예산 | 최종 설계 집합의 품질·다양성·비용 |
| 질량을 맞춘 BESO/LMTO류 | 같은 하중·재료·지지·체적 목표와 verifier | 강성 및 의미 표현의 trade-off |
| 국소 편집 | 바꾸지 않을 영역과 변경 영역 사전 고정 | 등받이 의미 변화, 하부 구조 보존, compliance 변화 |

대표 그림은 동일 카메라·단색 렌더로 만든다. 각 brief의 입력과 여러 질량 구간의 최종 대안을 함께 보여주고, compliance·부피·유효성 수치를 붙인다. 빈 칸과 실패도 표시한다. 국소 편집에서는 '등받이만 의미를 바꾸되 좌판과 하부를 유지'하는 경우를 우선한다.

## 구현 전에 필요한 작업

1. 새 chair CAD domain, keep-out, fixture/load patch, voxel mask와 카메라 정의.
2. 좌판 하중 단일 문제의 force balance, FEA 수렴, 연결성과 빈 공간 보존 확인.
3. 복수 하중 patch/방향을 사용하는 evaluator 확장. 현재 bracket runner의 단일 load direction을 그대로 복사해서 다중 하중을 지원한다고 가정하지 않는다.
4. 좌판과 등받이의 긴 범위를 고려한 FEA 해상도·메모리·시간 측정. bracket의 tet size·material·compliance cutoff·가이드 가중치를 그대로 복사하지 않는다.
5. neutral / penguin / avocado 이미지 세트를 고정한 작은 파일럿. 시작 예산안은 기존 bracket과 같은 방법당 9회, 실제 15개 작업이지만, 1회 preflight 시간 측정 후 동결한다.
6. 의미가 3D에서 유지되고 좌판/지지 조건을 통과하면 본 비교에 포함한다. 스타일이 계속 사라진다면 이미지 수만 늘리지 말고 표현·조건 전달을 먼저 개선한다.

## 전체 계획에 반영

교량 추가 제안까지 반영한 최신 기본안은 8개 test CAD 중 산업 부품 4개 + 의자 문제 2개 + 교량 문제 2개다. 예: bracket 2, caliper 2, chair 2, bridge 2. Motor mount와 link는 후속 OOD 평가 후보로 둔다. 평가 수는 유지하되 새 사례의 실행시간은 별도로 측정한다.

의자 2개는 단순 크기 변환이나 다른 동물 prompt가 아니라 서로 다른 설계 영역/지지 구조의 문제여야 한다. 같은 의자의 여러 스타일은 독립 CAD 표본으로 세지 않는다. 개발용 중립 의자와 test chair는 구분한다.

의자 계산 속도는 아직 측정하지 않았다. 전체 예산표의 13분/회는 임시 환산값이므로 chair preflight 후 수정한다. 이 계획을 작성했다는 이유로 대규모 의자 생성을 자동 시작한 것은 아니다.
