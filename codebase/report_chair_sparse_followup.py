#!/usr/bin/env python3
"""Compare visual-hull-conditioned chair dense and sparse outputs."""
from __future__ import annotations

import report_chair_sparse_artifacts as report


BASE = report.BASE
OUT = BASE / "visual_hull_dense/sparse_followup"
report.OUT = OUT
report.CASES = {
    "image-only dense": BASE / "tuning_open_arm/p50_v35_pw00/dense/mesh_dense.obj",
    "image-only sparse peak80": BASE / "tuning_open_arm/p50_v35_pw00/sparse/mesh.obj",
    "image-only sparse peak10 mc040": BASE / "sparse_artifact_ablation/mc040_peak10/sparse/mesh.obj",
    "hull anchor100 dense": BASE / "visual_hull_dense/anchor100_sc20/dense/mesh_dense.obj",
    "hull anchor100 sparse peak80": OUT / "anchor100_peak80/sparse/mesh.obj",
    "hull anchor100 sparse peak10 mc040": OUT / "anchor100_peak10_mc040/sparse/mesh.obj",
    "hull anchor100 selected clean": OUT / "selected_clean.obj",
    "hull anchor300 sparse peak10 mc040": OUT / "anchor300_peak10_mc040/sparse/mesh.obj",
}


def main() -> None:
    report.main()
    page = OUT / "index.html"
    text = page.read_text()
    text = text.replace("Chair sparse artifact ablation",
                        "Chair dense-to-sparse visual hull followup")
    text = text.replace("의자 sparse 형상 이상: 동일 dense에서 파라미터 비교",
                        "의자: 이미지 기반 3D 조건을 dense→sparse까지 적용")
    text = text.replace(
        "FEA와 후처리 전 메시. 모든 경우 동일한 dense cache, 이미지, BC, seed를 사용합니다.",
        "FEA와 후처리 전 메시. 같은 이미지·BC·seed에서 이미지 기반 3D 조건의 유무와 sparse 설정을 비교합니다.")
    text = text.replace(
        "peak80은 기존 sparse, 나머지는 sparse guidance 최대치·두께 제약·mesh 추출 임계값만 변경했습니다.",
        "anchor100/300은 이미지 세 뷰에서 만든 3D 점유 목표를 dense에 추가한 경우입니다. peak10 mc040은 sparse guidance 최대치 10과 mc_threshold 0.4를 뜻합니다.")
    text = text.replace(
        "팔걸이 누락, 잘못된 등받이와 다리 접합은 dense부터 나타납니다. sparse guidance 상한을 낮춰도 해결되지 않았습니다. mc_threshold 0.4는 윤곽 점수를 조금 높였지만 형상 오류는 남습니다. FEA는 실행하지 않았습니다.",
        "선택 후보는 anchor100 peak10 mc040입니다. selected clean은 좌석 가장자리의 극소 분리 조각 2개(총 208면)만 제거한 파일입니다. 윤곽 점수만으로 팔걸이·접합부의 3D 품질을 판단할 수 없습니다. FEA는 실행하지 않았습니다.")
    page.write_text(text)


if __name__ == "__main__":
    main()
