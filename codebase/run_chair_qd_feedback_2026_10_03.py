"""Two post-hoc image corrections after the frozen 4+4 chair comparison."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import shutil

from build_chair_image_qd_2026_10_03 import normalized, metrics
from chair_qd_long_protocol_2026_10_03 import OUT as MAIN
from prepare_chair_text_reasoned_pilot_2026_10_03 import front_aperture
import run_chair_qd_long_3d_2026_10_03 as runner

OUT = MAIN / 'feedback_round_02'
CASES = [
    {'id': 'feedback_narrow_open', 'method': 'feedback', 'target_cell': [2, 0],
     'revision_from': 'qd_03', 'reason': '3D arm aperture 0.718 outside frozen range [0,.45]'},
    {'id': 'feedback_wide_open', 'method': 'feedback', 'target_cell': [2, 2],
     'revision_from': 'qd_04', 'reason': '3D outside envelope/repair 13.6%, image BC gate false'},
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    original = json.loads((MAIN / 'result.json').read_text())
    pilot = json.loads((MAIN.parent / 'protected_parts_qd/result.json').read_text())
    occupied = {tuple(map(int, key.split(','))) for key in pilot['archive']}
    for row in original['rows']:
        if row['method'] == 'control' and row['strict_eligible']:
            occupied.add(tuple(row['cell']))
    (OUT / 'initial_archive.json').write_text(json.dumps({'occupied_cells': sorted([list(x) for x in occupied]),
        'provenance': 'pilot archive plus strict-eligible control candidates from frozen 4+4 comparison'}, indent=2) + '\n')
    (OUT / 'protocol.json').write_text(json.dumps({
        'status': 'post-hoc exploratory feedback round after observed 4+4 failures',
        'comparison': 'no matched control; cannot be compared as prospective advantage',
        'cases': CASES,
        'limitations': ['post-hoc intervention; no matched control',
                        'unseeded image model; FEA is 35mm repaired voxel proxy']}, indent=2) + '\n')
    _, reference = normalized(runner.GATE_REF)
    image_rows = []
    for item in CASES:
        source = MAIN / (item['id'] + '.png')
        shutil.copy2(source, OUT / source.name)
        _, mask = normalized(source)
        m = metrics(mask, reference)
        m['front_arm_aperture_fraction'] = front_aperture(mask)
        image_rows.append({'id': item['id'], 'method': 'feedback', 'image': str(OUT / source.name),
                           'metrics': m, 'projected_interface_gate': m['projected_interface_retention'] >= .9,
                           'projected_back_load_gate': m['projected_back_load_retention'] >= .8})
    (OUT / 'image_metrics.json').write_text(json.dumps(image_rows, indent=2) + '\n')
    runner.OUT = OUT
    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(lambda pair: runner.run(*pair), zip(CASES, (0, 1))))
    (OUT / 'generation_results.json').write_text(json.dumps(result, indent=2) + '\n')
    if any(row['exit_code'] for row in result):
        raise SystemExit('3D generation failed')


if __name__ == '__main__':
    main()
