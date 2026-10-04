"""Evaluate paired FEA-weight candidates on the same chair spec and baseline."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import evaluate_chair_calibrated_qd_2026_10_03 as evaluation
from run_chair_round3_fea_weight_sweep_2026_10_04 import OUT, PARENT, WEIGHTS


def main() -> None:
    evaluation.OUT = OUT
    sys.argv = [sys.argv[0], '--round', '1', '--overflow-aperture-max', '.6']
    evaluation.main()
    current = json.loads((PARENT / 'round_01/evaluation.json').read_text())['rows'][0]
    others = json.loads((OUT / 'round_01/evaluation.json').read_text())['rows']
    rows = [dict(current, sp_fea_w=8e-9, weight_label='w1', source='original round 3')]
    rows += [dict(row, sp_fea_w=WEIGHTS[row['id']], weight_label=row['id'],
                  source='paired sweep') for row in others]
    summarized = []
    for row in rows:
        summarized.append({key: row.get(key) for key in
                           ('weight_label', 'sp_fea_w', 'id', 'source', 'cell', 'frozen_cell',
                            'descriptor', 'strict_eligible', 'geometry_reasons',
                            'mass_liters', 'worst_compliance_ratio', 'seat_bc', 'back_bc',
                            'outside_fraction', 'repair_fraction', 'generated_mesh',
                            'aligned_mesh', 'preview', 'fea_035')})
    (OUT / 'summary.json').write_text(json.dumps(summarized, indent=2) + '\n')
    for row in summarized:
        print(row['weight_label'], 'eligible=', row['strict_eligible'],
              'C/ref=', row['worst_compliance_ratio'], 'mass=', row['mass_liters'],
              'cell=', row['cell'], flush=True)


if __name__ == '__main__':
    main()
