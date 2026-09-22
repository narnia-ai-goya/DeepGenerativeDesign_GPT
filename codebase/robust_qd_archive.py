"""Seed-aware provisional and verified archives for uncertain realizations."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math

import numpy as np

from raqd_core import cell_index


def genome_key(genome):
    return json.dumps(genome, sort_keys=True, separators=(",", ":"))


class RobustRealizationArchive:
    def __init__(self, dims, ranges, min_replicates=2, max_replicates=4,
                 min_feasible_rate=2/3, min_cell_agreement=2/3, risk_quantile=.8):
        self.dims = tuple(dims); self.ranges = tuple(tuple(v) for v in ranges)
        self.min_replicates = min_replicates; self.max_replicates = max_replicates
        self.min_feasible_rate = min_feasible_rate
        self.min_cell_agreement = min_cell_agreement
        self.risk_quantile = risk_quantile
        self.groups = defaultdict(list)

    def record(self, result):
        self.groups[genome_key(result["genome"])].append(result)

    def assess(self, key):
        rows = self.groups[key]
        valid = [r for r in rows if r.get("valid")]
        feasible = [r for r in valid if r.get("constraints_satisfied")]
        pairs = [(r, cell_index(r["realized_descriptors"], self.dims, self.ranges))
                 for r in feasible]
        pairs = [(r, c) for r, c in pairs if c is not None]
        cells = [c for _, c in pairs]
        modal_cell, modal_hits = (Counter(cells).most_common(1)[0] if cells else (None, 0))
        feasible_rate = len(feasible) / len(rows) if rows else 0
        agreement = modal_hits / len(feasible) if feasible else 0
        compliant = [r["compliance_J"] for r, cell in pairs if cell == modal_cell]
        robust_c = float(np.quantile(compliant, self.risk_quantile)) if compliant else None
        verified = (len(rows) >= self.min_replicates and feasible_rate >= self.min_feasible_rate
                    and agreement >= self.min_cell_agreement and robust_c is not None)
        return {"genome": rows[0]["genome"], "evaluations": len(rows),
                "valid": len(valid), "feasible": len(feasible),
                "feasible_rate": feasible_rate,
                "modal_cell": list(modal_cell) if modal_cell is not None else None,
                "cell_agreement": agreement, "robust_compliance_J": robust_c,
                "verified": verified,
                "needs_reevaluation": len(rows) < self.max_replicates and not verified}

    def summary(self):
        assessments = [self.assess(key) for key in self.groups]
        elites = {}
        for item in assessments:
            if not item["verified"]:
                continue
            cell = tuple(item["modal_cell"])
            old = elites.get(cell)
            if old is None or item["robust_compliance_J"] < old["robust_compliance_J"]:
                elites[cell] = item
        return {"genomes": len(assessments), "provisional": sum(not a["verified"] for a in assessments),
                "verified_elites": len(elites),
                "verified_coverage": len(elites) / math.prod(self.dims),
                "elites": [{"cell": list(cell), **item} for cell, item in sorted(elites.items())],
                "assessments": assessments}

    def reevaluation_queue(self):
        """Prioritize one-shot or inconsistent genomes until verification/max budget."""
        queue = [self.assess(key) for key in self.groups]
        queue = [q for q in queue if q["needs_reevaluation"]]
        queue.sort(key=lambda q: (
            q["evaluations"] >= self.min_replicates,
            -q["feasible_rate"],
            q["evaluations"],
        ))
        return queue
