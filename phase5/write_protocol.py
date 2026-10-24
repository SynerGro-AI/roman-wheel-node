"""Phase 5: freeze the evaluation protocol BEFORE any model is trained or the held-out half is scored.

Writes results/phase5/protocol.json (with its own sha256). eval_pair_checker.py refuses to run if the file changes.
The numeric thresholds are fixed later by the 'prereg' step, from A_dev only, using the rule below.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairs_common as C  # noqa: E402

P = {
    "format": "rwn-phase5-protocol",
    "version": 1,
    "written": "2026-10-22 (America/Edmonton)",
    "split_manifest_sha256": C.load_manifest()["sha256"],
    "frozen_policy": "rwn-policy-4.0.0 254804f8ac9b500ef1951879fb64328b0f54303b8bef20e38b1684545f876eb1",
    "checker": {
        "base_model": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "input": "(evidence units joined by a space, claim), evidence truncated first",
        "train_on": "A_train only",
        "targets": "true claim -> entailment, fake claim -> contradiction (3-way head kept)",
        "score": "P(entailment), softmax over 3 classes, fp32",
    },
    "veto": "accept-only: ACCEPT -> NOT_SURE iff checker P(support) < tau or the score is missing; BLOCK and NOT_SURE "
    "never change, so the checker can only remove accepts",
    "tau_rule": "for retention r, tau_r = ascending-sorted P(support) of ALL true A_dev claims at index floor((1-r)*n)",
    "tau_sequence": [
        {"name": "r0.80", "retention": 0.80},
        {"name": "r0.90", "retention": 0.90},
        {"name": "r0.95", "retention": 0.95},
    ],
    "primary": {
        "base": "panel_4b",
        "setting": "r0.90",
        "why": "panel + frozen Phase 4b structural veto is the current best frozen rule; r0.90 is the middle "
        "of the declared sequence",
    },
    "secondary_bases": ["panel", "local"],
    "heldout": "B_test COVID-Fact items the panel judged (the 1,856 of Phase 4b); scored exactly once",
    "certification": {
        "alphas": [0.01, 0.02, 0.05],
        "delta": 0.05,
        "method": "Learn-Then-Test fixed-sequence over tau_sequence in order (strict -> loose), each "
        "hypothesis 'false-accept share > alpha' rejected iff the one-sided Clopper-Pearson "
        "upper bound at delta is <= alpha; stop at the first failure",
        "feasibility_note": "zero-error accepts needed: 299 (1%), 149 (2%), 59 (5%). The Phase 4b rule "
        "accepted 289 of the B items, so 1% cannot be certified on B by any veto",
    },
    "no_regression": {
        "sets": list(C.NOREG_SETS),
        "base": "panel_4b",
        "setting": "primary",
        "pass_if": "precision after >= precision before AND correct accepts lost <= max(1, 5% of "
        "correct accepts before)",
    },
    "success": "primary setting certified at 5% on B AND every no-regression set passes",
    "note": C.NOTE,
}
P["sha256"] = C.canon_sha(P)

if __name__ == "__main__":
    if os.path.exists(C.PROTOCOL):
        old = json.load(open(C.PROTOCOL))
        sys.exit(0 if old["sha256"] == P["sha256"] else "protocol.json exists and differs: refusing")
    json.dump(P, open(C.PROTOCOL, "w"), indent=1, sort_keys=True)
    print("protocol", P["sha256"])
