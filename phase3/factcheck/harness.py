"""Phase 3 harness loop (scaffold): score every claim with every source, gate it, learn, and log.

    for claim in dataset:
        passages = retriever.retrieve(claim.claim)
        scores   = {s.name: s.score(claim.claim, passages) for s in scorers}
        decision = trace.submit(claim.claim, scores, a=exact_copy_gate(passages, corpus_hashes))
        if claim.label is not None:                   # feedback = gold label (online, after the decision)
            live.observe(...); dependence.record(scores, claim.label); threshold.record(...)

Metrics are the same as Phase 2 (accuracy, true-accept, false-block, ECE) plus per-source accuracy,
the learned pairwise dependence between judges, and cost (tokens) per claim. Baselines: each source alone,
majority vote, bayes-hard (naive) and bayes-ess (correlation-aware).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence

from roman_wheel_node import DependenceLearner, LiveWeights, ThresholdLearner, give_feedback
from roman_wheel_node.trace import ClaimTrace

from .interfaces import ClaimItem, Retriever, Scorer, exact_copy_gate


def load_scifact(data_dir: str, split: str = "dev") -> List[ClaimItem]:
    """STUB. SciFact (allenai) claims_{train,dev}.jsonl + corpus.jsonl; tarball is 3.1 MB (see PLAN.md).

    Label rule: a claim is True if any gold evidence doc labels it SUPPORT, False if CONTRADICT, None if no evidence.
    """
    raise NotImplementedError("Phase 3: download SciFact into phase3/data/ first")


def run(items: Iterable[ClaimItem], retriever: Retriever, scorers: Sequence[Scorer], corpus_hashes: set,
        out_dir: str, variant: str = "ess", allow_hedge: bool = False) -> Dict[str, float]:
    """Run the gated loop. Writes the audit log and all ledgers under ``out_dir``. Returns summary metrics."""
    os.makedirs(out_dir, exist_ok=True)
    dep = DependenceLearner(os.path.join(out_dir, "dependence.jsonl"))
    live = LiveWeights(os.path.join(out_dir, "weights.jsonl"))
    trace = ClaimTrace(os.path.join(out_dir, "audit.jsonl"), mode="bayes", variant=variant, dependence=dep,
                       allow_hedge=allow_hedge)
    n = ok = 0
    for it in items:
        passages = retriever.retrieve(it.claim)
        scores = {s.name: s.score(it.claim, passages) for s in scorers}
        d = trace.submit(it.claim, scores, a=exact_copy_gate(passages, corpus_hashes))
        if it.label is None:
            continue
        n += 1; ok += d.admitted == it.label
        give_feedback(scores, it.label, live=live, dependence=dep)
    trace.verify()
    return {"n_labelled": n, "accuracy": ok / n if n else float("nan")}
