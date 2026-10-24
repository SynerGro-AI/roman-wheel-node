"""SciFact loading (allenai/scifact, release `data.tar.gz`, 3.1 MB).

Claim-level labels follow the SciFact convention:
* ``SUPPORT`` / ``CONTRADICT`` if the claim's rationales carry that label (a claim never mixes both in SciFact),
* ``NEI`` (not enough info) if the claim has no evidence at all; its ``cited_doc_ids`` are then topical but
  non-evidential.

The primary task is binary: **supported** (SUPPORT) vs **not supported** (CONTRADICT or NEI). It is the safety-relevant
one, because only supported claims should reach a final answer. The 3-way task is reported as well.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

LABELS3 = ("SUPPORT", "CONTRADICT", "NEI")


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    sentences: Tuple[str, ...]


@dataclass(frozen=True)
class SciFactClaim:
    claim_id: str
    claim: str
    label3: str  # SUPPORT | CONTRADICT | NEI
    cited_doc_ids: Tuple[str, ...]
    rationale: Tuple[Tuple[str, int], ...]  # (doc_id, sentence index) of gold rationale sentences

    @property
    def supported(self) -> bool:
        return self.label3 == "SUPPORT"


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_corpus(data_dir: str) -> Dict[str, Doc]:
    out: Dict[str, Doc] = {}
    with open(os.path.join(data_dir, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[str(d["doc_id"])] = Doc(str(d["doc_id"]), d["title"], tuple(d["abstract"]))
    return out


def sentence_hashes(corpus: Dict[str, Doc]) -> set:
    """sha256 of every corpus sentence (and title): the exact-copy gate's reference set."""
    hs = set()
    for d in corpus.values():
        hs.add(sha(d.title))
        hs.update(sha(s) for s in d.sentences)
    return hs


def load_claims(data_dir: str, split: str) -> List[SciFactClaim]:
    if split == "test":
        raise ValueError("SciFact test labels are hidden; evaluate on dev")
    out = []
    with open(os.path.join(data_dir, f"claims_{split}.jsonl"), encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            labels = {e["label"] for evs in c["evidence"].values() for e in evs}
            if len(labels) > 1:
                raise ValueError(f"claim {c['id']} mixes labels {labels}")
            label3 = labels.pop() if labels else "NEI"
            rat = tuple(
                sorted((str(d), int(s)) for d, evs in c["evidence"].items() for e in evs for s in e["sentences"])
            )
            out.append(SciFactClaim(str(c["id"]), c["claim"], label3, tuple(str(x) for x in c["cited_doc_ids"]), rat))
    return out
