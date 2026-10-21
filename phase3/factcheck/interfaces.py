"""Interfaces for the Phase 3 fact-verification harness (SciFact first, FEVER later).

Every evidence source maps (claim, evidence passages) to a score in (0, 1) meaning "P(claim is supported)".
The harness feeds those scores to ``ClaimTrace`` (``mode="bayes", variant="ess"``) with a ``DependenceLearner``,
so correlated judges (e.g. several LLMs) are discounted automatically. The exact-copy gate ``a`` is 1 only if every
quoted evidence passage occurs verbatim (sha256-matched) in the corpus.

Status:
* ``OverlapScorer`` and ``XAIJudge`` are implemented. ``XAIJudge`` uses stdlib urllib; the key comes from the
  XAI_API_KEY environment variable and is never logged.
* ``BM25Retriever`` and ``NLIScorer`` are stubs: they need the corpus and an ONNX model to be downloaded first (see PLAN.md).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class Passage:
    doc_id: str
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ClaimItem:
    claim_id: str
    claim: str
    label: Optional[bool]               # True = SUPPORT, False = CONTRADICT, None = NOT ENOUGH INFO / unknown
    gold_doc_ids: Tuple[str, ...] = ()


class Retriever(Protocol):
    def retrieve(self, claim: str, k: int = 3) -> List[Passage]: ...


class Scorer(Protocol):
    name: str
    def score(self, claim: str, passages: Sequence[Passage]) -> float: ...


def exact_copy_gate(passages: Sequence[Passage], corpus_hashes: set) -> int:
    """a = 1 iff every passage used as evidence is byte-identical to a corpus passage."""
    return int(bool(passages) and all(p.sha256 in corpus_hashes for p in passages))


_TOK = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an the of in on and or to is are was were be by for with as that this from at it its".split())


def _toks(s: str) -> set:
    return {t for t in _TOK.findall(s.lower()) if t not in _STOP}


class OverlapScorer:
    """Lexical support: fraction of claim content words found in the evidence, squashed into (0.05, 0.95).

    Weak and topical (it can't tell support from contradiction), which is why it is only one source among several.
    """
    name = "overlap"

    def score(self, claim: str, passages: Sequence[Passage]) -> float:
        c = _toks(claim)
        if not c or not passages:
            return 0.5
        ev = set().union(*(_toks(p.text) for p in passages))
        return 0.05 + 0.9 * (len(c & ev) / len(c))


class BM25Retriever:
    """STUB. Plan: rank_bm25-style Okapi BM25 over the 5,183 SciFact abstracts (pure Python is fine at this size)."""

    def __init__(self, corpus: Dict[str, str]) -> None:
        self.corpus = corpus

    def retrieve(self, claim: str, k: int = 3) -> List[Passage]:
        raise NotImplementedError("Phase 3: implement BM25 after the corpus is downloaded (see PLAN.md)")


class NLIScorer:
    """STUB. Plan: an ONNX cross-encoder NLI model run with onnxruntime on CPU (no torch needed).

    Score = P(entailment) / (P(entailment) + P(contradiction)) of the best-scoring (claim, sentence) pair, so
    neutral evidence lands near 0.5 rather than 0. Candidate: cross-encoder/nli-deberta-v3-small, qint8 ONNX (172 MB).
    """
    name = "nli"

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir

    def score(self, claim: str, passages: Sequence[Passage]) -> float:
        raise NotImplementedError("Phase 3: download the ONNX model + tokenizer first (see PLAN.md)")


class XAIJudge:
    """LLM judge via the xAI OpenAI-compatible chat API. Reads XAI_API_KEY from the environment and never logs it.

    It returns P(supported) in (0.01, 0.99). Temperature 0 with a fixed prompt, but outputs may still vary between calls,
    so the harness caches every response (keyed by model + prompt sha256) for exact reruns.
    """
    URL = "https://api.x.ai/v1/chat/completions"
    PROMPT = ("You are a careful scientific fact-checker. Using ONLY the evidence, estimate the probability that the "
              "claim is supported (1) rather than contradicted (0). Reply with JSON only: {\"p_supported\": <0..1>}.\n\n"
              "Claim: {claim}\n\nEvidence:\n{evidence}")

    def __init__(self, model: str = "grok-4.20-0309-non-reasoning", name: Optional[str] = None,
                 cache_path: Optional[str] = None, timeout: float = 60.0, max_tokens: int = 20) -> None:
        self.model, self.name = model, name or f"xai:{model}"
        self.cache_path, self.timeout, self.max_tokens = cache_path, timeout, max_tokens
        self._cache: Dict[str, dict] = {}
        if cache_path and os.path.exists(cache_path):
            for line in open(cache_path):
                if line.strip():
                    r = json.loads(line); self._cache[r["key"]] = r
        self.last_usage: Optional[dict] = None

    def _prompt(self, claim: str, passages: Sequence[Passage]) -> str:
        ev = "\n".join(f"- [{p.doc_id}] {p.text}" for p in passages) or "(none)"
        return self.PROMPT.replace("{claim}", claim).replace("{evidence}", ev)

    def score(self, claim: str, passages: Sequence[Passage]) -> float:
        prompt = self._prompt(claim, passages)
        key = hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]["p"]
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("XAI_API_KEY is not set")
        body = json.dumps({"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                           "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(self.URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read())
        text = out["choices"][0]["message"]["content"]
        self.last_usage = out.get("usage")
        m = re.search(r"([01](?:\.\d+)?|\.\d+)", text.split("p_supported", 1)[-1])
        p = min(0.99, max(0.01, float(m.group(1)))) if m else 0.5
        rec = {"key": key, "model": self.model, "p": p, "raw": text, "usage": self.last_usage}
        self._cache[key] = rec
        if self.cache_path:
            with open(self.cache_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return p
