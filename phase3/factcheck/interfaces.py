"""Interfaces for the Phase 3 fact-verification harness (SciFact first, FEVER later).

Every evidence source maps (claim, evidence passages) to a score in (0, 1) meaning "P(claim is supported)".
The harness feeds those scores to ``ClaimTrace`` (``mode="bayes", variant="ess"``) with a ``DependenceLearner``,
so correlated judges (e.g. several LLMs) are discounted automatically. The exact-copy gate ``a`` is 1 only if every
quoted evidence passage occurs verbatim (sha256-matched) in the corpus.

Status:
* ``OverlapScorer``, ``BM25Retriever``, ``NLIScorer`` (ONNX, CPU) and ``XAIJudge`` are implemented. ``XAIJudge`` uses
  stdlib urllib; the key comes from the XAI_API_KEY environment variable and is never logged or cached.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
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
    label: Optional[bool]  # True = SUPPORT, False = CONTRADICT, None = NOT ENOUGH INFO / unknown
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
    """Okapi BM25 (k1=0.9, b=0.4, the Anserini defaults) over title + abstract, in pure Python.

    ``retrieve`` returns the top-k documents' ids. The harness turns them into per-sentence passages.
    """

    def __init__(self, docs: Dict[str, str], k1: float = 0.9, b: float = 0.4) -> None:
        self.k1, self.b = k1, b
        self.ids = sorted(docs)
        self.tf: List[Dict[str, int]] = []
        df: Dict[str, int] = {}
        for i in self.ids:
            tf: Dict[str, int] = {}
            for t in _TOK.findall(docs[i].lower()):
                if t not in _STOP:
                    tf[t] = tf.get(t, 0) + 1
            self.tf.append(tf)
            for t in tf:
                df[t] = df.get(t, 0) + 1
        self.len = [sum(tf.values()) for tf in self.tf]
        self.avg = sum(self.len) / len(self.len)
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> List[Tuple[float, str]]:
        q = [t for t in _TOK.findall(query.lower()) if t not in _STOP and t in self.idf]
        out = []
        for j, tf in enumerate(self.tf):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.len[j] / self.avg)
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                out.append((s, self.ids[j]))
        out.sort(key=lambda x: (-x[0], x[1]))
        return out

    def retrieve(self, claim: str, k: int = 3) -> List[str]:
        return [d for _, d in self.scores(claim)[:k]]


class NLIScorer:
    """ONNX cross-encoder NLI run on CPU with onnxruntime (no torch). Default: cross-encoder/nli-deberta-v3-small, int8.

    ``probs(claim, passages)`` returns one (P(contradiction), P(entailment), P(neutral)) per passage, in the order of
    the model config's ``id2label``. The premise is the evidence sentence and the hypothesis is the claim. ``score`` =
    max over passages of P(entailment) (a raw score; the harness calibrates it on the training split).
    """

    name = "nli"

    def __init__(
        self, model_dir: str, onnx_file: str = "model_qint8_avx512_vnni.onnx", max_len: int = 256, threads: int = 0
    ) -> None:
        import numpy as np  # local import: only the NLI scorer needs numpy/onnxruntime
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.np = np
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.labels = [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")  # don't starve torch in the same process
        self.sess = ort.InferenceSession(os.path.join(model_dir, onnx_file), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def probs(self, claim: str, passages: Sequence[str], batch: int = 16) -> List[Dict[str, float]]:
        np = self.np
        out: List[Dict[str, float]] = []
        for i in range(0, len(passages), batch):
            encs = self.tok.encode_batch([(p, claim) for p in passages[i : i + batch]])
            L = max(len(e.ids) for e in encs)
            ids = np.zeros((len(encs), L), dtype=np.int64)
            mask = np.zeros_like(ids)
            tt = np.zeros_like(ids)
            for j, e in enumerate(encs):
                n = len(e.ids)
                ids[j, :n] = e.ids
                mask[j, :n] = 1
                tt[j, :n] = e.type_ids
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = tt
            logits = self.sess.run(None, feed)[0].astype(np.float64)
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            out.extend({lab: float(round(row[k], 6)) for k, lab in enumerate(self.labels)} for row in p)
        return out

    def score(self, claim: str, passages: Sequence[Passage]) -> float:
        ps = self.probs(claim, [p.text for p in passages])
        return max((d["entailment"] for d in ps), default=0.5)


# USD per 1M tokens (input, cached input, output), from docs.x.ai/docs/models (prompts < 200k tokens), checked
# 2026-10-21.
XAI_PRICES = {
    "grok-4.20-0309-non-reasoning": (1.25, 0.20, 2.50),
    "grok-4.20-0309-reasoning": (1.25, 0.20, 2.50),
    "grok-4.3": (1.25, 0.20, 2.50),
    "grok-build-0.1": (1.00, 0.20, 2.00),
    "grok-4.7": (2.00, 0.50, 6.00),
    "grok-4.6": (2.00, 0.50, 6.00),
    "grok-4.5": (2.00, 0.30, 6.00),
    "grok-4.20-multi-agent-0309": (1.25, 0.20, 2.50),
}


def usage_cost_usd(model: str, usage: Optional[dict]) -> float:
    """Cost of one call from its ``usage`` block and the published price list (reasoning tokens bill as output)."""
    if not usage:
        return 0.0
    pin, pcache, pout = XAI_PRICES[model]
    prompt = usage.get("prompt_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    out = usage.get("completion_tokens", 0)
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    if reasoning and out < reasoning + 1:  # some APIs exclude reasoning from completion_tokens
        out += reasoning
    return ((prompt - cached) * pin + cached * pcache + out * pout) / 1e6


class BillingError(RuntimeError):
    """xAI refused the call for billing, credit or auth reasons: stop the whole run cleanly."""


class BudgetExceeded(RuntimeError):
    pass


class SpendLedger:
    """Append-only record of API spend with a hard cap, safe across threads *and* processes.

    Before every call it re-reads lines appended by other processes, so concurrent runs share one cap.
    """

    def __init__(self, path: str, cap_usd: float) -> None:
        import threading

        self.path, self.cap = path, cap_usd
        self.lock = threading.Lock()
        self.total, self.calls, self._off = 0.0, 0, 0
        self._sync()

    def _sync(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            f.seek(self._off)
            while True:
                line = f.readline()
                if not line or not line.endswith("\n"):
                    break
                self._off = f.tell()
                if line.strip():
                    self.total += json.loads(line)["usd"]
                    self.calls += 1

    def check(self) -> None:
        with self.lock:
            self._sync()
            if self.total >= self.cap:
                raise BudgetExceeded(f"spend ${self.total:.4f} reached the cap ${self.cap:.2f}")

    def add(self, model: str, usd: float, ticks: Optional[int]) -> None:
        with self.lock:
            with open(self.path, "a") as f:
                f.write(json.dumps({"model": model, "usd": usd, "ticks": ticks}) + "\n")
            self._sync()


JUDGE_PROMPTS = {
    # A: direct 3-way fact-checker that quotes its decisive sentence (for the exact-copy gate)
    "A": (
        "You are a careful scientific fact-checker. Below is a claim and numbered sentences from research abstracts. "
        "Decide whether the evidence SUPPORTS the claim, CONTRADICTS it, or gives NOT ENOUGH INFO. Use only the "
        "evidence. Be strict about the direction of effect, magnitude, population and causality.\n"
        'Reply with JSON only: {"support": p, "contradict": p, "nei": p, "quote": "..."} where the three '
        "probabilities sum to 1 and quote is the single most decisive evidence sentence copied exactly, character for "
        "character (empty string if NOT ENOUGH INFO).\n\nClaim: {claim}\n\nEvidence:\n{evidence}"
    ),
    # B: skeptical reviewer, different vocabulary and decision rule
    "B": (
        "Act as a skeptical peer reviewer. A claim counts as ESTABLISHED only if the abstracts below state it, or "
        "something that directly entails it, for the same population and outcome; REFUTED if they state the opposite "
        "or a clearly incompatible finding; otherwise UNRESOLVED. Ignore your own background knowledge.\n\n"
        "Abstracts:\n{evidence}\n\nClaim under review: {claim}\n\n"
        'Answer with JSON only: {"verdict": "ESTABLISHED"|"REFUTED"|"UNRESOLVED", "confidence": c} with c in [0,1].'
    ),
    # D: SciFact-style annotation guidelines (written from the SciFact paper's definitions; no examples from any split)
    "D": (
        "You are annotating claims against scientific abstracts, following these guidelines.\n"
        "SUPPORTS: some sentence(s) of an abstract report a finding that, read as a domain scientist would, implies "
        "the claim is true (claims are often paraphrased or slightly generalized from the finding; that still "
        "counts).\n"
        "REFUTES: a reported finding implies the claim is false (including the opposite direction of effect or a "
        "contradicting number).\n"
        "NOINFO: the abstracts are on the topic but no reported finding decides the claim either way. Do not use "
        "outside knowledge.\n\nClaim: {claim}\n\nAbstracts:\n{evidence}\n\n"
        'Return JSON only: {"support": p, "contradict": p, "nei": p, "quote": "..."} (probabilities for '
        'SUPPORTS, REFUTES, NOINFO summing to 1; quote = the decisive sentence copied exactly, or "" for NOINFO).'
    ),
    # E: two-sided: argue both sides, then decide (support-vs-refute)
    "E": (
        "Claim: {claim}\n\nEvidence (numbered sentences from research abstracts):\n{evidence}\n\n"
        "Step 1: in one sentence, give the strongest evidence-based case that the claim is SUPPORTED. "
        "Step 2: in one sentence, the strongest case that it is REFUTED or NOT DECIDED by this evidence. "
        "Step 3: weigh them using only the evidence. End with a final line of JSON only: "
        '{"support": p, "contradict": p, "nei": p} (probabilities summing to 1).'
    ),
    # F: adversarial devil's advocate: starts from "not established"
    "F": (
        "You are a devil's advocate auditing an AI system that tends to accept claims too easily. Start from the "
        "assumption that the claim below is NOT established by the abstracts, and look for any mismatch: different "
        "population, species, outcome, dose, direction, effect size, causal vs correlational wording, or a finding "
        "that is merely related. Only if the abstracts clearly establish the claim despite your audit, say so. If "
        "they clearly establish the opposite, say that.\n\nClaim: {claim}\n\nAbstracts:\n{evidence}\n\n"
        'Output JSON only: {"support": p, "contradict": p, "nei": p} (probabilities summing to 1).'
    ),
    # H: primary-abstract / NEI-focused (designed against spurious support from merely related abstracts)
    "H": (
        "Several abstracts were retrieved for the claim below; most may be only loosely related. Step 1: find the ONE "
        "abstract whose own reported results are about the claim's specific subject (same intervention or entity, same "
        "outcome, same population or system). If no abstract is, the answer is NOINFO. Step 2: using only that "
        "abstract's results, decide whether they establish the claim (SUPPORTS), establish its negation or an "
        "incompatible result (REFUTES), or neither (NOINFO). Related findings about other entities, or background "
        "statements, do not count.\n\nClaim: {claim}\n\nAbstracts:\n{evidence}\n\n"
        'Return JSON only: {"doc": "<doc id or empty>", "support": p, "contradict": p, "nei": p} '
        "with probabilities summing to 1."
    ),
    # V: strict proof verifier (Phase 3c). Quote first, then a no-extrapolation entailment test.
    "V": (
        "You are a strict proof checker. Your job is to decide whether the evidence below PROVES the claim, with no "
        "extrapolation. First copy, character for character, the single evidence sentence that comes closest to "
        "proving the claim (a contiguous span of one sentence; do not paraphrase, do not join sentences). Then test "
        "the claim against that quote. It is PROVEN only if the quoted finding directly states the claim (or "
        "something that logically entails it) for the same population/species/system, the same intervention or "
        "entity, the same outcome or measure, and the same direction. Answer NOT PROVEN if the quote is hedged or "
        "speculative "
        '("may", "suggests", "could", hypotheses, background or introduction statements), supports only part '
        "of the claim, concerns a different population (e.g. mice or cells vs humans), a different outcome or measure, "
        "a different dose or condition, or turns an association into causation. Answer DISPROVEN only if the quote "
        "directly states the opposite or an incompatible result. Use no outside knowledge.\n\n"
        "Claim: {claim}\n\nEvidence:\n{evidence}\n\n"
        'Return JSON only, in this key order: {"quote": "...", "support": p, "contradict": p, "nei": p} where '
        "support = P(PROVEN), contradict = P(DISPROVEN), nei = P(NOT PROVEN), summing to 1."
    ),
    # W: component-wise verifier (Phase 3c): decompose the claim, check each part against one quote, then decide.
    "W": (
        "Claim: {claim}\n\nEvidence (numbered sentences from research abstracts):\n{evidence}\n\n"
        "Verify the claim like an auditor. Step 1: copy exactly (character for character) the one evidence sentence "
        "most relevant to the claim. Step 2: split the claim into its components (subject/population, intervention or "
        "entity, outcome, direction or magnitude, causal vs associative wording) and, for each, say in a few words "
        "whether the quote states it explicitly. Step 3: the claim is SUPPORTED only if every component is explicitly "
        "stated by the quote, as a finding (not a hypothesis, aim, background claim or hedge). It is REFUTED only if "
        "the quote states the opposite. Otherwise it is NOT SUPPORTED. End with a final line of JSON only: "
        '{"quote": "...", "support": p, "contradict": p, "nei": p} (probabilities summing to 1).'
    ),
    # C: plain question-answering framing
    "C": (
        "Question: based only on these study abstracts, is the following statement true?\n"
        "Statement: {claim}\n\n{evidence}\n\n"
        'Give JSON only: {"p_true": x, "p_false": y} where x is the probability that the abstracts show the statement '
        "is true, y that they show it is false, and 1-x-y that they do not say."
    ),
}


def _num(text: str, key: str) -> Optional[float]:
    m = re.search(r'"?' + re.escape(key) + r'"?\s*:\s*"?([0-9]*\.?[0-9]+(?:[eE]-?\d+)?)', text)
    return float(m.group(1)) if m else None


def parse_judge(style: str, text: str) -> Tuple[Dict[str, float], str]:
    """Map a judge reply to (P(SUPPORT), P(CONTRADICT), P(NEI)) and its quote. Unparseable -> uniform (counted)."""
    quote = ""
    if style in ("A", "D", "E", "F", "G", "H", "V", "W"):
        if style in ("E", "W"):  # use the final JSON line
            i = text.rfind("{")
            text = text[i:] if i >= 0 else text
        ps = {k: _num(text, k) for k in ("support", "contradict", "nei")}
        m = re.search(r'"quote"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m:
            try:
                quote = json.loads('"' + m.group(1) + '"')
            except ValueError:
                quote = m.group(1)
        vals = {"SUPPORT": ps["support"], "CONTRADICT": ps["contradict"], "NEI": ps["nei"]}
    elif style == "B":
        v = re.search(r"ESTABLISHED|REFUTED|UNRESOLVED", text)
        c = _num(text, "confidence")
        if not v or c is None:
            vals = {"SUPPORT": None, "CONTRADICT": None, "NEI": None}
        else:
            lab = {"ESTABLISHED": "SUPPORT", "REFUTED": "CONTRADICT", "UNRESOLVED": "NEI"}[v.group(0)]
            c = min(1.0, max(0.0, c))
            vals = {k: (c if k == lab else (1 - c) / 2) for k in ("SUPPORT", "CONTRADICT", "NEI")}
    else:
        x, y = _num(text, "p_true"), _num(text, "p_false")
        vals = {"SUPPORT": x, "CONTRADICT": y, "NEI": None if x is None or y is None else max(0.0, 1 - x - y)}
    if any(v is None for v in vals.values()):
        return {"SUPPORT": 1 / 3, "CONTRADICT": 1 / 3, "NEI": 1 / 3, "_unparsed": 1.0}, quote
    tot = sum(max(0.0, v) for v in vals.values()) or 1.0
    return {k: max(0.0, v) / tot for k, v in vals.items()}, quote


class XAIJudge:
    """LLM judge via the xAI OpenAI-compatible chat API. Reads XAI_API_KEY from the environment; never logs or caches
    it.

    ``judge(claim, evidence_text)`` returns ``{"SUPPORT", "CONTRADICT", "NEI"}`` probabilities and an optional quote.
    Temperature 0 with a fixed prompt, but outputs can still vary between calls, so every response is cached by
    sha256(model + prompt) in an append-only JSONL file. Reruns read the cache and spend nothing.
    ``score`` (the Scorer protocol) is P(SUPPORT).
    """

    URL = "https://api.x.ai/v1/chat/completions"
    PROMPT = JUDGE_PROMPTS["A"]

    def __init__(
        self,
        model: str = "grok-4.20-0309-non-reasoning",
        name: Optional[str] = None,
        cache_path: Optional[str] = None,
        timeout: float = 120.0,
        max_tokens: int = 300,
        style: str = "A",
        spend: Optional[SpendLedger] = None,
        temperature: float = 0.0,
        samples: int = 1,
    ) -> None:
        import threading

        self.model, self.name, self.style = model, name or f"xai:{model}", style
        self.cache_path, self.timeout, self.max_tokens, self.spend = cache_path, timeout, max_tokens, spend
        self.temperature, self.samples = float(temperature), int(samples)
        self._lock = threading.Lock()
        self._cache: Dict[str, dict] = {}
        if cache_path and os.path.exists(cache_path):
            for line in open(cache_path):
                if line.strip():
                    r = json.loads(line)
                    self._cache[r["key"]] = r
        self.last_usage: Optional[dict] = None

    def _prompt(self, claim: str, passages: Sequence[Passage]) -> str:
        ev = "\n".join(f"- [{p.doc_id}] {p.text}" for p in passages) or "(none)"
        return self._render(claim, ev)

    def _render(self, claim: str, evidence: str) -> str:
        return JUDGE_PROMPTS[self.style].replace("{claim}", claim).replace("{evidence}", evidence)

    def _key(self, prompt: str, sample: int = 0) -> str:
        if self.temperature == 0 and sample == 0:  # (keeps the original cache keys valid)
            return hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()
        return hashlib.sha256(f"{self.model}\nT={self.temperature}\n#{sample}\n{prompt}".encode()).hexdigest()

    def cached(self, claim: str, evidence: str) -> bool:
        p = self._render(claim, evidence)
        return all(self._key(p, i) in self._cache for i in range(self.samples))

    def raw(self, prompt: str, sample: int = 0) -> dict:
        key = self._key(prompt, sample)
        if key in self._cache:
            return self._cache[key]
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("XAI_API_KEY is not set")
        if self.spend is not None:
            self.spend.check()
        body = json.dumps(
            {
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        req = urllib.request.Request(
            self.URL,
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        import time as _t

        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:  # billing / auth errors stop immediately (never echo headers or key)
                body = ""
                try:
                    body = e.read().decode("utf-8", "replace")[:300]
                except Exception:
                    pass
                if e.code in (401, 402, 403) or re.search(
                    r"credit|billing|quota|insufficient|spending limit", body, re.I
                ):
                    raise BillingError(f"xAI HTTP {e.code}: {body}") from None
                if attempt == 4:
                    raise
                _t.sleep(2**attempt)
            except Exception:  # rate limit / transient error: back off, never echo headers
                if attempt == 4:
                    raise
                _t.sleep(2**attempt)
        usage = out.get("usage") or {}
        usd = usage_cost_usd(self.model, usage)
        if self.spend is not None:
            self.spend.add(self.model, usd, usage.get("cost_in_usd_ticks"))
        rec = {
            "key": key,
            "model": self.model,
            "style": self.style,
            "raw": out["choices"][0]["message"]["content"],
            "usage": usage,
            "usd": usd,
        }
        with self._lock:
            self._cache[key] = rec
            if self.cache_path:
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        self.last_usage = usage
        return rec

    def judge(self, claim: str, evidence: str) -> Tuple[Dict[str, float], str]:
        """One call, or the mean of ``samples`` temperature-sampled calls (self-consistency)."""
        p = self._render(claim, evidence)
        outs = [parse_judge(self.style, self.raw(p, i)["raw"]) for i in range(self.samples)]
        if len(outs) == 1:
            return outs[0]
        ok = [o for o, _ in outs if "_unparsed" not in o] or [outs[0][0]]
        return {k: sum(o[k] for o in ok) / len(ok) for k in ("SUPPORT", "CONTRADICT", "NEI")}, outs[0][1]

    def score(self, claim: str, passages: Sequence[Passage]) -> float:
        prompt = self._prompt(claim, passages)
        key = self._key(prompt)
        if key in self._cache and "p" in self._cache[key]:  # legacy cache records
            return self._cache[key]["p"]
        p = parse_judge(self.style, self.raw(prompt)["raw"])[0]["SUPPORT"]
        return min(0.99, max(0.01, p))
