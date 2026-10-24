"""Evidence sources for the deployable node (Phase 4). Free and local by default; paid judges are opt-in.

* :func:`exact_gate`      - the exact-copy gate: a = 1 iff there is evidence and every evidence unit is byte-identical
  (sha256) to a unit of the trusted corpus. No trusted corpus -> a = 0 (BLOCK): unverifiable evidence never enters.
* :class:`BM25`           - pure-Python retrieval over a trusted corpus (used when the agent gives no evidence).
* :class:`LocalSources`   - the Phase 3 local sources (``device="cuda"`` runs nli2 on torch CUDA and the ONNX model on
  onnxruntime-gpu's CUDAExecutionProvider when available): ``overlap`` (always), ``nli`` (DeBERTa-v3-small int8 ONNX;
  needs onnxruntime + tokenizers and the model directory) and ``nli2`` (DeBERTa-v3-base mnli-fever-anli; needs torch +
  transformers). A source that is not available is simply absent, so a profile that needs it fails closed to NOT_SURE.
* :class:`XAIPanel`       - the frozen Phase 3d judge panel, OFF unless constructed with ``enable=True`` and a spend
  cap. It never reads ``XAI_API_KEY`` before that, and never prints or logs it.
All raw outputs are ``{source: {"S": P(support)[, "C": P(contradiction)]}}``: max over evidence sentences (as in Phase
3).
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = ["sha256", "split_sentences", "exact_gate", "BM25", "LocalSources", "XAIPanel", "OverlapScorer"]

_TOK = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an the of in on and or to is are was were be by for with as that this from at it its".split())
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in SPLIT.split(text.strip()) if s.strip()]


def exact_gate(evidence: Sequence[str], trusted_hashes: Optional[Iterable[str]]) -> int:
    """a = 1 iff evidence is non-empty and every unit's sha256 is in ``trusted_hashes`` (None -> 0)."""
    if not evidence or trusted_hashes is None:
        return 0
    hs = trusted_hashes if isinstance(trusted_hashes, (set, frozenset)) else set(trusted_hashes)
    return int(all(isinstance(e, str) and sha256(e) in hs for e in evidence))


def _toks(s: str) -> set:
    return {t for t in _TOK.findall(s.lower()) if t not in _STOP}


class OverlapScorer:
    """Fraction of claim content words found in the evidence, squashed into (0.05, 0.95) (Phase 3 ``overlap``)."""

    def score(self, claim: str, sents: Sequence[str]) -> float:
        c = _toks(claim)
        if not c or not sents:
            return 0.5
        ev = set().union(*(_toks(p) for p in sents))
        return 0.05 + 0.9 * (len(c & ev) / len(c))


class BM25:
    """Okapi BM25 (k1 = 0.9, b = 0.4) over ``{doc_id: [sentence, ...]}``; ``retrieve`` returns the top sentences."""

    def __init__(self, docs: Mapping[str, Sequence[str]], k1: float = 0.9, b: float = 0.4) -> None:
        self.docs = {k: list(v) for k, v in docs.items()}
        self.k1, self.b, self.ids = k1, b, sorted(self.docs)
        self.tf: List[Dict[str, int]] = []
        df: Dict[str, int] = {}
        for i in self.ids:
            tf: Dict[str, int] = {}
            for t in _TOK.findall(" ".join(self.docs[i]).lower()):
                if t not in _STOP:
                    tf[t] = tf.get(t, 0) + 1
            self.tf.append(tf)
            for t in tf:
                df[t] = df.get(t, 0) + 1
        self.len = [sum(tf.values()) for tf in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 1.0
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def hashes(self) -> set:
        return {sha256(s) for v in self.docs.values() for s in v}

    def retrieve(self, claim: str, k: int = 1) -> List[str]:
        q = [t for t in _TOK.findall(claim.lower()) if t not in _STOP and t in self.idf]
        scored = []
        for j, tf in enumerate(self.tf):
            s, norm = 0.0, self.k1 * (1 - self.b + self.b * self.len[j] / max(1e-9, self.avg))
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if s > 0:
                scored.append((-s, self.ids[j]))
        return [sent for _, d in sorted(scored)[:k] for sent in self.docs[d]]


class LocalSources:
    """Free local sources. ``nli_dir``: directory of the ONNX DeBERTa-v3-small NLI model (config.json, tokenizer.json,
    model_qint8_avx512_vnni.onnx). ``nli2``: a Hugging Face id or path for DeBERTa-v3-base mnli-fever-anli (torch).
    Either may be None or fail to load; ``available`` lists what loaded. ``device``: "cpu" (default) or "cuda" for nli2.
    """

    def __init__(
        self,
        nli_dir: Optional[str] = None,
        nli2: Optional[str] = None,
        hf_cache: Optional[str] = None,
        device: str = "cpu",
        threads: int = 0,
    ) -> None:
        self.overlap = OverlapScorer()
        self.errors: Dict[str, str] = {}
        self._nli = self._nli2 = None
        if nli_dir:
            try:
                self._nli = _OnnxNLI(nli_dir, threads=threads, device=device)
            except Exception as e:  # missing package / model: the source is absent (profiles fail closed)
                self.errors["nli"] = f"{type(e).__name__}: {e}"[:200]
        if nli2:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._torch = torch
                self._tok2 = AutoTokenizer.from_pretrained(nli2, cache_dir=hf_cache)
                self._m2 = (
                    AutoModelForSequenceClassification.from_pretrained(nli2, cache_dir=hf_cache).eval().to(device)
                )
                self._dev = device
                self._lab2 = {v.lower(): int(k) for k, v in self._m2.config.id2label.items()}
                self._nli2 = True
            except Exception as e:
                self.errors["nli2"] = f"{type(e).__name__}: {e}"[:200]

    @property
    def available(self) -> Tuple[str, ...]:
        return tuple(n for n, ok in (("nli", self._nli), ("nli2", self._nli2), ("overlap", True)) if ok)

    def __call__(self, claim: str, sents: Sequence[str]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {"overlap": {"S": self.overlap.score(claim, sents)}}
        if not sents:
            return out
        if self._nli is not None:
            ps = self._nli.probs(claim, list(sents))
            out["nli"] = {"S": max(d["entailment"] for d in ps), "C": max(d["contradiction"] for d in ps)}
        if self._nli2:
            torch = self._torch
            e2 = c2 = 0.0
            for b in range(0, len(sents), 32):
                ch = list(sents[b : b + 32])
                x = self._tok2(
                    ch, [claim] * len(ch), truncation=True, max_length=256, padding=True, return_tensors="pt"
                )
                x = {k: v.to(self._dev) for k, v in x.items()}
                with torch.inference_mode():
                    p = torch.softmax(self._m2(**x).logits.float(), dim=-1)
                e2 = max(e2, float(p[:, self._lab2["entailment"]].max()))
                c2 = max(c2, float(p[:, self._lab2["contradiction"]].max()))
            out["nli2"] = {"S": round(e2, 6), "C": round(c2, 6)}
        return out


class _OnnxNLI:
    """ONNX cross-encoder NLI on CPU (premise = evidence sentence, hypothesis = claim)."""

    def __init__(
        self, model_dir: str, onnx_file: Optional[str] = None, max_len: int = 256, threads: int = 0, device: str = "cpu"
    ) -> None:
        import json

        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.np = np
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.labels = [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_truncation(max_length=max_len)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        if onnx_file is None:  # prefer a full-precision export if present (deterministic), else int8
            onnx_file = next(
                (
                    f
                    for f in ("model.onnx", "model_fp16.onnx", "model_qint8_avx512_vnni.onnx")
                    if os.path.exists(os.path.join(model_dir, f))
                ),
                "model.onnx",
            )
        prov = ["CPUExecutionProvider"]
        if device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]  # onnxruntime-gpu on an NVIDIA GPU
        self.sess = ort.InferenceSession(os.path.join(model_dir, onnx_file), so, providers=prov)
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


class XAIPanel:
    """The frozen Phase 3d xAI judge panel (J1 J2 J4 J7 J9 J10), PAID and OFF by default.

    Construction without ``enable=True`` raises; ``XAI_API_KEY`` is read only inside the judge call (by the Phase 3
    client, which never logs or caches it). ``cap_usd`` is a hard cap over ``ledger_path`` (append-only spend ledger,
    shared across processes); a billing/credit/auth error raises and the shadow runner turns it into NOT_SURE.
    It needs a repository checkout (``repo_root`` with phase3/), because the prompts, few-shot examples and caches live
    there; outside a checkout use your own judge callables with the same output shape.
    """

    PANEL = ("J1", "J2", "J4", "J7", "J9", "J10")

    def __init__(
        self,
        *,
        enable: bool = False,
        cap_usd: Optional[float] = None,
        repo_root: Optional[str] = None,
        ledger_path: Optional[str] = None,
    ) -> None:
        if not enable:
            raise PermissionError("xAI judges are off by default: pass enable=True and a cap_usd to opt in (paid)")
        if cap_usd is None or not cap_usd > 0:
            raise ValueError("cap_usd (a hard spend cap in USD) is required to enable paid judges")
        import sys

        root = repo_root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        p3 = os.path.join(root, "phase3")
        if not os.path.isdir(p3):
            raise FileNotFoundError("XAIPanel needs a roman-wheel-node checkout with phase3/ (repo_root=...)")
        if p3 not in sys.path:
            sys.path.insert(0, p3)
        from factcheck.interfaces import SpendLedger  # type: ignore
        from features import CACHE  # type: ignore
        from judges import make_judges  # type: ignore

        self.spend = SpendLedger(ledger_path or os.path.join(CACHE, "spend.jsonl"), float(cap_usd))
        self.judges = {n: j for n, j in make_judges(self.spend).items() if n in self.PANEL}

    def __call__(self, claim: str, evidence_text: str) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
        raw, quotes = {}, {}
        for n, j in self.judges.items():
            p, q = j.judge(claim, evidence_text)
            raw[n] = {"S": p["SUPPORT"], "C": p["CONTRADICT"], "N": p["NEI"]}
            quotes[n] = q or ""
        return raw, quotes
