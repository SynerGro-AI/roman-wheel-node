"""Gated agent trace: the only way a claim can enter an agent's context is :func:`roman_wheel_node`.

Invariants enforced by :class:`ClaimTrace`:

1. **One entry point.** :meth:`ClaimTrace.submit` is the only way to create a :class:`Claim`. Every call
   goes through :func:`roman_wheel_node.core.roman_wheel_node`. ``Claim`` cannot be constructed directly
   (private token), is immutable (``__slots__``, ``__setattr__`` raises), and the trace exposes its
   claims only as a tuple. There is no append, insert or remove.
2. **Zero-propagation.** ``a = 0`` and BLOCK never enter context. A claim whose parent is not an
   admitted claim *of this trace* (blocked, unknown, forged, or from another trace) is evaluated with
   ``a = 0`` and therefore BLOCKs, so descendants of a blocked claim can never enter either.
3. **Containment.** A blocked claim is invisible: it is not in :meth:`context`, :meth:`get` treats its id
   exactly like an id that never existed, and it can't be used as a parent. HEDGE is admitted only
   when ``allow_hedge=True`` (off by default). **NOT_SURE** (selective mode) is never admitted, whatever
   ``allow_hedge`` says: a claim whose posterior falls inside the trace's ``not_sure_band``, or that the caller
   flags with ``submit(..., not_sure=True)`` (e.g. its sources disagree), is logged with status ``NOT_SURE``
   and behaves exactly like a blocked claim (invisible, and no descendant can enter).
4. **Tamper-evident audit.** Every call (admitted or not) appends one record to a JSONL log holding the full
   audit tuple: inputs, reliabilities and cut-off used, eff, opcode, status, parent ids and timestamp.
   Each record carries ``prev`` and ``hash = H(prev || canonical(record))``, so the log is a hash chain.
   Pass ``hmac_key`` to make the chain keyed (a forger then needs the key, not just sha256).
   :func:`replay` verifies the chain *and* recomputes every decision from the logged inputs.

Threat model: this is an API-level guarantee. In-process Python can always reach private state
(``object.__setattr__``, ``gc``, importing ``_TOKEN``). Such tampering is caught by identity checks on
parents (a forged ``Claim`` is never "of this trace") and by :meth:`ClaimTrace.verify`, which replays
the log and compares it with the in-memory claims. Anchor :meth:`ClaimTrace.head` externally if you
need to detect a full rewrite of an unkeyed log.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import math
import os
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from .core import LiveWeights, ThresholdLearner, clamp01, roman_wheel_node
from .dependence import DEP_VARIANTS, FrozenDependence, votes_of

__all__ = ["Claim", "Decision", "ClaimTrace", "replay", "ReplayReport", "TamperError"]

GENESIS = "0" * 64
_TOKEN = object()  # private construction token (see threat model above)
_LOG_VERSION = 1


class TamperError(Exception):
    """The audit log or the in-memory trace does not match what was recorded."""


class Claim:
    """An admitted, immutable claim. Only :class:`ClaimTrace` can create one."""

    __slots__ = ("id", "seq", "content", "scores", "a", "parents", "eff", "opcode", "status", "record_hash")

    def __init__(
        self,
        *,
        _token: object,
        id: str,
        seq: int,
        content: str,
        scores: Mapping[str, float],
        a: int,
        parents: Tuple[str, ...],
        eff: float,
        opcode: str,
        status: str,
        record_hash: str,
    ) -> None:
        if _token is not _TOKEN:
            raise TypeError("Claim objects can only be created by ClaimTrace.submit()")
        for k, v in (
            ("id", id),
            ("seq", seq),
            ("content", content),
            ("scores", MappingProxyType(dict(scores))),
            ("a", a),
            ("parents", tuple(parents)),
            ("eff", eff),
            ("opcode", opcode),
            ("status", status),
            ("record_hash", record_hash),
        ):
            object.__setattr__(self, k, v)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Claim is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Claim is immutable")

    def __reduce__(self):  # no pickling back into existence
        raise TypeError("Claim cannot be pickled; replay the audit log instead")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __repr__(self) -> str:
        return f"Claim(id={self.id!r}, status={self.status}, eff={self.eff}, parents={self.parents})"


@dataclass(frozen=True)
class Decision:
    """Result of one :meth:`ClaimTrace.submit` call."""

    admitted: bool
    eff: float
    opcode: str
    status: str
    reason: str  # "ok" | "hedge_not_allowed" | "blocked" | "not_sure" | "parent_not_admitted"
    claim: Optional[Claim]  # None unless admitted
    seq: int


class _Frozen:
    """Replays logged reliabilities / cut-offs into roman_wheel_node (duck-typed live/threshold)."""

    def __init__(
        self, rel: Optional[Mapping[str, float]] = None, cutoff: Optional[float] = None, hedge_width: float = 0.15
    ) -> None:
        self._rel, self._cutoff, self.hedge_width = rel or {}, cutoff, hedge_width

    def reliability(self, k: str) -> float:
        return self._rel[k]

    def cutoff(self, mode: str = "") -> float:
        return self._cutoff  # type: ignore[return-value]


def _canon(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _chain_hash(prev: str, body: dict, key: Optional[bytes]) -> str:
    msg = prev.encode() + b"|" + _canon(body)
    if key:
        return _hmac.new(key, msg, hashlib.sha256).hexdigest()
    return hashlib.sha256(msg).hexdigest()


def _decide(rec: dict) -> Tuple[float, str, str]:
    """Recompute a decision purely from a logged record."""
    live = _Frozen(rec["reliability"]) if rec["reliability"] is not None else None
    thr = _Frozen(cutoff=rec["cutoff"], hedge_width=rec["hedge_width"]) if rec["learned_cutoff"] else None
    dep = FrozenDependence(rec["dependence_model"]) if rec.get("dependence_model") is not None else None
    return roman_wheel_node(
        rec["a_eff"],
        rec["scores"],
        live,
        mode=rec["mode"],
        flip=rec["flip"],
        threshold=thr,
        variant=rec["variant"],
        prior=rec["prior"],
        assumed_reliability=rec["assumed_reliability"],
        cutoff=None if rec["learned_cutoff"] else rec["cutoff"],
        dependence=dep,
    )


def _apply_not_sure(rec: dict, status: str) -> str:
    """Selective mode: fail closed to NOT_SURE inside the logged band or when the logged flag is set.

    A gate failure (a != 1, incl. a parent that is not admitted) stays BLOCK: that is a decided rejection by
    construction. A claim the node would reject on its posterior but that falls inside the band (or is flagged) becomes
    NOT_SURE too, so
    "decided" means confident in either direction. Neither BLOCK nor NOT_SURE is ever admitted."""
    if status == "BLOCK" and rec.get("a_eff", rec.get("a")) != 1:
        return status
    band = rec.get("not_sure_band")
    if rec.get("not_sure_forced"):
        return "NOT_SURE"
    if band is not None and band[0] <= rec["eff"] < band[1]:
        return "NOT_SURE"
    return status


def _admit(status: str, allow_hedge: bool) -> Tuple[bool, str]:
    if status == "NOT_SURE":
        return False, "not_sure"
    if status == "ACCEPT":
        return True, "ok"
    if status == "HEDGE":
        return (True, "ok") if allow_hedge else (False, "hedge_not_allowed")
    return False, "blocked"


class ClaimTrace:
    """An agent's gated context. See the module docstring for the invariants.

    Args:
        log_path: append-only, hash-chained JSONL audit log. If it exists, it is replayed and verified
            first and the trace resumes from it (raises :class:`TamperError` if it doesn't verify).
        live: learned reliabilities (:class:`LiveWeights`), or None.
        mode, flip, variant, prior, assumed_reliability, cutoff: passed to ``roman_wheel_node``.
        threshold: optional :class:`ThresholdLearner` (its current cut-off is logged with each call).
        dependence: optional :class:`~roman_wheel_node.dependence.DependenceLearner`, required for the variants
            ``"ess"``, ``"tan"`` and ``"stack"``. The exact model snapshot used is logged with each call.
        allow_hedge: admit HEDGE results into context (default False).
        hmac_key: optional secret; makes the chain an HMAC chain.
        clock: timestamp source (injectable for tests).
    """

    def __init__(
        self,
        log_path: str,
        live: Optional[LiveWeights] = None,
        mode: str = "geomean",
        flip: bool = False,
        threshold: Optional[ThresholdLearner] = None,
        variant: str = "hard",
        prior: float = 0.5,
        assumed_reliability: float = 0.75,
        cutoff: Union[None, float, str] = None,
        allow_hedge: bool = False,
        hmac_key: Optional[bytes] = None,
        clock: Callable[[], float] = time.time,
        dependence: Optional[object] = None,
        not_sure_band: Optional[Tuple[float, float]] = None,
    ) -> None:
        if not_sure_band is not None and not (0.0 <= float(not_sure_band[0]) <= float(not_sure_band[1])):
            raise ValueError("not_sure_band must be (lo, hi) with 0 <= lo <= hi")
        self._band = None if not_sure_band is None else [float(not_sure_band[0]), float(not_sure_band[1])]
        if mode == "bayes" and variant in DEP_VARIANTS and dependence is None:
            raise ValueError(f"variant {variant!r} needs dependence=DependenceLearner(...)")
        self._log_path = log_path
        self._dep = dependence if (mode == "bayes" and variant in DEP_VARIANTS) else None
        self._live, self._threshold = live, threshold
        self._cfg = dict(
            mode=mode,
            flip=bool(flip),
            variant=variant,
            prior=float(prior),
            assumed_reliability=float(assumed_reliability),
            allow_hedge=bool(allow_hedge),
        )
        self._fixed_cutoff = cutoff
        self._key = hmac_key
        self._clock = clock
        self._claims: List[Claim] = []
        self._by_id: Dict[str, Claim] = {}
        self._head, self._seq = GENESIS, 0
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            rep = replay(log_path, hmac_key=hmac_key)
            for c in rep.claims:
                self._register(c)
            self._head, self._seq = rep.head, rep.n_records

    # ---- read-only views --------------------------------------------------------------------
    def context(self) -> Tuple[Claim, ...]:
        """Admitted claims in order (an immutable tuple)."""
        return tuple(self._claims)

    def __iter__(self) -> Iterator[Claim]:
        return iter(tuple(self._claims))

    def __len__(self) -> int:
        return len(self._claims)

    def get(self, claim_id: str) -> Optional[Claim]:
        """The admitted claim with this id, or None (blocked and never-existing ids look the same)."""
        return self._by_id.get(claim_id)

    def head(self) -> str:
        """Hash of the last audit record. Anchor it externally to detect a full log rewrite."""
        return self._head

    # ---- the one entry point ----------------------------------------------------------------
    def submit(
        self,
        content: str,
        scores: Mapping[str, float],
        a: int,
        parents: Sequence[Union[Claim, str]] = (),
        not_sure: bool = False,
    ) -> Decision:
        """Evaluate a claim with roman_wheel_node and admit it only if it passes.

        ``parents`` may be :class:`Claim` objects or ids. Each must be an admitted claim of *this*
        trace (checked by identity), otherwise the claim is evaluated with a = 0 (zero-propagation).
        """
        if not isinstance(content, str):
            raise TypeError("content must be str")
        if isinstance(scores, (list, tuple)):
            scores = {f"s{i}": w for i, w in enumerate(scores)}
        sc = {str(k): float(v) for k, v in dict(scores).items()}
        if any(math.isnan(v) for v in sc.values()):
            raise ValueError("NaN score")
        sc = {k: (v if math.isfinite(v) else clamp01(v)) for k, v in sc.items()}  # +/-inf -> 1/0
        a_in = 1 if (a is True or (type(a) in (int, float) and a == 1)) else 0  # strict: exact match only
        pids, parents_ok = [], True
        for p in parents:
            pid = p.id if isinstance(p, Claim) else (p if isinstance(p, str) else None)
            held = self._by_id.get(pid) if pid is not None else None
            if held is None or (isinstance(p, Claim) and p is not held):
                parents_ok = False
            pids.append(pid if isinstance(pid, str) else repr(p))
        a_eff = a_in if parents_ok else 0
        rel = None
        if self._live is not None:
            rel = {k: self._live.reliability(k) for k in sc}
        learned = self._threshold is not None
        key = f"bayes:{self._cfg['variant']}" if self._cfg["mode"] == "bayes" else self._cfg["mode"]
        rec = {
            "v": _LOG_VERSION,
            "seq": self._seq,
            "ts": float(self._clock()),
            "content": content,
            "scores": sc,
            "a": a_in,
            "a_eff": a_eff,
            "parents": pids,
            "parents_ok": parents_ok,
            "reliability": rel,
            "learned_cutoff": learned,
            "cutoff": self._threshold.cutoff(key) if learned else self._fixed_cutoff,
            "hedge_width": self._threshold.hedge_width if learned else 0.15,
            **self._cfg,
        }
        if self._dep is not None:
            voting = [k for k, v in votes_of(sc).items() if v]
            rec["dependence_model"] = self._dep.model(voting, self._cfg["variant"])  # type: ignore[attr-defined]
        if self._band is not None:
            rec["not_sure_band"] = list(self._band)
        if not_sure:
            rec["not_sure_forced"] = True
        eff, op, status = _decide(rec)
        rec["eff"] = eff
        status = _apply_not_sure(rec, status)
        admitted, reason = _admit(status, self._cfg["allow_hedge"])
        if not parents_ok:
            reason = "parent_not_admitted"
        rec.update(eff=eff, opcode=op, status=status, admitted=admitted, reason=reason, prev=self._head)
        h = _chain_hash(self._head, rec, self._key)
        rec["hash"] = h
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(_canon(rec).decode() + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._head, seq = h, self._seq
        self._seq += 1
        claim = None
        if admitted:
            claim = _make_claim(rec)
            self._register(claim)
        return Decision(admitted, eff, op, status, reason, claim, seq)

    def verify(self) -> "ReplayReport":
        """Replay the log and check that the in-memory trace matches it exactly."""
        rep = replay(self._log_path, hmac_key=self._key)
        if rep.head != self._head or len(rep.claims) != len(self._claims):
            raise TamperError("in-memory trace and audit log diverge")
        for mem, logc in zip(self._claims, rep.claims):
            for f in Claim.__slots__:
                mv, lv = getattr(mem, f), getattr(logc, f)
                if f == "scores":
                    mv, lv = dict(mv), dict(lv)
                if mv != lv:
                    raise TamperError(f"claim {mem.id}: field {f!r} differs from audit log")
            if self._by_id.get(mem.id) is not mem:
                raise TamperError(f"claim index corrupted at {mem.id}")
        return rep

    def _register(self, c: Claim) -> None:
        self._claims.append(c)
        self._by_id[c.id] = c


def _make_claim(rec: dict) -> Claim:
    return Claim(
        _token=_TOKEN,
        id=rec["hash"][:16],
        seq=rec["seq"],
        content=rec["content"],
        scores=rec["scores"],
        a=rec["a_eff"],
        parents=tuple(rec["parents"]),
        eff=rec["eff"],
        opcode=rec["opcode"],
        status=rec["status"],
        record_hash=rec["hash"],
    )


@dataclass(frozen=True)
class ReplayReport:
    n_records: int
    n_admitted: int
    head: str
    claims: Tuple[Claim, ...]


def replay(log_path: str, hmac_key: Optional[bytes] = None) -> ReplayReport:
    """Verify the hash chain and recompute every decision from the logged inputs.

    Checks for every record: ``prev`` links to the previous hash, ``hash`` recomputes, ``seq`` is
    contiguous, parent admission matches the reconstructed trace, and ``a_eff``, eff, opcode, status,
    ``admitted`` and ``reason`` are exactly what roman_wheel_node and the admission rule give for the
    logged inputs. Raises :class:`TamperError` on the first mismatch.
    """
    prev, claims, by_id = GENESIS, [], {}
    n = 0
    if not os.path.exists(log_path):  # empty trace; a deleted log is caught by head mismatch
        return ReplayReport(0, 0, GENESIS, ())
    with open(log_path, encoding="utf-8") as f:
        lines = [ln for ln in f.read().split("\n") if ln.strip()]
    for n, line in enumerate(lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise TamperError(f"record {n}: not valid JSON") from e
        h = rec.pop("hash", None)
        if rec.get("prev") != prev:
            raise TamperError(f"record {n}: broken chain (prev mismatch)")
        if h != _chain_hash(prev, rec, hmac_key):
            raise TamperError(f"record {n}: hash mismatch (edited or forged)")
        if rec.get("seq") != n or rec.get("v") != _LOG_VERSION:
            raise TamperError(f"record {n}: bad sequence number or version")
        parents_ok = all(p in by_id for p in rec["parents"])
        if parents_ok != rec["parents_ok"] or rec["a_eff"] != (rec["a"] if parents_ok else 0):
            raise TamperError(f"record {n}: parent admission inconsistent with trace")
        try:
            eff, op, status = _decide(rec)
        except (ValueError, KeyError, TypeError) as e:
            raise TamperError(f"record {n}: inputs cannot be re-evaluated ({e})") from e
        status = _apply_not_sure(dict(rec, eff=eff), status)
        admitted, reason = _admit(status, rec["allow_hedge"])
        if not parents_ok:
            reason = "parent_not_admitted"
        if (eff, op, status, admitted, reason) != (
            rec["eff"],
            rec["opcode"],
            rec["status"],
            rec["admitted"],
            rec["reason"],
        ):
            raise TamperError(f"record {n}: decision does not recompute from its inputs")
        rec["hash"] = h
        if admitted:
            c = _make_claim(rec)
            claims.append(c)
            by_id[c.id] = c
        prev = h
    return ReplayReport(len(lines), len(claims), prev, tuple(claims))
