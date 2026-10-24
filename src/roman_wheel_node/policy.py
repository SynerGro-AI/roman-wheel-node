"""Frozen, hash-pinned deployable policy (Phase 4).

A policy file is JSON with everything a frozen node needs: per-source isotonic calibrators, Beta reliabilities, the
learned cut-off, the accept-only margin and the disagreement rule, per *profile*. It is produced once from calibration
data (``phase6/refit_policy_42.py``) and never learns at run time.

Outcomes are exactly three:

* ``ACCEPT``   - every required source scored, the exact-copy gate passed, no disagreement flag, and the posterior
  clears the accept-only margin.
* ``BLOCK``    - the exact-copy gate failed (``a != 1``). A decided rejection.
* ``NOT_SURE`` - everything else, including any missing source, any source error, a malformed score, an unknown
  profile, or anything unexpected. The policy never rejects on its posterior (accept-only), and it **fails closed**:
  NOT_SURE is never admitted.

Honesty: the shipped policy is 4.2.0, fitted only on SciFact train (claims CC BY 4.0, abstracts ODC-By 1.0; see
DATA_LICENSES.md). Its only certificate is the judge-panel profile ``panel``: ``accept_precision`` risk, SciFact
population, alpha 10%, delta 0.05 (phase6/refit_policy_42.py). Nothing is certified at 5%, 2% or 1%, and the free
local profiles are not certified at any level. See SAFETY.md.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = ["Policy", "Profile", "PolicyError", "load_policy", "policy_sha256", "DEFAULT_POLICY_PATH",
           "DEFAULT_POLICY_SHA256", "PINNED", "OUTCOMES", "ACCEPT", "NOT_SURE", "BLOCK"]

ACCEPT, NOT_SURE, BLOCK = "ACCEPT", "NOT_SURE", "BLOCK"
OUTCOMES = (ACCEPT, NOT_SURE, BLOCK)
DEFAULT_POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies", "rwn-policy-4.2.0.json")
# sha256 of the canonical policy body (every key except "sha256"); pinned here so a swapped or edited file is refused
DEFAULT_POLICY_SHA256 = "6f7bb9e3762754dc6ca6343c44f000c7caf05a819a1f63c857471856d3946c2a"
PINNED = {os.path.abspath(DEFAULT_POLICY_PATH): DEFAULT_POLICY_SHA256}
FINGERPRINTS = ("fp1", "fp2")
EPS = 0.01                      # calibrated scores are clamped to [EPS, 1 - EPS] (as in Phase 3)


class PolicyError(Exception):
    """The policy file is missing, malformed, or does not match its pinned hash."""


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def policy_sha256(doc: Mapping) -> str:
    """sha256 of the canonical policy body (all keys except ``sha256``)."""
    return hashlib.sha256(_canon({k: v for k, v in doc.items() if k != "sha256"})).hexdigest()


def _lg(p: float) -> float:
    p = min(1 - 1e-9, max(1e-9, p))
    return math.log(p / (1 - p))


def _iso(m: Sequence[Sequence[float]], x: float) -> float:
    xs, vs = m
    for xm, v in zip(xs, vs):
        if x <= xm:
            return min(1 - EPS, max(EPS, v))
    return min(1 - EPS, max(EPS, vs[-1]))


class _FrozenRel:
    """Duck-typed LiveWeights / ThresholdLearner for ClaimTrace (fixed values, no ledger)."""

    def __init__(self, rel: Mapping[str, float], cutoff: float, hedge_width: float) -> None:
        self._rel, self._cut, self.hedge_width = dict(rel), float(cutoff), float(hedge_width)

    def reliability(self, k: str) -> float:
        return self._rel[k]

    def cutoff(self, mode: str = "") -> float:
        return self._cut


@dataclass(frozen=True)
class Profile:
    name: str
    sources: Tuple[str, ...]
    variant: str
    prior: float
    reliability: Mapping[str, float]
    cutoff: float
    hedge_width: float
    accept_margin: float
    calibrators: Mapping[str, Mapping[str, Sequence[Sequence[float]]]]
    disagreement: Mapping[str, bool]
    certified: Mapping
    needs_judges: bool
    gate_judge_quote: Optional[str]
    raw: Mapping = field(repr=False, default_factory=dict)

    # ---- pure functions of (profile, raw source outputs) -------------------------------------
    def calibrate(self, raw: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
        """Calibrated P(SUPPORT) per required source. Raises KeyError/ValueError on anything missing or malformed."""
        out = {}
        for s in self.sources:
            x = float(raw[s]["S"])
            if not math.isfinite(x) or not 0.0 <= x <= 1.0:
                raise ValueError(f"source {s}: score {x!r} outside [0, 1]")
            out[s] = _iso(self.calibrators[s]["S"], x)
        return out

    def accept_threshold(self) -> float:
        """Smallest posterior eff that clears the accept-only margin: logit(eff) >= logit(cutoff) + margin."""
        z = _lg(self.cutoff) + self.accept_margin
        return 1.0 / (1.0 + math.exp(-z)) if z < 700 else 1.0

    def disagrees(self, raw: Mapping[str, Mapping[str, float]], eff: float) -> bool:
        """Phase 3 pre-registered disagreement rule (only the parts this profile's sources allow)."""
        flag = False
        if self.disagreement.get("judge_split"):
            js = [s for s in self.sources if s.startswith("J")]
            v = [_iso(self.calibrators[s]["S"], float(raw[s]["S"])) > 0.5 for s in js]
            flag = flag or (sum(v) >= 2 and (len(v) - sum(v)) >= 2)
        if self.disagreement.get("nli2_contradiction") and "nli2" in self.sources:
            flag = flag or (eff >= self.cutoff and _iso(self.calibrators["nli2"]["C"], float(raw["nli2"]["C"])) > 0.5)
        return bool(flag)

    def frozen(self) -> _FrozenRel:
        return _FrozenRel(self.reliability, self.cutoff, self.hedge_width)


@dataclass(frozen=True)
class Policy:
    version: str
    sha256: str
    default_profile: str
    profiles: Mapping[str, Profile]
    doc: Mapping = field(repr=False, default_factory=dict)
    fingerprint: str = "fp1"          # exact-copy gate fingerprint: fp1 = raw sha256, fp2 = normalized (4.2.0)

    def profile(self, name: Optional[str] = None) -> Profile:
        n = name or self.default_profile
        if n not in self.profiles:
            raise PolicyError(f"unknown profile {n!r}; have {sorted(self.profiles)}")
        return self.profiles[n]


def _profile(name: str, d: Mapping) -> Profile:
    req = ("sources", "variant", "prior", "reliability", "cutoff", "hedge_width", "accept_margin", "calibrators")
    for k in req:
        if k not in d:
            raise PolicyError(f"profile {name}: missing {k}")
    if d["variant"] not in ("hard", "soft", "logit"):
        raise PolicyError(f"profile {name}: variant {d['variant']!r} not supported by the frozen loader")
    srcs = tuple(d["sources"])
    for s in srcs:
        if s not in d["reliability"] or s not in d["calibrators"] or "S" not in d["calibrators"][s]:
            raise PolicyError(f"profile {name}: source {s} lacks a reliability or calibrator")
    if not (0.0 < float(d["cutoff"]) < 1.0) or not float(d["accept_margin"]) > 0.0:
        raise PolicyError(f"profile {name}: cutoff must be in (0, 1) and accept_margin > 0 (accept-only)")
    return Profile(name=name, sources=srcs, variant=d["variant"], prior=float(d["prior"]),
                   reliability={k: float(v) for k, v in d["reliability"].items()}, cutoff=float(d["cutoff"]),
                   hedge_width=float(d["hedge_width"]), accept_margin=float(d["accept_margin"]),
                   calibrators=d["calibrators"], disagreement=dict(d.get("disagreement") or {}),
                   certified=dict(d.get("certified") or {}), needs_judges=bool(d.get("needs_judges")),
                   gate_judge_quote=d.get("gate_judge_quote"), raw=d)


def load_policy(path: Optional[str] = None, expected_sha256: Optional[str] = None) -> Policy:
    """Load and verify a policy file.

    The default file is checked against :data:`DEFAULT_POLICY_SHA256`. Any other file is checked against
    ``expected_sha256`` if given, and always against the ``sha256`` it carries. Any mismatch raises PolicyError.
    """
    p = path or DEFAULT_POLICY_PATH
    try:
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        raise PolicyError(f"cannot read policy {p}: {e}") from e
    h = policy_sha256(doc)
    if doc.get("sha256") != h:
        raise PolicyError("policy body does not match its own sha256 (edited?)")
    pin = expected_sha256 or PINNED.get(os.path.abspath(p))
    if pin is not None and pin != h:
        raise PolicyError(f"policy sha256 {h[:12]}... does not match the pinned {pin[:12]}...")
    if doc.get("format") != "rwn-policy" or tuple(doc.get("outcomes", ())) != OUTCOMES:
        raise PolicyError("not an rwn-policy file with outcomes ACCEPT/NOT_SURE/BLOCK")
    profs = {n: _profile(n, d) for n, d in doc["profiles"].items()}
    if doc.get("default_profile") not in profs:
        raise PolicyError("default_profile missing")
    if profs[doc["default_profile"]].needs_judges:
        raise PolicyError("the default profile must not need paid judges")
    fpv = (doc.get("gate") or {}).get("fingerprint", "fp1")
    if fpv not in FINGERPRINTS:
        raise PolicyError(f"unknown gate fingerprint {fpv!r}; have {FINGERPRINTS}")
    if fpv == "fp2":
        from .fingerprint import NORMALIZE_V2_TABLE_SHA256
        if (doc.get("gate") or {}).get("normalize_table_sha256") != NORMALIZE_V2_TABLE_SHA256:
            raise PolicyError("policy fp2 normalization table does not match this package's normalize_v2")
    return Policy(version=str(doc["version"]), sha256=h, default_profile=doc["default_profile"], profiles=profs, doc=doc,
                  fingerprint=fpv)
