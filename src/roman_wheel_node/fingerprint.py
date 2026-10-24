"""Versioned text fingerprints for the exact-copy gate (policy 4.1.0+).

Fingerprint versions:

* ``fp1``: the Phase 1-4 gate, ``sha256(text.encode("utf-8"))``. A byte-identical copy is required. Frozen policy
  rwn-policy-4.0.0 uses this, and it is unchanged.
* ``fp2``: ``"rwn-fp2:" + sha256(normalize_v2(text))``. ``normalize_v2`` only removes differences that cannot change
  what a sentence says:

  1. Unicode **NFC** (canonical composition: "e" + U+0301 and U+00E9 are the same letter). **Not NFKC**, because NFKC
     folds compatibility characters that carry meaning ("10²" -> "102", "½" -> "1⁄2", "ﬁ" ligatures are harmless,
     but superscripts are not).
  2. Invisible format characters are removed: zero-width space/joiners (U+200B-U+200D, U+2060), BOM (U+FEFF) and the
     soft hyphen (U+00AD).
  3. Curly or typographic single quotes and apostrophes -> ``'``; curly or typographic double quotes -> ``"``.
  4. Hyphen and dash variants (U+2010-U+2015, U+2212 minus, U+FE58, U+FE63, U+FF0D) -> ``-``.
  5. Any run of Unicode whitespace (including NBSP, thin and ideographic spaces, tabs, newlines) -> one space, and the
     ends are stripped.

  Case, digits, punctuation other than quotes/dashes, and word order are never touched. The version tag is part of
  the fingerprint, so a fp1 hash can never be mistaken for a fp2 hash and a later fp3 cannot collide with either.

The fp2 gate is a strict superset of the fp1 gate. A unit passes if its fp1 hash or its fp2 fingerprint is trusted,
so anything fp1 admits, fp2 also admits. ``quote_verbatim_v2`` is the matching superset of the Phase 3 judge-quote
gate.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable, Optional, Sequence, Set

__all__ = ["FP_VERSIONS", "normalize_v2", "fp1", "fp2", "fingerprint", "trusted_set", "exact_gate_v2",
           "quote_verbatim_v2", "NORMALIZE_V2_TABLE_SHA256"]

FP_VERSIONS = ("fp1", "fp2")
_FP2_TAG = "rwn-fp2:"

_INVISIBLE = "\u200b\u200c\u200d\u2060\ufeff\u00ad"
_SQUOTES = "\u2018\u2019\u201a\u201b\u2032\u2035\u02bc\uff07"
_DQUOTES = "\u201c\u201d\u201e\u201f\u2033\u2036\u00ab\u00bb\uff02"
_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"
_TABLE = {ord(c): None for c in _INVISIBLE}
_TABLE.update({ord(c): "'" for c in _SQUOTES})
_TABLE.update({ord(c): '"' for c in _DQUOTES})
_TABLE.update({ord(c): "-" for c in _DASHES})
_WS = re.compile(r"\s+", re.UNICODE)
# pinned so an edit to the mapping changes a visible constant (tests check it)
NORMALIZE_V2_TABLE_SHA256 = hashlib.sha256(repr(sorted(_TABLE.items())).encode()).hexdigest()


def normalize_v2(s: str) -> str:
    """NFC, drop invisible format chars, unify quotes and dashes, collapse whitespace. Idempotent."""
    s = unicodedata.normalize("NFC", s).translate(_TABLE)
    return _WS.sub(" ", s).strip()


def fp1(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fp2(s: str) -> str:
    return _FP2_TAG + hashlib.sha256(normalize_v2(s).encode("utf-8")).hexdigest()


def fingerprint(s: str, version: str = "fp2") -> str:
    if version == "fp1":
        return fp1(s)
    if version == "fp2":
        return fp2(s)
    raise ValueError(f"unknown fingerprint version {version!r}; have {FP_VERSIONS}")


def trusted_set(units: Iterable[str], version: str = "fp2") -> Set[str]:
    """Hash set for a trusted corpus. fp2 stores both fp1 and fp2 so the fp2 gate is a superset of fp1."""
    out: Set[str] = set()
    for u in units:
        if isinstance(u, str):
            out.add(fp1(u))
            if version == "fp2":
                out.add(fp2(u))
    return out


def _ok(e: str, hs: Set[str]) -> bool:
    return fp1(e) in hs or fp2(e) in hs


def exact_gate_v2(evidence: Sequence[str], trusted: Optional[Iterable[str]]) -> int:
    """a = 1 iff evidence is non-empty and every unit's fp1 or fp2 fingerprint is trusted (None -> 0)."""
    if not evidence or trusted is None:
        return 0
    hs = trusted if isinstance(trusted, (set, frozenset)) else set(trusted)
    return int(all(isinstance(e, str) and _ok(e, hs) for e in evidence))


def quote_verbatim_v2(q: str, units: Sequence[str], trusted: Set[str]) -> bool:
    """fp2 judge-quote gate: every sentence piece of the quote occurs in a trusted unit after ``normalize_v2``.

    Same shape as the Phase 3 rule (>= 20 characters, split on sentence ends, the containing unit must be trusted).
    Elisions ("..."), paraphrase, and dropped or added words still fail, because only formatting is normalized.
    """
    if not isinstance(q, str) or len(q) < 20:
        return False
    nu = [(normalize_v2(u), u) for u in units if isinstance(u, str)]
    for piece in [x for x in re.split(r"(?<=[.!?])\s+", normalize_v2(q)) if x]:
        hit = [u for n, u in nu if piece in n]
        if not hit or not _ok(hit[0], trusted):
            return False
    return True
