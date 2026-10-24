"""Deterministic structural veto for ACCEPTs (Phase 4b).

The text of a claim and of its evidence is turned into a *structural vector*: one field per kind of structure
(numbers, negations, identifiers, content words, quantifiers/absolutes, hedges, direction words). The fields stay
separate. They are never summed into one scalar, because a sum collides: "2 numbers + 0 negations" and
"0 numbers + 2 negations" would look the same. That collision is the flaw of the Seraphina Rule 24 glyph score
(sides*10 + points + dots + intersections + spirals*8); :func:`rule24_fields` exports a per-field tuple instead.

:func:`check` compares a claim with the evidence it was accepted on, field by field. :func:`veto_accept` applies
the frozen rule :data:`FROZEN`. It can only ever turn an ``ACCEPT`` into ``NOT_SURE`` (fail closed). It never
touches ``BLOCK`` or ``NOT_SURE`` and never makes anything more permissive.

Everything here is pure stdlib, case-folded and NFKC-normalised, and deterministic: the same input always gives
the same vector, the same digest and the same decision. It needs no model, no network and no randomness.

The frozen rule's veto fields were chosen on half A of a COVID-Fact split only, before half B was scored. See
results/phase4b/. A field vetoes only if it met the pre-declared bar on half A: at least 3 false accepts caught,
and at least 3 caught for every correct accept lost. The number, negation, direction and absolute fields are
computed and exported, but on half A they did not meet that bar, so they do not veto.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["StructVec", "StructCheck", "FROZEN", "vector", "check", "veto_accept", "rule24_fields", "tokens", "key"]

FROZEN: Dict = {
    "version": "4b.0",
    "veto_fields": ("identifier_mutation", "fragment", "near_copy_swap"),
    "swap_context": 5,  # >= 5 matching tokens around the swapped slot ...
    "swap_min_side": 1,  # ... with at least 1 on each side
    "mutation_suffix": 4,  # identifier shares >= 4 trailing characters with an evidence identifier
    "on_veto": "NOT_SURE",
}

_WORD = re.compile(r"[a-z0-9\u00b5\u03b1-\u03c9]+(?:[-.][a-z0-9\u00b5\u03b1-\u03c9]+)*")
STOP = frozenset(
    """a an the of in on at to for from by with and or but as is are was were be been being has have had do does did
this that these those it its which who whom whose what when where why how than then there their they them we our you
your he she
his her i me my us not no nor so such can could may might will would shall should must also into over under about after
before
between during per via vs versus up down out off all any some each more most other another only own same very just
new""".split()
)
NEGATION = frozenset("not no never none nor without cannot neither non".split())
ABSOLUTE = frozenset(
    "all every always never none entirely completely totally only sole solely guaranteed proven proves definitely "
    "certainly 100".split()
)
HEDGE = frozenset(
    "may might could possibly possible likely unlikely suggest suggests suggested appear appears potential "
    "potentially preliminary probably perhaps".split()
)
NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "first": "1st",
    "second": "2nd",
    "third": "3rd",
    "single": "1",
    "twice": "2",
    "half": "50",
    "dozen": "12",
    "hundred": "100",
    "thousand": "1000",
    "million": "1000000",
}
_DIR_AXES = (
    (
        "increase increases increased increasing rise rises rising higher high more greater elevated boost boosts "
        "enhance enhances enhanced promote promotes raise raises",
        "decrease decreases decreased decreasing reduce reduces reduced "
        "reduction fall falls lower low less fewer decline declines diminished drop drops lowers",
    ),
    ("major large big broad wide strong strongly", "minor small narrow weak weakly slight"),
    ("better improve improves improved improvement benefit beneficial", "worse worsen worsens worsened harm harmful"),
    ("positive", "negative"),
    ("safe", "unsafe dangerous"),
    ("effective efficacious", "ineffective"),
    ("before earlier early prior", "after later late subsequent"),
    ("with", "against"),
    ("prevent prevents prevented protect protects", "cause causes caused trigger triggers"),
    ("inhibit inhibits inhibited block blocks blocked", "activate activates activated stimulate stimulates"),
)
DIRECTION: Dict[str, Tuple[int, int]] = {}
for _i, (_up, _down) in enumerate(_DIR_AXES):
    for _w in _up.split():
        DIRECTION.setdefault(_w, (_i, +1))
    for _w in _down.split():
        DIRECTION.setdefault(_w, (_i, -1))
_SUFFIXES = (
    "ations",
    "ation",
    "ically",
    "ions",
    "ion",
    "ities",
    "ity",
    "ments",
    "ment",
    "ness",
    "ingly",
    "ing",
    "edly",
    "ed",
    "ies",
    "es",
    "s",
    "ly",
    "ive",
    "ives",
    "al",
    "er",
    "ers",
    "ors",
    "or",
)
_NUM = re.compile(r"^\d+(?:[.,]\d+)*(?:st|nd|rd|th)?$")


def tokens(text: str) -> List[str]:
    """NFKC + casefold, possessive 's dropped, words kept with inner hyphens/dots (sars-cov-2, u.s., 99.9)."""
    s = unicodedata.normalize("NFKC", text).casefold().replace("\u2019", "'")
    s = re.sub(r"\s*'s\b", "", s)
    return [t.strip(".") for t in _WORD.findall(s) if t.strip(".")]


def _stem(t: str) -> str:
    for suf in _SUFFIXES:
        if len(t) - len(suf) >= 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def key(t: str) -> str:
    """Comparison key: number words -> digits, hyphens/dots dropped, crude suffix stem."""
    t = NUMBER_WORDS.get(t, t)
    return _stem(t.replace("-", "").replace(".", ""))


def morph_eq(a: str, b: str) -> bool:
    """Same word up to morphology/abbreviation: equal, prefix (>=3), compound tail (>=5) or shared 5-prefix."""
    if a == b:
        return True
    s, longer = sorted((a, b), key=len)
    if len(s) >= 3 and longer.startswith(s):
        return True
    if len(s) >= 5 and longer.endswith(s):
        return True
    n = 0
    while n < len(s) and a[n] == b[n]:
        n += 1
    return n >= 5


def _number(t: str) -> Optional[str]:
    t = NUMBER_WORDS.get(t, t)
    if _NUM.match(t):
        return re.sub(r"(st|nd|rd|th)$", "", t.replace(",", ""))
    return None


def _is_identifier(t: str) -> bool:
    return "-" in t or any(c.isdigit() for c in t)


@dataclass(frozen=True)
class StructVec:
    """The structural vector of one text. Every field is kept separately; there is no total."""

    numbers: Tuple[str, ...] = ()
    negations: int = 0
    identifiers: Tuple[str, ...] = ()
    content: Tuple[str, ...] = ()
    absolutes: Tuple[str, ...] = ()
    hedges: Tuple[str, ...] = ()
    directions: Tuple[Tuple[int, int], ...] = ()

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        """sha256 of the canonical per-field encoding: equal digests iff equal vectors (no collisions by design)."""
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()


def vector(text: str) -> StructVec:
    T = tokens(text)
    return StructVec(
        numbers=tuple(sorted({n for n in (_number(t) for t in T) if n})),
        negations=sum(t in NEGATION for t in T),
        identifiers=tuple(sorted({t for t in T if _is_identifier(t) and not _number(t)})),
        content=tuple(sorted({key(t) for t in T if t not in STOP})),
        absolutes=tuple(sorted({t for t in T if t in ABSOLUTE})),
        hedges=tuple(sorted({t for t in T if t in HEDGE})),
        directions=tuple(sorted({DIRECTION[t] for t in T if t in DIRECTION})),
    )


def rule24_fields(v: StructVec) -> Dict[str, object]:
    """Seraphina Rule 24-compatible export: the five glyph fields as a per-field tuple, plus the digest.

    sides = content words, points = numbers, dots = negations, intersections = direction words,
    spirals = hedges + absolutes. The Rule 24 scalar (sides*10 + points + dots + intersections + spirals*8) is
    deliberately NOT computed: different vectors share the same sum. Compare ``fields`` or ``digest`` instead.
    """
    f = (len(v.content), len(v.numbers), v.negations, len(v.directions), len(v.hedges) + len(v.absolutes))
    return {"names": ("sides", "points", "dots", "intersections", "spirals"), "fields": f, "digest": v.digest()}


@dataclass
class StructCheck:
    """Field-by-field comparison of a claim with its evidence. ``veto`` follows :data:`FROZEN`."""

    identifier_mutation: List[Tuple[str, str]] = field(default_factory=list)
    fragment: List[str] = field(default_factory=list)
    near_copy_swap: List[Tuple[str, str, int, int]] = field(default_factory=list)
    numbers_missing: List[str] = field(default_factory=list)
    negation: Tuple[int, int] = (0, 0)
    direction_conflict: List[Tuple[int, int]] = field(default_factory=list)
    absolutes_unbacked: List[str] = field(default_factory=list)
    hedges: Tuple[int, int] = (0, 0)
    veto: bool = False
    reasons: Tuple[str, ...] = ()

    def pattern(self) -> Dict[str, int]:
        """Counts only (no text): safe to log in aggregates."""
        return {
            "identifier_mutation": len(self.identifier_mutation),
            "fragment": len(self.fragment),
            "near_copy_swap": len(self.near_copy_swap),
            "numbers_missing": len(self.numbers_missing),
            "negation_claim": self.negation[0],
            "negation_evidence": self.negation[1],
            "direction_conflict": len(self.direction_conflict),
            "absolutes_unbacked": len(self.absolutes_unbacked),
        }


def _mutations(CT: Sequence[str], EK: Iterable[str], min_suffix: int) -> List[Tuple[str, str]]:
    """A claim identifier absent from the evidence that shares a >= min_suffix tail with a digit-bearing evidence
    identifier but has a different head (e.g. a wrong variant / drug / gene code next to the right one)."""
    EK = set(EK)
    ids = sorted(e for e in EK if any(c.isdigit() for c in e))
    out = []
    for t in CT:
        k = key(t)
        if not _is_identifier(t) or k in EK:
            continue
        for e in ids:
            if e == k:
                continue
            n = 0
            while n < min(len(e), len(k)) and e[-1 - n] == k[-1 - n]:
                n += 1
            if min_suffix <= n < len(k) and n < len(e):
                out.append((t, e))
                break
    return out


def _swaps(
    CT: Sequence[str], units: Sequence[str], EK: Sequence[str], ctx: int, side: int
) -> List[Tuple[str, str, int, int]]:
    """A claim content word with no match anywhere in the evidence, sitting in a slot of an evidence sentence where the
    surrounding context matches (>= ctx tokens, >= side on each side) but the evidence has a different content word."""
    C = [key(t) for t in CT]
    out = []
    unmatched = [i for i, t in enumerate(CT) if t not in STOP and not any(morph_eq(C[i], e) for e in EK)]
    if not unmatched:
        return out
    for u in units:
        ST = tokens(u)
        S = [key(t) for t in ST]
        for i in unmatched:
            for j in range(len(S)):
                if ST[j] in STOP or morph_eq(S[j], C[i]):
                    continue
                left = 0
                while i - left - 1 >= 0 and j - left - 1 >= 0 and morph_eq(C[i - left - 1], S[j - left - 1]):
                    left += 1
                r = 0
                while i + r + 1 < len(C) and j + r + 1 < len(S) and morph_eq(C[i + r + 1], S[j + r + 1]):
                    r += 1
                if left + r >= ctx and min(left, r) >= side:
                    out.append((CT[i], ST[j], left, r))
                    break
    return out


def check(
    claim: str, evidence: Sequence[str], quotes: Optional[Sequence[str]] = None, rule: Optional[Dict] = None
) -> StructCheck:
    """Compare ``claim`` with its evidence sentences (plus optional quotes, used as extra sentences)."""
    rule = FROZEN if rule is None else rule
    units = [u for u in list(evidence) + list(quotes or ()) if u]
    ET = tokens(" ".join(units))
    ES = set(ET)
    EK = [key(e) for e in ET]
    CT = tokens(claim)
    cv, ev = vector(claim), vector(" ".join(units))
    res = StructCheck(
        identifier_mutation=_mutations(CT, EK, rule["mutation_suffix"]),
        fragment=sorted({t for t in CT if len(t) == 1 and t.isalpha() and t not in ("a", "i") and t not in ES}),
        near_copy_swap=_swaps(CT, units, EK, rule["swap_context"], rule["swap_min_side"]),
        numbers_missing=sorted(set(cv.numbers) - set(ev.numbers)),
        negation=(cv.negations, ev.negations),
        direction_conflict=sorted(
            d for d in cv.directions if (d[0], -d[1]) in ev.directions and d not in ev.directions
        ),
        absolutes_unbacked=sorted(set(cv.absolutes) - set(ev.absolutes)),
        hedges=(len(cv.hedges), len(ev.hedges)),
    )
    res.reasons = tuple(f for f in rule["veto_fields"] if getattr(res, f))
    res.veto = bool(res.reasons)
    return res


def veto_accept(
    outcome: str,
    claim: str,
    evidence: Sequence[str],
    quotes: Optional[Sequence[str]] = None,
    rule: Optional[Dict] = None,
) -> Tuple[str, Optional[StructCheck]]:
    """Apply the structural veto. Only an ACCEPT is ever changed (to NOT_SURE); every other outcome is returned as
    is."""
    if outcome != "ACCEPT":
        return outcome, None
    c = check(claim, evidence, quotes, rule)
    return ((FROZEN if rule is None else rule)["on_veto"] if c.veto else outcome), c


def frozen_rule_sha256() -> str:
    """Hash of the frozen rule's parameters together with this module's source (pins the pre-registered rule)."""
    import inspect
    import sys

    src = inspect.getsource(sys.modules[__name__])
    return hashlib.sha256((json.dumps(FROZEN, sort_keys=True) + "\n" + src).encode("utf-8")).hexdigest()
