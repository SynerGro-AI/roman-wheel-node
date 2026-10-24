"""Research only (not shipped): the Phase 3 judge-quote gates, moved out of the package with the xAI panel.

The shipped package has no third-party judges, so it has no quote gate. These two functions remain here only so the
historical measurements (phase5/measure_fp2.py) and their tests can be re-run from a checkout.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Sequence, Set

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from roman_wheel_node.fingerprint import _ok, normalize_v2  # noqa: E402
from roman_wheel_node.sources import sha256  # noqa: E402

__all__ = ["quote_verbatim", "quote_verbatim_v2"]


def quote_verbatim(q: str, units: Sequence[str], hashes: set) -> bool:
    """Phase 3 judge-quote gate: every sentence piece of the quote occurs byte-for-byte in a trusted unit (>= 20
    chars)."""
    if len(q or "") < 20:
        return False
    for piece in [x for x in re.split(r"(?<=[.!?])\s+", q.strip()) if x]:
        hit = [u for u in units if piece in u]
        if not hit or sha256(hit[0]) not in hashes:
            return False
    return True


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
