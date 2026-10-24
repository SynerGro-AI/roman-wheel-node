"""Phase 5 shared definitions: the leakage-safe COVID-Fact split, the internal data bundle, and the veto rule.

Nothing here imports torch, so the split, bundle and certification logic can be tested on any machine.

Split (frozen, deterministic, no randomness beyond sha256):
* group = sha256(evidence text)[:16]. That is the Phase 4b group key: COVID-Fact builds every counter-claim from a
  true claim's evidence, so a true claim and all of its counter-claims share one group (1,290 groups).
* half = the Phase 4b half (sha256("rwn-phase4b-structural-2026-10-22:" + group) parity). **B = held-out test**, so the
  assignment is identical to Phase 4b. No B item is ever used to train, select or tune anything in Phase 5.
* A is divided into:
  - A_quarantine: A groups that share any evidence sentence with a B group, or whose true claim is a near-duplicate
    (token Jaccard >= 0.8) of a B true claim. They are kept out of training to remove leakage across the boundary.
  - A_dev: sha256("rwn-phase5-dev:" + group) % 5 == 0. This is about 20%, used only to pick the veto threshold by the
    pre-declared rule before B is scored.
  - A_train: the rest.
* The committed manifest (results/phase5/split_manifest.json) lists only salted group hashes
  (sha256("rwn-phase5-pairs-2026-10-22:" + group)[:20]) and roles. It holds no claim or evidence text.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results", "phase5")
MANIFEST = os.path.join(RESULTS, "split_manifest.json")
PROTOCOL = os.path.join(RESULTS, "protocol.json")
PREREG = os.path.join(RESULTS, "prereg_pairs.json")
HELDOUT = os.path.join(RESULTS, "heldout_B_pairs.json")
BUNDLE = os.environ.get("RWN_P5_BUNDLE", os.path.join(HERE, "bundle", "pairs_bundle.json"))
P4B_SEED = "rwn-phase4b-structural-2026-10-22"
SALT = "rwn-phase5-pairs-2026-10-22"
DEV_SEED = "rwn-phase5-dev"
NOTE = "HealthVer and COVID-Fact: internal, unlicensed datasets; aggregates only, no text or per-item output committed"
ROLES = ("A_train", "A_dev", "A_quarantine", "B_test")
NOREG_SETS = ("scifact_dev", "healthver_test", "climatefever", "hv_test_unjudged")


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canon_sha(obj) -> str:
    return sha(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def group_of(evidence: str) -> str:
    return sha(evidence)[:16]


def p4b_half(group: str) -> str:
    return "A" if int(sha(f"{P4B_SEED}:{group}"), 16) % 2 == 0 else "B"


def gid(group: str) -> str:
    return sha(f"{SALT}:{group}")[:20]


def is_dev(group: str) -> bool:
    return int(sha(f"{DEV_SEED}:{group}"), 16) % 5 == 0


def lst(x):
    return x if isinstance(x, list) else ast.literal_eval(x)


_TOK = re.compile(r"[a-z0-9]+")


def _jacc(a: str, b: str) -> float:
    x, y = set(_TOK.findall(a.lower())), set(_TOK.findall(b.lower()))
    return len(x & y) / len(x | y) if x | y else 0.0


def build_split(items: Sequence[Mapping]) -> Dict[str, str]:
    """group -> role for COVID-Fact items ({claim, label3, evidence, units}). Deterministic."""
    groups: Dict[str, List[Mapping]] = {}
    for it in items:
        groups.setdefault(group_of(it["evidence"]), []).append(it)
    half = {g: p4b_half(g) for g in groups}
    sent_groups: Dict[str, set] = {}
    for g, its in groups.items():
        for it in its:
            for u in lst(it["units"]):
                sent_groups.setdefault(u, set()).add(g)
    quarantine = set()
    for gs in sent_groups.values():
        if {half[g] for g in gs} == {"A", "B"}:
            quarantine |= {g for g in gs if half[g] == "A"}
    b_true = [it["claim"] for g, its in groups.items() if half[g] == "B" for it in its if it["label3"] == "SUPPORT"]
    for g, its in groups.items():
        if half[g] == "A" and g not in quarantine:
            if any(_jacc(it["claim"], b) >= 0.8 for it in its if it["label3"] == "SUPPORT" for b in b_true):
                quarantine.add(g)
    roles = {}
    for g in groups:
        roles[g] = (
            "B_test" if half[g] == "B" else "A_quarantine" if g in quarantine else "A_dev" if is_dev(g) else "A_train"
        )
    return roles


def manifest_doc(items: Sequence[Mapping], roles: Mapping[str, str]) -> dict:
    per: Dict[str, Dict[str, int]] = {r: {"groups": 0, "items": 0, "supported": 0, "refuted": 0} for r in ROLES}
    gcount: Dict[str, Tuple[int, int]] = {}
    for it in items:
        g = group_of(it["evidence"])
        r = roles[g]
        per[r]["items"] += 1
        per[r]["supported" if it["label3"] == "SUPPORT" else "refuted"] += 1
        n, s = gcount.get(g, (0, 0))
        gcount[g] = (n + 1, s + (it["label3"] == "SUPPORT"))
    for g, r in roles.items():
        per[r]["groups"] += 1
    body = {
        "format": "rwn-phase5-split",
        "dataset": "COVID-Fact (internal, unlicensed; not redistributed)",
        "group_key": "sha256(evidence text)[:16] (the Phase 4b group); listed only as sha256(SALT:group)[:20]",
        "salt": SALT,
        "phase4b_seed": P4B_SEED,
        "dev_rule": f"sha256('{DEV_SEED}:'+group) % 5 == 0",
        "quarantine_rule": "A group sharing any evidence sentence with a B group, or a true claim with token Jaccard "
        ">= 0.8 to a B true claim",
        "counts": per,
        "groups": {gid(g): roles[g] for g in sorted(roles, key=gid)},
        "note": NOTE,
    }
    body["sha256"] = canon_sha({k: v for k, v in body.items() if k != "sha256"})
    return body


def load_manifest(path: str = MANIFEST) -> dict:
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    if m.get("sha256") != canon_sha({k: v for k, v in m.items() if k != "sha256"}):
        raise SystemExit("split manifest does not match its own sha256 (edited?)")
    return m


def check_roles(items: Iterable[Mapping], manifest: Mapping) -> Dict[str, str]:
    """item id -> role from the manifest; refuses if a group is unknown or the recomputed split disagrees."""
    items = list(items)
    want = build_split(items)
    out = {}
    for it in items:
        g = group_of(it["evidence"])
        r = manifest["groups"].get(gid(g))
        if r is None or r != want[g]:
            raise SystemExit(f"split mismatch for group {gid(g)}: manifest {r}, recomputed {want[g]}")
        out[it["id"]] = r
    return out


def load_bundle(path: str = BUNDLE) -> dict:
    if not os.path.exists(path):
        raise SystemExit(
            f"bundle not found: {path} (build it with phase5/make_bundle.py on the box, or set RWN_P5_BUNDLE)"
        )
    b = json.load(open(path, encoding="utf-8"))
    if b.get("format") != "rwn-phase5-bundle":
        raise SystemExit("not an rwn-phase5 bundle")
    return b


def premise(units: Sequence[str]) -> str:
    return " ".join(u.strip() for u in units if u and u.strip())


# ---------------------------------------------------------------------------------------------------------------
# threshold rule and certification (pre-declared in protocol.json before B is scored)


def tau_for_retention(p_true: Sequence[float], retention: float) -> float:
    """Lower (1 - retention) empirical quantile of P(support) over TRUE A_dev claims: index floor((1-r) * n) of the
    ascending sort. A claim is vetoed iff P(support) < tau, so about ``retention`` of true claims keep their ACCEPT."""
    xs = sorted(float(x) for x in p_true)
    if not xs:
        raise ValueError("no true claims to set tau")
    i = min(len(xs) - 1, int(math.floor((1.0 - retention) * len(xs) + 1e-9)))  # epsilon: float-safe floor
    return xs[i]


def apply_veto(outcome: Optional[str], p_support: Optional[float], tau: float) -> Optional[str]:
    """Accept-only: an ACCEPT whose checker P(support) is below tau (or missing) becomes NOT_SURE. Nothing else
    moves."""
    if outcome != "ACCEPT":
        return outcome
    if p_support is None or not math.isfinite(p_support) or p_support < tau:
        return "NOT_SURE"
    return "ACCEPT"


def fixed_sequence_ltt(stats_in_order: Sequence[Tuple[str, dict]], alphas: Sequence[float]) -> Dict[str, dict]:
    """Learn-Then-Test, fixed sequence: hypotheses H_k (false-accept share > alpha) in the pre-declared order (strict
    -> loose). Test each at delta with its one-sided Clopper-Pearson bound; stop at the first that is not rejected.
    FWER <= delta without correction. Returns, per alpha, the certified prefix and the loosest certified setting."""
    out = {}
    for al in alphas:
        cert = []
        for name, s in stats_in_order:
            if s["accepted"] > 0 and s["false_accept_share_upper"] <= al:
                cert.append(name)
            else:
                break
        out[f"{al}"] = {"certified": cert, "loosest_certified": cert[-1] if cert else None}
    return out
