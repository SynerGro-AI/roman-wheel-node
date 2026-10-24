"""Fit your own frozen policy from your own labelled data (free local sources only; no API calls).

    python -m roman_wheel_node.fit my_labels.jsonl -o my-policy.json --nli2 MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

    from roman_wheel_node.fit import fit_policy, write_policy
    doc = fit_policy("my_labels.jsonl", profiles={"local": ("nli2", "overlap")}, sources=LocalSources(nli2=...))
    write_policy(doc, "my-policy.json")
    node = ShadowRunner("rwn_shadow/", policy=load_policy("my-policy.json"), nli2=...)

Input: JSONL, one labelled example per line::

    {"claim": "...", "evidence": ["sentence", ...] or "text", "label": "SUPPORT" | "CONTRADICT" | "NEI" | true | false,
     "id": "optional", "scores": {"nli2": {"S": .., "C": ..}, "overlap": {"S": ..}}  (optional)}

``label`` true means SUPPORT and false means NEI (not supported). A line with ``scores`` is used as given; otherwise
the scores are computed with ``sources`` (default :class:`~roman_wheel_node.sources.LocalSources`, the same sources
the runtime uses, so the fitted calibrators see the same features). Only the open local sources ``nli``, ``nli2``
and ``overlap`` can be fitted: a policy that needs anything else is refused by
:func:`~roman_wheel_node.policy.load_policy`.

Method (the same one that produced the shipped policy 4.3.0, ``phase6/refit_policy_43.py``):

1. Per-source isotonic calibration (pool-adjacent violators, Laplace-smoothed blocks) of P(SUPPORT) and, for the NLI
   sources, P(CONTRADICT).
2. Beta source reliabilities (:class:`~roman_wheel_node.core.LiveWeights`, no forgetting) and the learned cut-off
   (:class:`~roman_wheel_node.core.ThresholdLearner`) of the ``bayes`` / ``logit`` node on the calibrated scores.
3. Accept-only margin, pre-registered from calibration data only: out-of-fold posteriors (``folds``-fold, example
   ``i`` in fold ``i % folds``, so input order fixes the folds), the fixed path of ``K`` margins (step q accepts the top
   q/K of the out-of-fold accepts), and the step with the lowest one-sided Clopper-Pearson upper bound (delta 0.05) on
   the share of false accepts.
4. Optional certificate: Learn-Then-Test fixed-sequence testing along the same path at each ``alphas`` level
   (one-sided Clopper-Pearson, ``delta / n_tests``). If one certifies, the margin of the tightest certified alpha
   replaces the pre-registered one and the certificate is written into the profile; otherwise the profile says it is
   not certified. A certificate is only about data exchangeable with your calibration examples.

Everything is deterministic: the same examples (in the same order) and scores give a byte-identical policy file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ._version import __version__
from .core import LiveWeights, ThresholdLearner, give_feedback, roman_wheel_node
from .fresh import cp_upper
from .policy import LOCAL_SOURCES, OUTCOMES, policy_sha256

__all__ = [
    "LABELS",
    "load_examples",
    "score_examples",
    "isotonic_fit",
    "fit_profile",
    "build_policy",
    "fit_policy",
    "write_policy",
    "FitError",
]

LABELS = ("SUPPORT", "CONTRADICT", "NEI")
S_ONLY = ("overlap",)  # sources without a P(CONTRADICT) output
EPS = 0.01  # calibrated scores are clamped to [EPS, 1 - EPS], as in policy.Profile.calibrate
INF = float("inf")
MODE = "bayes:logit"
DEFAULT_PROFILES = {"local": ("nli2", "overlap"), "local-lite": ("nli", "overlap")}

Example = Dict[str, object]


class FitError(ValueError):
    """The examples or the fit request are unusable (bad label, missing source score, too few examples, ...)."""


# ---- input --------------------------------------------------------------------------------------------------------
def _norm_example(d: Mapping, where: str) -> Example:
    if not isinstance(d, Mapping):
        raise FitError(f"{where}: expected a JSON object")
    claim = d.get("claim")
    if not isinstance(claim, str) or not claim.strip():
        raise FitError(f"{where}: 'claim' must be a non-empty string")
    ev = d.get("evidence", [])
    if isinstance(ev, str):
        from .sources import split_sentences

        ev = split_sentences(ev)
    if not isinstance(ev, (list, tuple)) or not all(isinstance(s, str) for s in ev):
        raise FitError(f"{where}: 'evidence' must be a string or a list of strings")
    lab = d.get("label")
    if isinstance(lab, bool):
        lab = "SUPPORT" if lab else "NEI"
    if not isinstance(lab, str) or lab.upper() not in LABELS:
        raise FitError(f"{where}: 'label' must be one of {LABELS} or true/false, got {lab!r}")
    out: Example = {"id": str(d.get("id", where)), "claim": claim, "evidence": [s for s in ev if s.strip()]}
    out["label"] = lab.upper()
    if d.get("scores") is not None:
        out["scores"] = _check_scores(d["scores"], where)
    return out


def _check_scores(sc, where: str) -> Dict[str, Dict[str, float]]:
    if not isinstance(sc, Mapping):
        raise FitError(f"{where}: 'scores' must be an object {{source: {{'S': p, 'C': p}}}}")
    out: Dict[str, Dict[str, float]] = {}
    for s, v in sc.items():
        if not isinstance(v, Mapping):
            raise FitError(f"{where}: scores[{s!r}] must be an object")
        out[str(s)] = {}
        for t, x in v.items():
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1:
                raise FitError(f"{where}: scores[{s!r}][{t!r}] = {x!r} is not a probability")
            out[str(s)][str(t)] = float(x)
    return out


def load_examples(path: str) -> List[Example]:
    """Read labelled examples from a JSONL file (one JSON object per line; blank lines are skipped)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    d = json.loads(line)
                except (ValueError, RecursionError) as e:  # RecursionError: nesting deeper than the parser allows
                    raise FitError(f"{path}:{i}: not JSON ({type(e).__name__}: {str(e)[:200]})") from e
                out.append(_norm_example(d, f"{os.path.basename(path)}:{i}"))
    return out


def score_examples(
    examples: Iterable[Mapping], sources: Optional[Callable] = None, needed: Sequence[str] = ()
) -> List[Example]:
    """Fill in ``scores`` for every example that has none, with ``sources(claim, sentences) -> raw`` (for example a
    :class:`~roman_wheel_node.sources.LocalSources`). Raises FitError if a source in ``needed`` is missing."""
    out = []
    for i, d in enumerate(examples):
        e = _norm_example(d, str(d.get("id", f"example {i}")) if isinstance(d, Mapping) else f"example {i}")
        if "scores" not in e:
            if sources is None:
                raise FitError(f"{e['id']}: no 'scores' and no sources to compute them")
            e["scores"] = _check_scores(sources(e["claim"], list(e["evidence"])), str(e["id"]))
        miss = [s for s in needed if s not in e["scores"] or "S" not in e["scores"][s]]
        if miss:
            raise FitError(f"{e['id']}: missing score for source(s) {miss}")
        out.append(e)
    return out


# ---- calibration and learners -------------------------------------------------------------------------------------
def isotonic_fit(xs: Sequence[float], ys: Sequence[int]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Isotonic (pool-adjacent-violators) calibration with Laplace smoothing (pos+1)/(n+2) per block.
    Returns (upper x of each block, calibrated value of each block)."""
    pts = sorted(zip(xs, ys))
    blocks: List[List[float]] = []  # [x_max, pos, n]
    i = 0
    while i < len(pts):  # tie groups first: equal x must share a value
        j, pos = i, 0
        while j < len(pts) and pts[j][0] == pts[i][0]:
            pos += pts[j][1]
            j += 1
        blocks.append([pts[i][0], float(pos), float(j - i)])
        while len(blocks) > 1 and (blocks[-2][1] + 1) / (blocks[-2][2] + 2) >= (blocks[-1][1] + 1) / (
            blocks[-1][2] + 2
        ):
            x, p_, n_ = blocks.pop()
            blocks[-1][0] = x
            blocks[-1][1] += p_
            blocks[-1][2] += n_
        i = j
    return tuple(b[0] for b in blocks), tuple((b[1] + 1) / (b[2] + 2) for b in blocks)


def _iso(m, x: float) -> float:
    xs, vs = m
    for xm, v in zip(xs, vs):
        if x <= xm:
            return min(1 - EPS, max(EPS, v))
    return min(1 - EPS, max(EPS, vs[-1]))


def _y(e: Mapping) -> bool:
    return e["label"] == "SUPPORT"


def _calibrators(rows: Sequence[Mapping], srcs: Sequence[str]) -> Dict[Tuple[str, str], tuple]:
    cal = {}
    for s in srcs:
        for t in ("S", "C"):
            if t == "C" and (s in S_ONLY or any("C" not in r["scores"][s] for r in rows)):
                continue
            target = "SUPPORT" if t == "S" else "CONTRADICT"
            cal[(s, t)] = isotonic_fit([r["scores"][s][t] for r in rows], [int(r["label"] == target) for r in rows])
    return cal


class _Fitted:
    """Calibrators, reliabilities and cut-off of the bayes/logit node fitted on ``rows`` (no forgetting)."""

    def __init__(self, rows: Sequence[Mapping], srcs: Sequence[str], tmp: str) -> None:
        self.srcs, self.cal = tuple(srcs), _calibrators(rows, srcs)
        self.lw = LiveWeights(os.path.join(tmp, "w.jsonl"), half_life=None)
        for r in rows:
            give_feedback(self.sc(r), _y(r), live=self.lw)
        thr = ThresholdLearner(os.path.join(tmp, "t.jsonl"), half_life=None)
        for r in rows:
            thr.record(self.eff(r), _y(r), MODE)
        self.cut, self.hedge_width = thr.cutoff(MODE), thr.hedge_width
        self.rel = {s: self.lw.reliability(s) for s in self.srcs}

    def sc(self, r: Mapping) -> Dict[str, float]:
        return {s: _iso(self.cal[(s, "S")], r["scores"][s]["S"]) for s in self.srcs}

    def eff(self, r: Mapping) -> float:
        return roman_wheel_node(1, self.sc(r), live=self.lw, mode="bayes", variant="logit")[0]

    def disagrees(self, r: Mapping, eff: float, rules: Mapping[str, bool]) -> bool:
        flag = False
        if rules.get("nli2_contradiction") and "nli2" in self.srcs and ("nli2", "C") in self.cal:
            flag = eff >= self.cut and _iso(self.cal[("nli2", "C")], r["scores"]["nli2"]["C"]) > 0.5
        return bool(eff > 0.0 and flag)


def _fit(rows: Sequence[Mapping], srcs: Sequence[str]) -> _Fitted:
    with tempfile.TemporaryDirectory(prefix="rwn_fit_") as tmp:
        return _Fitted(rows, srcs, tmp)


def _oof(rows, srcs, folds, disagreement):
    """Out-of-fold posteriors, cut-offs and disagreement flags: example i is scored by a fit on the other folds."""
    effs, cuts, dis = [0.0] * len(rows), [0.5] * len(rows), [False] * len(rows)
    for f in range(folds):
        m = _fit([r for i, r in enumerate(rows) if i % folds != f], srcs)
        for i in range(f, len(rows), folds):
            effs[i], cuts[i] = m.eff(rows[i]), m.cut
            dis[i] = m.disagrees(rows[i], effs[i], disagreement)
    return effs, cuts, dis


def _items(rows, effs, cuts, dis) -> List[dict]:
    out = []
    for r, e, c, d in zip(rows, effs, cuts, dis):
        e2, c2 = min(1 - 1e-9, max(1e-9, e)), min(1 - 1e-9, max(1e-9, c))
        out.append(
            {
                "accept_side": e >= c and not d,
                "z": abs(math.log(e2 / (1 - e2)) - math.log(c2 / (1 - c2))),
                "y": _y(r),
            }
        )
    return out


def _path(items: Sequence[dict], K: int) -> List[float]:
    za = sorted((it["z"] for it in items if it["accept_side"]), reverse=True)
    out = []
    for q in range(1, K + 1):
        n = int(round(q / K * len(za)))
        out.append(INF if n <= 0 else za[n - 1])
    return out


def _risk(items: Sequence[dict], m: float) -> Tuple[int, int]:
    acc = [it for it in items if it["accept_side"] and it["z"] >= m]
    return len(acc), sum(1 for it in acc if not it["y"])


def _ltt(items, alpha: float, delta: float, K: int) -> dict:
    n_min = math.ceil(math.log(delta) / math.log1p(-alpha))
    best: dict = {"certified": False}
    for q, m in enumerate(_path(items, K), 1):
        n, k = _risk(items, m)
        if n < n_min:
            continue  # cannot pass even with 0 errors; skipping uses no label
        ub = cp_upper(k, n, delta)
        if ub > alpha:
            break
        best = {"certified": True, "margin": m, "step": q, "accepted": n, "false_accepts": k, "upper": ub}
    return best


def _check_folds(folds: object) -> int:
    """Cross-fitting needs at least 2 folds (with 1, every out-of-fold fit would be trained on nothing)."""
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise FitError(f"folds must be an integer >= 2; got {folds!r}")
    return folds


def fit_profile(
    examples: Sequence[Mapping],
    sources: Sequence[str],
    *,
    folds: int = 5,
    K: int = 40,
    alphas: Sequence[float] = (),
    delta: float = 0.05,
    n_tests: int = 1,
    disagreement: Optional[Mapping[str, bool]] = None,
    pool_note: str = "",
) -> dict:
    """Fit one profile (a JSON-ready dict for :func:`build_policy`) on scored examples. See the module docstring."""
    srcs = tuple(sources)
    bad = [s for s in srcs if s not in LOCAL_SOURCES]
    if not srcs or bad:
        raise FitError(f"profile sources must be a non-empty subset of {LOCAL_SOURCES}; got {list(srcs)}")
    dis_cfg = {k: bool(v) for k, v in (disagreement or {}).items() if v}
    if set(dis_cfg) - {"nli2_contradiction"}:
        raise FitError(f"unsupported disagreement rule(s) {sorted(set(dis_cfg) - {'nli2_contradiction'})}")
    _check_folds(folds)
    rows = list(examples)
    for r in rows:
        for s in srcs:
            if "S" not in r.get("scores", {}).get(s, {}):
                raise FitError(f"{r.get('id')}: missing score for source {s}")
    ys = [_y(r) for r in rows]
    if len(rows) < 10 * folds:
        raise FitError(f"need at least {10 * folds} examples for {folds}-fold cross-fitting; got {len(rows)}")
    for f in range(folds):  # fold f = examples f, f + folds, ... (the same split _oof uses)
        fy = ys[f::folds]
        if all(fy) or not any(fy):
            raise FitError(
                f"need at least {10 * folds} examples with both classes in every fold; fold {f + 1} of {folds} has "
                f"only {'SUPPORT' if fy and fy[0] else 'non-SUPPORT'} examples ({len(fy)}); reorder or add examples"
            )
    full = _fit(rows, srcs)
    cal, rel, cut, hw = full.cal, full.rel, full.cut, full.hedge_width
    effs, cuts, dis = _oof(rows, srcs, folds, dis_cfg)
    items = _items(rows, effs, cuts, dis)
    pre = None
    for q, m in enumerate(_path(items, K), 1):
        n, k = _risk(items, m)
        ub = cp_upper(k, n, 0.05)
        if math.isfinite(m) and (pre is None or ub <= pre["upper95"]):
            pre = {"step": q, "margin": m, "accepted_oof": n, "false_accepts_oof": k, "upper95": ub}
    if pre is None:
        raise FitError("no out-of-fold accept at any margin: the sources never lean 'supported' on these examples")
    pre.update(
        n_pool=len(rows),
        rule=f"argmin one-sided 95% CP upper bound of the false-accept share over the fixed K={K} accept-only "
        f"margin path, {folds}-fold out-of-fold posteriors",
    )
    certs = {f"{a}": _ltt(items, float(a), delta / n_tests, K) for a in alphas}
    chosen = min((float(a) for a, c in certs.items() if c["certified"]), default=None)
    calj: Dict[str, Dict[str, list]] = {}
    for (s, t), m in cal.items():
        calj.setdefault(s, {})[t] = [list(m[0]), list(m[1])]
    prof = {
        "sources": list(srcs),
        "variant": "logit",
        "prior": 0.5,
        "reliability": rel,
        "cutoff": cut,
        "hedge_width": hw,
        "calibrators": calj,
        "disagreement": dis_cfg,
        "calibration": {"pool": pool_note, "n": len(rows), "folds": folds},
        "preregistration": pre,
        "ltt": {"alphas": certs, "delta": delta, "n_tests": n_tests, "K": K} if alphas else None,
    }
    if chosen is None:
        prof["accept_margin"] = pre["margin"]
        prof["certified"] = {
            "alpha": None,
            "note": "NOT certified at any level"
            + (f" tested ({', '.join(f'{float(a):.0%}' for a in alphas)})" if alphas else " (no certificate requested)")
            + f"; best out-of-fold upper bound on the false-accept share {pre['upper95']:.1%} (preregistration)",
        }
    else:
        c = certs[f"{chosen}"]
        prof["accept_margin"] = c["margin"]
        prof["certified"] = {
            "alpha": chosen,
            "delta": delta,
            "n_tests": n_tests,
            "risk": "accept_precision (false accepts among ACCEPTs)",
            "population": f"the calibration examples (out-of-fold, n={len(rows)})",
            "calibration_accepted": c["accepted"],
            "calibration_false_accepts": c["false_accepts"],
            "upper": c["upper"],
            "not_certified": [f"alpha {a}" for a in alphas if not certs[f"{a}"]["certified"]],
        }
    return prof


# ---- policy file --------------------------------------------------------------------------------------------------
def build_policy(
    profiles: Mapping[str, Mapping],
    *,
    version: str,
    default_profile: str,
    data_licences: str = "",
    fit_script: str = "roman_wheel_node.fit",
    extra: Optional[Mapping] = None,
) -> dict:
    """Assemble a complete ``rwn-policy`` document (fp2 exact-copy gate) and stamp its sha256."""
    from .fingerprint import NORMALIZE_V2_TABLE_SHA256

    if default_profile not in profiles:
        raise FitError(f"default_profile {default_profile!r} is not one of {sorted(profiles)}")
    cert = [f"{n} at {p['certified']['alpha']:.0%}" for n, p in profiles.items() if p["certified"].get("alpha")]
    doc = {
        "format": "rwn-policy",
        "version": str(version),
        "outcomes": list(OUTCOMES),
        "default_profile": default_profile,
        "fit_script": fit_script,
        "fitted_with": f"roman-wheel-node {__version__} (roman_wheel_node.fit)",
        "data_licences": data_licences,
        "gate": {
            "fingerprint": "fp2",
            "normalize": "NFC; drop U+200B-200D,2060,FEFF,00AD; curly/typographic quotes "
            "-> ' and \"; hyphen/dash/minus variants -> -; whitespace runs -> one space; strip. Not NFKC.",
            "normalize_table_sha256": NORMALIZE_V2_TABLE_SHA256,
            "superset_of": "fp1 (a unit passes if its raw sha256 or its fp2 fingerprint is trusted)",
        },
        "fail_closed": "missing or erroring source, malformed score, disagreement flag, or posterior below the accept "
        "margin -> NOT_SURE (never admitted); exact-copy gate failure -> BLOCK",
        "honesty": ("Certified (accept precision, calibration population only): " + "; ".join(cert) + ".")
        if cert
        else "No profile is certified at any level.",
        "profiles": {n: dict(p) for n, p in profiles.items()},
    }
    doc.update(extra or {})
    doc["sha256"] = policy_sha256(doc)
    return doc


def write_policy(doc: Mapping, path: str) -> str:
    """Write a policy document (canonical layout, trailing newline). Returns its sha256."""
    if doc.get("sha256") != policy_sha256(doc):
        raise FitError("policy body does not match its sha256")
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    return str(doc["sha256"])


def fit_policy(
    examples: Union[str, Sequence[Mapping]],
    *,
    profiles: Optional[Mapping[str, Sequence[str]]] = None,
    sources: Optional[Callable] = None,
    default_profile: Optional[str] = None,
    version: str = "custom-1",
    data_licences: str = "",
    folds: int = 5,
    K: int = 40,
    alphas: Sequence[float] = (),
    delta: float = 0.05,
    disagreement: Optional[Mapping[str, Mapping[str, bool]]] = None,
) -> dict:
    """Examples (a JSONL path or a list of dicts) -> a complete, loadable policy document.

    ``profiles`` maps a profile name to its sources (default: ``local`` = nli2 + overlap and ``local-lite`` = nli +
    overlap, keeping only those every example can be scored for). ``alphas``: certificate levels to test (Bonferroni
    over all profiles: each test runs at ``delta / number of profiles``).
    """
    _check_folds(folds)  # before any (slow) scoring
    rows = load_examples(examples) if isinstance(examples, str) else list(examples)
    if profiles is None:
        rows = score_examples(rows, sources)
        have = set.intersection(*(set(r["scores"]) for r in rows)) if rows else set()
        profiles = {n: s for n, s in DEFAULT_PROFILES.items() if set(s) <= have}
        if not profiles:
            raise FitError(f"no default profile can be fitted: every example has scores for {sorted(have)} only")
    for n, v in profiles.items():  # refuse non-local sources before scoring anything
        if not v or any(s not in LOCAL_SOURCES for s in v):
            raise FitError(f"profile {n}: profile sources must be a non-empty subset of {LOCAL_SOURCES}; got {list(v)}")
    need = sorted({s for v in profiles.values() for s in v})
    rows = score_examples(rows, sources, need)
    note = f"{len(rows)} labelled examples" + (f"; {data_licences}" if data_licences else "")
    fitted = {
        n: fit_profile(
            rows,
            s,
            folds=folds,
            K=K,
            alphas=alphas,
            delta=delta,
            n_tests=len(profiles),
            disagreement=(disagreement or {}).get(n),
            pool_note=note,
        )
        for n, s in profiles.items()
    }
    dp = default_profile or ("local" if "local" in fitted else sorted(fitted)[0])
    return build_policy(fitted, version=version, default_profile=dp, data_licences=data_licences)


# ---- command line -------------------------------------------------------------------------------------------------
def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m roman_wheel_node.fit",
        description="Fit a frozen rwn-policy file from your own labelled examples (JSONL of claim, evidence, label).",
    )
    ap.add_argument("examples", help="labelled JSONL: {claim, evidence, label[, id][, scores]} per line")
    ap.add_argument("-o", "--out", required=True, help="output policy JSON")
    ap.add_argument(
        "--profile",
        action="append",
        default=None,
        metavar="NAME=SRC[,SRC]",
        help="profile to fit, e.g. local=nli2,overlap (repeatable; default: local and local-lite where scorable)",
    )
    ap.add_argument("--default-profile", default=None)
    ap.add_argument("--version-tag", default="custom-1", help="the policy's own version string (default custom-1)")
    ap.add_argument("--data-licences", default="", help="where the examples come from and under which licence")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alphas", default="", help="certificate levels to test, e.g. 0.10,0.05 (default: none)")
    ap.add_argument("--nli-dir", default=os.environ.get("RWN_NLI_DIR"))
    ap.add_argument("--nli2", default=os.environ.get("RWN_NLI2"))
    ap.add_argument("--hf-cache", default=os.environ.get("RWN_HF_CACHE"))
    ap.add_argument("--device", default=os.environ.get("RWN_DEVICE", "cpu"))
    args = ap.parse_args(argv)
    profiles = None
    if args.profile:
        profiles = {}
        for spec in args.profile:
            name, sep, srcs = spec.partition("=")
            if not sep or not name or not srcs:
                ap.error(f"--profile {spec!r}: expected NAME=SRC[,SRC]")
            profiles[name] = tuple(s.strip() for s in srcs.split(",") if s.strip())
    try:
        alphas = tuple(float(a) for a in args.alphas.split(",") if a.strip())
    except ValueError:
        ap.error(f"--alphas {args.alphas!r}: expected comma-separated numbers")
    if args.folds < 2:
        ap.error(f"--folds {args.folds}: must be at least 2 (cross-fitting needs a held-out fold)")
    try:
        rows = load_examples(args.examples)
    except FitError as e:
        ap.error(str(e))
    except OSError as e:
        ap.error(f"cannot read {args.examples}: {e.strerror or e}")
    except UnicodeDecodeError as e:
        ap.error(f"{args.examples}: not UTF-8 text ({e.reason} at byte {e.start})")
    src = None
    if any("scores" not in r for r in rows):
        from .sources import LocalSources

        src = LocalSources(nli_dir=args.nli_dir, nli2=args.nli2, hf_cache=args.hf_cache, device=args.device)
    try:
        doc = fit_policy(
            rows,
            profiles=profiles,
            sources=src,
            default_profile=args.default_profile,
            version=args.version_tag,
            data_licences=args.data_licences,
            folds=args.folds,
            alphas=alphas,
        )
    except FitError as e:
        print(f"fit failed: {e}", file=sys.stderr)
        return 1
    try:
        h = write_policy(doc, args.out)
    except OSError as e:
        print(f"fit failed: cannot write {args.out}: {e.strerror or e}", file=sys.stderr)
        return 1
    summary = {
        "policy": args.out,
        "sha256": h,
        "default_profile": doc["default_profile"],
        "profiles": {
            n: {
                "sources": p["sources"],
                "cutoff": p["cutoff"],
                "accept_margin": p["accept_margin"],
                "certified_alpha": p["certified"].get("alpha"),
                "oof_accepts": p["preregistration"]["accepted_oof"],
                "oof_false_accepts": p["preregistration"]["false_accepts_oof"],
            }
            for n, p in doc["profiles"].items()
        },
        "load_with": f"roman_wheel_node.load_policy({args.out!r}, expected_sha256={h!r})",
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
