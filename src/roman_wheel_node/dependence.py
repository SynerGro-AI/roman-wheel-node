"""Correlation-aware evidence fusion: learn how sources depend on each other, and stop double-counting.

Naive Bayes (``variant="hard"``) adds one log-likelihood ratio per source, as if sources were independent given
the truth. Sources that copy each other then count several times. :class:`DependenceLearner` learns the
dependence from feedback, using the same append-only, replayable ledger design as :class:`LiveWeights`.

Every source's *correctness* on a resolved claim is ``c_k = [vote_k == truth]`` (vote = score > 0.5). Under
independence, the correctness indicators of two sources are independent, so any association between them is
duplicated evidence. Three combiners are built on that idea (``roman_wheel_node(mode="bayes", variant=...,
dependence=learner)``):

* ``"ess"`` (**recommended**: the best accuracy and calibration in the 20-seed dependence benchmark): an
  effective-sample-size discount. Each source's naive-Bayes vote ``logit(r_k)`` is weighted by
  ``1 / (1 + Σ_j max(0, φ_jk))`` over its *significant* partners (φ = correlation of correctness; significance by
  the G-test below). A cluster of m near-duplicates therefore counts about once. With no significant pairs it is
  *exactly* ``variant="hard"`` on the learner's own reliabilities.
* ``"tan"``: **tree-augmented naive Bayes** in correctness space.
  A Chow-Liu maximum-spanning *forest* is fitted over the sources' correctness indicators, keeping only edges
  whose dependence is significant (G-test, ``2·N·MI > 6.63``, i.e. p < 0.01 with 1 dof). Each source's likelihood
  is then conditioned on its tree parent: ``P(c_k | c_parent)``, shrunk toward the marginal reliability with
  ``shrink`` pseudo-counts. With no significant edges it reduces *exactly* to ``variant="hard"`` on the
  learner's own reliabilities.
* ``"ess_cal"``: the ESS discount applied to *calibrated confidences* instead of hard votes, for sources whose
  scores are calibrated posteriors ``P(true | score_k)`` (e.g. isotonic-calibrated model scores). With the learned base
  rate π, the log-odds is ``logit(π) + Σ_k w_k (logit(s_k) - logit(π))``: naive Bayes on calibrated posteriors, with
  each source's evidence down-weighted by its ESS weight ``w_k``. It keeps the confidence that hard votes throw away.
* ``"stack"``: stacking. An online logistic regression on the signed votes, with an L2 pull toward the naive-Bayes
  weights ``logit(r_k)``. A constant learning rate gives exponential forgetting.

All state is derived by replaying the ledger. Decay (``half_life``, in claims) is applied during replay, so the
ledger itself is never edited. :meth:`DependenceLearner.model` returns a JSON-serialisable snapshot of the exact
parameters used for one claim, and :class:`FrozenDependence` recomputes the same LLR from that snapshot
(that is how :class:`~roman_wheel_node.trace.ClaimTrace` replays decisions).
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = ["DependenceLearner", "FrozenDependence", "DEP_VARIANTS", "votes_of", "llr_from_model"]

DEP_VARIANTS = ("tan", "ess", "stack", "ess_cal")
_P_MIN, _P_MAX = 0.01, 0.99
G_THRESHOLD = 6.635  # chi-square(1) 0.99 quantile


def _clip(p: float) -> float:
    return min(_P_MAX, max(_P_MIN, p))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def votes_of(scores: Mapping[str, float]) -> Dict[str, int]:
    """Signed hard votes: +1 if score > 0.5, -1 if < 0.5, 0 (abstain) if exactly 0.5. Scores are clamped first."""
    out = {}
    for k, v in scores.items():
        s = max(0.0, min(1.0, float(v)))
        out[str(k)] = 1 if s > 0.5 else (-1 if s < 0.5 else 0)
    return out


def llr_from_model(model: dict, votes: Mapping[str, int], scores: Optional[Mapping[str, float]] = None) -> float:
    """Log-likelihood ratio of 'true' vs 'false' for these votes (or, for ``ess_cal``, scores) under a model
    snapshot."""
    v = model["variant"]
    if v == "ess_cal":
        if scores is None:
            raise ValueError("variant 'ess_cal' needs the scores")
        lp = _logit(model["prior"])
        return lp + math.fsum(
            model["w"][k] * (_logit(_clip(float(scores[k]))) - lp) for k in sorted(model["w"]) if votes.get(k, 0)
        )
    if v == "stack":
        return model["bias"] + math.fsum(model["w"][k] * votes[k] for k in sorted(model["w"]) if votes.get(k, 0))
    if v == "ess":
        return math.fsum(
            model["w"][k] * votes[k] * _logit(model["r"][k]) for k in sorted(model["w"]) if votes.get(k, 0)
        )
    if v == "tan":
        terms = []
        for f in model["factors"]:
            k, par = f["source"], f["parent"]
            vk = votes.get(k, 0)
            if not vk:
                continue
            c_t = 1 if vk > 0 else 0  # correctness of k if the claim is true
            if par is None or not votes.get(par, 0):
                p1_t = p1_f = f["r"]
            else:
                cp_t = 1 if votes[par] > 0 else 0
                p1_t = f["p1_given_parent"][cp_t]
                p1_f = f["p1_given_parent"][1 - cp_t]
            pt = p1_t if c_t else 1 - p1_t
            pf = p1_f if (1 - c_t) else 1 - p1_f
            terms.append(math.log(pt) - math.log(pf))
        return math.fsum(terms)
    raise ValueError(f"unknown dependence variant {v!r}")


class FrozenDependence:
    """Replays a logged model snapshot (duck-types the ``dependence=`` argument of roman_wheel_node)."""

    def __init__(self, model: dict) -> None:
        self._model = model

    def model(self, sources: Iterable[str], variant: str) -> dict:
        if self._model["variant"] != variant:
            raise ValueError("snapshot variant mismatch")
        return self._model

    def llr(self, scores: Mapping[str, float], variant: str) -> float:
        return llr_from_model(self.model((), variant), votes_of(scores), scores)


class DependenceLearner:
    """Learns source reliabilities *and* pairwise dependence from resolved claims (append-only ledger).

    Args:
        ledger_path: JSONL ledger of ``{"votes": {source: +1/-1/0}, "truth": bool}`` events; replayed on construction.
        half_life: claims after which old evidence counts half (None = never forget). Default 150, matching LiveWeights.
        shrink: pseudo-counts pulling conditional tables toward the marginal reliability (TAN).
        g_threshold: G-statistic needed to treat a pair as dependent (default 6.635, p < 0.01).
        stack_lr, stack_l2: learning rate and L2 pull toward naive-Bayes weights for ``variant="stack"``.
    """

    def __init__(
        self,
        ledger_path: str = "dependence_ledger.jsonl",
        half_life: Optional[float] = 150,
        shrink: float = 5.0,
        g_threshold: float = G_THRESHOLD,
        stack_lr: float = 0.05,
        stack_l2: float = 0.02,
    ) -> None:
        self.path = ledger_path
        self.half_life = half_life
        self.growth = 2.0 ** (1.0 / half_life) if half_life else 1.0
        self.shrink, self.g_threshold = shrink, g_threshold
        self.stack_lr, self.stack_l2 = stack_lr, stack_l2
        self._w = 1.0  # weight of the next event (growth trick)
        self._single: Dict[str, List[float]] = {}  # source -> [wrong, right] (scaled)
        self._pair: Dict[Tuple[str, str], List[List[float]]] = {}  # (a, b), a < b -> [[c_a][c_b]]
        self._sw: Dict[str, float] = {}  # stacking weights
        self._sb = 0.0
        self.n = 0
        self._truth = [0.0, 0.0]  # decayed [false, true] counts (base rate for ess_cal)
        self._cache: Dict[Tuple[Tuple[str, ...], str], dict] = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self._apply(json.loads(line))

    # ---- learning --------------------------------------------------------------------------------
    def record(self, scores: Mapping[str, float], truth: bool, note: str = "") -> None:
        """Append one resolved claim (its source scores and the truth). Append-only."""
        ev = {"t": time.time(), "votes": votes_of(scores), "truth": bool(truth), "note": note}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, sort_keys=True) + "\n")
            f.flush()
        self._apply(ev)

    def _apply(self, ev: dict) -> None:
        votes = {k: int(v) for k, v in ev["votes"].items()}
        y = 1 if ev["truth"] else 0
        # stacking: predict with the current state, then one SGD step (pure function of history)
        live = sorted(k for k, v in votes.items() if v)
        for k in live:
            self._sw.setdefault(k, _logit(_clip(self.reliability(k))))
        z = self._sb + math.fsum(self._sw[k] * votes[k] for k in live)
        p = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))
        g = p - y
        for k in live:
            nb = _logit(_clip(self.reliability(k)))
            self._sw[k] -= self.stack_lr * (g * votes[k] + self.stack_l2 * (self._sw[k] - nb))
        self._sb -= self.stack_lr * g
        # counts (decayed by the growth trick)
        corr = {k: (1 if (votes[k] > 0) == bool(y) else 0) for k in live}
        w = self._w
        self._truth[y] += w
        for k in live:
            self._single.setdefault(k, [0.0, 0.0])[corr[k]] += w
        for i, a in enumerate(live):
            for b in live[i + 1 :]:
                t = self._pair.setdefault((a, b), [[0.0, 0.0], [0.0, 0.0]])
                t[corr[a]][corr[b]] += w
        self._w *= self.growth
        self.n += 1
        if self._w > 1e150:
            s = 1.0 / self._w
            self._truth = [self._truth[0] * s, self._truth[1] * s]
            for v in self._single.values():
                v[0] *= s
                v[1] *= s
            for t in self._pair.values():
                for row in t:
                    row[0] *= s
                    row[1] *= s
            self._w = 1.0
        self._cache.clear()

    # ---- read-out ----------------------------------------------------------------------------------
    def _scale(self) -> float:
        return self.growth / self._w  # stored -> decayed counts (latest event = 1)

    def reliability(self, source: str) -> float:
        """Decayed Beta(1,1)-smoothed P(source is right)."""
        wr = self._single.get(source)
        if wr is None:
            return 0.5
        s = self._scale()
        return (wr[1] * s + 1.0) / ((wr[0] + wr[1]) * s + 2.0)

    def base_rate(self) -> float:
        """Decayed Beta(1,1)-smoothed P(truth) over the ledger."""
        s = self._scale() if self.n else 1.0
        return (self._truth[1] * s + 1.0) / ((self._truth[0] + self._truth[1]) * s + 2.0)

    def pair_table(self, a: str, b: str) -> List[List[float]]:
        """Decayed joint counts [c_a][c_b] of correctness for sources a, b."""
        key, flip = ((a, b), False) if a < b else ((b, a), True)
        t = self._pair.get(key)
        if t is None:
            return [[0.0, 0.0], [0.0, 0.0]]
        s = self._scale()
        t = [[t[0][0] * s, t[0][1] * s], [t[1][0] * s, t[1][1] * s]]
        return [[t[0][0], t[1][0]], [t[0][1], t[1][1]]] if flip else t

    def dependence(self, a: str, b: str) -> Tuple[float, float, float]:
        """(mutual information in nats, G statistic, phi correlation) of the two correctness indicators."""
        t = self.pair_table(a, b)
        # math.fsum (exactly rounded) instead of sum(): builtin float sum() changed in Python 3.12, and
        # these values gate edges near the threshold, so they must be identical on every Python version.
        t = [[x + 0.5 for x in row] for row in t]  # Jeffreys smoothing
        n = math.fsum((t[0][0], t[0][1], t[1][0], t[1][1]))
        ra, rb = math.fsum(t[0]), math.fsum(t[1])
        ca, cb = t[0][0] + t[1][0], t[0][1] + t[1][1]
        pa = [ra / n, rb / n]
        pb = [ca / n, cb / n]
        mi = math.fsum((t[i][j] / n) * math.log((t[i][j] / n) / (pa[i] * pb[j])) for i in (0, 1) for j in (0, 1))
        phi = (t[1][1] * t[0][0] - t[1][0] * t[0][1]) / math.sqrt(ra * rb * ca * cb)
        return max(0.0, mi), 2.0 * (n - 2.0) * max(0.0, mi), phi

    def model(self, sources: Iterable[str], variant: str = "ess") -> dict:
        """JSON-serialisable snapshot of the parameters used for a claim with these (voting) sources."""
        ks = tuple(sorted(set(sources)))
        key = (ks, variant)
        if key in self._cache:
            return self._cache[key]
        r = {k: _clip(self.reliability(k)) for k in ks}
        if variant == "stack":
            m = {"variant": "stack", "bias": self._sb, "w": {k: self._sw.get(k, _logit(r[k])) for k in ks}}
        else:
            edges = []
            for i, a in enumerate(ks):
                for b in ks[i + 1 :]:
                    mi, g, phi = self.dependence(a, b)
                    if g > self.g_threshold:
                        edges.append((mi, a, b, phi))
            if variant in ("ess", "ess_cal"):
                excess = {k: 0.0 for k in ks}
                for _, a, b, phi in edges:
                    excess[a] += max(0.0, phi)
                    excess[b] += max(0.0, phi)
                if variant == "ess":
                    m = {"variant": "ess", "r": r, "w": {k: 1.0 / (1.0 + excess[k]) for k in ks}}
                else:
                    m = {
                        "variant": "ess_cal",
                        "prior": _clip(self.base_rate()),
                        "w": {k: 1.0 / (1.0 + excess[k]) for k in ks},
                    }
            elif variant == "tan":
                m = {"variant": "tan", "factors": self._forest(ks, r, edges)}
            else:
                raise ValueError(f"unknown dependence variant {variant!r}; choose from {DEP_VARIANTS}")
        self._cache[key] = m
        return m

    def _forest(self, ks: Sequence[str], r: Dict[str, float], edges: list) -> List[dict]:
        """Chow-Liu maximum-spanning forest over significant edges (Kruskal, deterministic tie-break)."""
        parent_uf = {k: k for k in ks}

        def find(x):
            while parent_uf[x] != x:
                parent_uf[x] = parent_uf[parent_uf[x]]
                x = parent_uf[x]
            return x

        adj: Dict[str, List[str]] = {k: [] for k in ks}
        for mi, a, b, _ in sorted(edges, key=lambda e: (-e[0], e[1], e[2])):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent_uf[ra] = rb
                adj[a].append(b)
                adj[b].append(a)
        factors, seen = [], set()
        for root in ks:  # orient each tree from its smallest-named node
            if root in seen:
                continue
            seen.add(root)
            factors.append({"source": root, "parent": None, "r": r[root]})
            queue = [root]
            while queue:
                u = queue.pop(0)
                for v in sorted(adj[u]):
                    if v in seen:
                        continue
                    seen.add(v)
                    queue.append(v)
                    t = self.pair_table(u, v)  # [c_u][c_v]
                    p1 = [_clip((t[b][1] + self.shrink * r[v]) / (t[b][0] + t[b][1] + self.shrink)) for b in (0, 1)]
                    factors.append({"source": v, "parent": u, "r": r[v], "p1_given_parent": p1})
        return factors

    def llr(self, scores: Mapping[str, float], variant: str = "ess") -> float:
        """Log-likelihood ratio for a claim's scores (abstaining sources ignored)."""
        votes = votes_of(scores)
        return llr_from_model(self.model([k for k, v in votes.items() if v], variant), votes, scores)
