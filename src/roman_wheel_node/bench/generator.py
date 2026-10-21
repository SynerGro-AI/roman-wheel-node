"""Synthetic claim-stream generator: the exact generator behind every published benchmark number.

Model (per claim i, with ``rng = random.Random(seed)``, consumed in exactly this order):

1. ``truth = rng.random() < base_rate`` (base_rate = 0.5 in every standard scenario).
2. For each source in declaration order:

   * independent source with accuracy ``acc(i)``: ``says_true = truth if rng.random() < acc(i) else not truth``.
   * follower in a :class:`CopyGroup`: with prob. ``copy_prob`` (one ``rng.random()``) it repeats the
     leader's call for this claim; otherwise it makes its own call with ``follower_accuracy`` (a second
     ``rng.random()``). The leader must be declared before its followers.
   * score noise: ``s = rng.betavariate(8, 2)`` if the source says "true", else ``rng.betavariate(2, 8)``;
     then ``s = max(0.01, s)``, because an exact 0 is a hard veto in the node.

3. Drift: ``Source(accuracy=a, drift=(k, b))`` has accuracy a for claims i < k and b from claim k on.

The oracle log-likelihood ratio ``log P(scores | true) / P(scores | false)`` is computed exactly from this
model (followers marginalise the leader's hidden call). ``oracle_llr > 0`` is the Bayes-optimal decision,
and its accuracy is the ceiling no policy can beat in expectation.

CLI::

    python -m roman_wheel_node.bench.generator --list
    python -m roman_wheel_node.bench.generator --scenario drift --seed 3 [--n 100] [--out stream.jsonl]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

__all__ = ["Source", "CopyGroup", "Scenario", "Claim", "SCENARIOS", "stream", "density_ratio", "oracle_llr"]

BETA_SAYS_TRUE = (8, 2)
BETA_SAYS_FALSE = (2, 8)
SCORE_FLOOR = 0.01


@dataclass(frozen=True)
class Source:
    """An evidence source. ``drift=(k, b)``: the accuracy switches to b from claim index k on."""
    name: str
    accuracy: float
    drift: Optional[Tuple[int, float]] = None

    def acc(self, i: int) -> float:
        if self.drift is not None and i >= self.drift[0]:
            return self.drift[1]
        return self.accuracy


@dataclass(frozen=True)
class CopyGroup:
    """``followers`` repeat ``leader``'s call w.p. ``copy_prob``, else call on their own at ``follower_accuracy``."""
    leader: str
    followers: Tuple[str, ...]
    copy_prob: float
    follower_accuracy: float


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    n: int
    sources: Tuple[Source, ...]
    copy: Optional[CopyGroup] = None
    base_rate: float = 0.5
    description: str = ""

    def true_accuracy(self, i: int) -> Dict[str, float]:
        """Marginal P(source's call is right) at claim i (for followers: copy-mixture marginal)."""
        acc = {s.name: s.acc(i) for s in self.sources}
        if self.copy:
            lead = acc[self.copy.leader]
            for f in self.copy.followers:
                acc[f] = self.copy.copy_prob * lead + (1 - self.copy.copy_prob) * self.copy.follower_accuracy
        return acc

    def spec(self) -> dict:
        return {"key": self.key, "title": self.title, "n": self.n, "base_rate": self.base_rate,
                "description": self.description,
                "sources": [{"name": s.name, "accuracy": s.accuracy, "drift": list(s.drift) if s.drift else None}
                            for s in self.sources],
                "copy": None if not self.copy else {"leader": self.copy.leader, "followers": list(self.copy.followers),
                                                    "copy_prob": self.copy.copy_prob,
                                                    "follower_accuracy": self.copy.follower_accuracy},
                "noise": {"says_true": list(BETA_SAYS_TRUE), "says_false": list(BETA_SAYS_FALSE), "floor": SCORE_FLOOR}}


@dataclass(frozen=True)
class Claim:
    i: int
    truth: bool
    scores: Dict[str, float]
    true_accuracy: Dict[str, float]
    oracle_llr: float


_FIVE = (Source("drone", .92), Source("trial", .80), Source("draw", .65), Source("harmony", .50), Source("instr", .25))

SCENARIOS: Dict[str, Scenario] = {s.key: s for s in [
    Scenario("baseline", "baseline", 3000, _FIVE,
             description="five independent sources from 92% down to 25% (a reliable liar)"),
    Scenario("drift", "drift (drone goes bad at 1500)", 3000,
             (Source("drone", .92, drift=(1500, .15)), Source("trial", .75), Source("draw", .65), Source("harmony", .50)),
             description="the best source turns into a liar halfway through"),
    Scenario("bad_majority", "bad majority", 3000,
             (Source("good", .85), Source("bad1", .30), Source("bad2", .30), Source("bad3", .35)),
             description="three of four sources are mostly wrong"),
    Scenario("all_noisy", "all noisy (8 weak sources)", 3000, tuple(Source(f"w{j}", .58) for j in range(8)),
             description="eight independent 58% sources: an information-limited case"),
    Scenario("cold_start", "cold start (first 50 claims)", 50, _FIVE,
             description="baseline sources, but only 50 claims to learn from"),
    Scenario("copycats", "copycats (3 copy a 65% leader)", 3000,
             (Source("lead", .65), Source("c1", .55), Source("c2", .55), Source("c3", .55), Source("solo", .80)),
             copy=CopyGroup("lead", ("c1", "c2", "c3"), 0.9, 0.55),
             description="three followers copy a 65% leader 90% of the time; one independent 80% source"),
]}


def _score(says_true: bool, rng: random.Random) -> float:
    return max(SCORE_FLOOR, rng.betavariate(*BETA_SAYS_TRUE) if says_true else rng.betavariate(*BETA_SAYS_FALSE))


def density_ratio(s: float) -> float:
    """Beta(8,2)/Beta(2,8) density ratio at s = (s/(1-s))^6: evidence that the source said 'true'."""
    s = min(1 - 1e-9, s)
    return (s / (1 - s)) ** 6


def oracle_llr(sc: Scenario, i: int, scores: Dict[str, float]) -> float:
    """Exact log P(scores | true) / P(scores | false) under the generating model."""
    followers = set(sc.copy.followers) if sc.copy else set()
    leader = sc.copy.leader if sc.copy else None
    z = 0.0
    for s in sc.sources:
        if s.name in followers or s.name == leader:
            continue
        a, x = s.acc(i), density_ratio(scores[s.name])
        z += math.log((a * x + (1 - a)) / (a + (1 - a) * x))
    if sc.copy:
        cg, la = sc.copy, next(s for s in sc.sources if s.name == leader).acc(i)

        def lik(truth: bool) -> float:
            tot = 0.0
            for lead_says in (True, False):
                p_lead = la if lead_says == truth else 1 - la
                h = density_ratio(scores[leader]) if lead_says else 1.0
                prod = 1.0
                for f in cg.followers:
                    p_own_true = cg.follower_accuracy if truth else 1 - cg.follower_accuracy
                    p_true = cg.copy_prob * (1.0 if lead_says else 0.0) + (1 - cg.copy_prob) * p_own_true
                    prod *= p_true * density_ratio(scores[f]) + (1 - p_true)
                tot += p_lead * h * prod
            return tot
        z += math.log(lik(True) / lik(False))
    return z


def stream(sc: Scenario, seed: int, n: Optional[int] = None) -> Iterator[Claim]:
    """Yield the claim stream for (scenario, seed). Identical across runs and platforms (Python's Mersenne Twister)."""
    rng = random.Random(seed)
    followers = set(sc.copy.followers) if sc.copy else set()
    for i in range(sc.n if n is None else n):
        truth = rng.random() < sc.base_rate
        says: Dict[str, bool] = {}
        scores: Dict[str, float] = {}
        for s in sc.sources:
            if s.name in followers:
                if rng.random() < sc.copy.copy_prob:          # type: ignore[union-attr]
                    v = says[sc.copy.leader]                  # type: ignore[union-attr]
                else:
                    v = truth if rng.random() < sc.copy.follower_accuracy else not truth  # type: ignore[union-attr]
            else:
                v = truth if rng.random() < s.acc(i) else not truth
            says[s.name] = v
            scores[s.name] = _score(v, rng)
        yield Claim(i, truth, scores, sc.true_accuracy(i), oracle_llr(sc, i, scores))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m roman_wheel_node.bench.generator",
                                description="Emit a synthetic claim stream as JSONL.")
    p.add_argument("--list", action="store_true", help="list standard scenarios and their specs")
    p.add_argument("--scenario", choices=sorted(SCENARIOS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=None, help="number of claims (default: the scenario's n)")
    p.add_argument("--out", default="-", help="output path, or - for stdout")
    a = p.parse_args(argv)
    if a.list:
        print(json.dumps({k: s.spec() for k, s in SCENARIOS.items()}, indent=1))
        return 0
    if not a.scenario:
        p.error("--scenario is required (or --list)")
    out = sys.stdout if a.out == "-" else open(a.out, "w")
    for c in stream(SCENARIOS[a.scenario], a.seed, a.n):
        out.write(json.dumps({"i": c.i, "truth": c.truth, "scores": c.scores, "true_accuracy": c.true_accuracy,
                              "oracle_llr": c.oracle_llr}, sort_keys=True) + "\n")
    if out is not sys.stdout:
        out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
