"""FROZEN reference copy of Roman Wheel Node v2 (pre-package). Used only by tests to prove the
geomean/product paths of the current package are bit-identical. Do not edit."""
"""Roman Wheel Node v2: exact-copy gate + live, learning source weights.

- Empty weights -> BLOCK (no evidence, no entry).
- Each source has a live reliability r = alpha / (alpha + beta) (a Beta mean).
- Feedback only ADDS to alpha/beta and is appended to a JSONL ledger.
  Nothing is ever overwritten; state is rebuilt by replaying the ledger.
- Claim scores are combined with a reliability-weighted geometric mean,
  so trusted sources pull harder, but any score of 0 still hard-BLOCKs.
"""
import json, math, os, time

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def opcode(eff):
    if eff >= 0.85: return "111", "ACCEPT"
    if eff >= 0.65: return "110", "ACCEPT"
    if eff >= 0.35: return "101", "ACCEPT"
    if eff > 0.05:  return "100", "HEDGE"
    return "000", "BLOCK"


class LiveWeights:
    def __init__(self, ledger_path="weights_ledger.jsonl", prior=(1.0, 1.0), half_life=None):
        """half_life: number of observations of a source after which old evidence
        counts half. None = never forget. The ledger is never edited either way;
        decay is applied while replaying, so any half_life can be re-run later."""
        self.path = ledger_path
        self.prior = prior
        self.decay = 0.5 ** (1.0 / half_life) if half_life else 1.0
        self.state = {}  # source -> [alpha, beta]
        if os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    if line.strip():
                        self._apply(json.loads(line))

    def _apply(self, ev):
        a, b = self.state.setdefault(ev["source"], list(self.prior))
        pa, pb = self.prior
        a = pa + (a - pa) * self.decay
        b = pb + (b - pb) * self.decay
        self.state[ev["source"]] = [a + ev["d_alpha"], b + ev["d_beta"]]

    def reliability(self, source):
        a, b = self.state.get(source, self.prior)
        return a / (a + b)

    def observe(self, source, outcome, strength=1.0, note=""):
        """outcome in [0,1]: 1 = source was right, 0 = wrong. Append-only."""
        o, s = clamp01(outcome), max(0.0, float(strength))
        ev = {"t": time.time(), "source": source, "outcome": o,
              "strength": s, "d_alpha": o * s, "d_beta": (1 - o) * s,
              "note": note}
        with open(self.path, "a") as f:
            f.write(json.dumps(ev) + "\n")
        self._apply(ev)
        return self.reliability(source)


def roman_wheel_node(a, scores, live=None, mode="geomean", log_path=None, flip=False):
    """
    a: 1 (exact copy) or 0
    scores: dict {source: score} (or a plain list, treated as equal-trust sources)
    live: LiveWeights or None (None = every source trusted equally)
    flip: True = invert sources the ledger shows are usually wrong
    Note: a raw score of exactly 0 is still a hard veto, even from a flipped source.
    Returns: (eff, opcode, status)
    """
    if isinstance(scores, (list, tuple)):
        scores = {f"s{i}": w for i, w in enumerate(scores)}
    if a <= 0 or not scores:
        result = (0.0, "000", "BLOCK")          # gate fail or empty-weights fix
    else:
        ws = {k: clamp01(v) for k, v in scores.items()}
        if any(v == 0.0 for v in ws.values()):
            result = (0.0, "000", "BLOCK")
        else:
            rel = {k: (live.reliability(k) if live else 1.0) for k in ws}
            if flip:
                # r < 0.5 means the source is reliably WRONG: invert what it says.
                # Influence = how far from a coin flip it is: |2r - 1|.
                use = {k: (1.0 - ws[k] if rel[k] < 0.5 else ws[k]) for k in ws}
                use = {k: max(1e-3, v) for k, v in use.items()}
                inf = {k: abs(2 * rel[k] - 1) for k in ws}
            else:
                use, inf = ws, rel
            if sum(inf.values()) < 1e-9:
                result = (0.0, "000", "BLOCK")   # every source is pure noise
                inf = None
            if inf is not None and mode == "product":
                eff = math.prod(use[k] ** inf[k] for k in ws)
            elif inf is not None:
                eff = math.exp(sum(inf[k] * math.log(use[k]) for k in ws) / sum(inf.values()))
            if inf is not None:
                eff = round(min(1.0, eff), 3)
                result = (eff, *opcode(eff))
    if log_path:
        with open(log_path, "a") as f:
            f.write(json.dumps({"t": time.time(), "a": a, "scores": scores,
                                "reliability": {k: live.reliability(k) for k in scores} if live else None,
                                "eff": result[0], "opcode": result[1], "status": result[2]}) + "\n")
    return result


if __name__ == "__main__":
    for a, ws in [(1, [0.9, 0.92, 0.88]), (1, [0.7] * 4), (0, [0.95, 0.9]),
                  (1, [0.8, 0.0, 0.9]), (1, [1.5, 0.9]), (1, [])]:
        print(" ", a, ws, roman_wheel_node(a, ws))
