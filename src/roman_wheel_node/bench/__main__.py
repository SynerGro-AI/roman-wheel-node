"""CLI: python -m roman_wheel_node.bench --seeds 20 [--out results] [--check DIR] [--no-plots]"""
from __future__ import annotations

import argparse
import os
import sys

from .generator import SCENARIOS
from .runner import run, to_json, to_text


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m roman_wheel_node.bench",
                                description="Run the reproducible Roman Wheel Node benchmark.")
    p.add_argument("--seeds", type=int, default=20, help="seeds 0..N-1 per scenario (default 20)")
    p.add_argument("--scenarios", nargs="*", choices=sorted(SCENARIOS), help="subset (default: all six)")
    p.add_argument("--ceiling-claims", type=int, default=200_000, help="claims for the oracle ceiling estimate")
    p.add_argument("--jobs", type=int, default=None, help="worker processes (default: all cores; output is identical)")
    p.add_argument("--out", default="results", help="output directory (benchmark.json, benchmark.txt, charts)")
    p.add_argument("--no-plots", action="store_true", help="skip charts (matplotlib not needed)")
    p.add_argument("--check", metavar="DIR", help="rerun and byte-compare benchmark.json/.txt with DIR; writes nothing")
    a = p.parse_args(argv)
    res = run(a.seeds, a.scenarios, a.ceiling_claims, a.jobs)
    js, txt = to_json(res), to_text(res)
    if a.check:
        bad = [n for n, body in (("benchmark.json", js), ("benchmark.txt", txt))
               if not os.path.exists(os.path.join(a.check, n)) or open(os.path.join(a.check, n)).read() != body]
        print("IDENTICAL" if not bad else f"DIFFERENT: {', '.join(bad)}")
        return 1 if bad else 0
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "benchmark.json"), "w") as f:
        f.write(js)
    with open(os.path.join(a.out, "benchmark.txt"), "w") as f:
        f.write(txt)
    print(txt)
    if not a.no_plots:
        try:
            from .plots import plot_all
        except ImportError as e:  # pragma: no cover
            print(f"(charts skipped: {e}; pip install roman-wheel-node[bench])", file=sys.stderr)
        else:
            for path in plot_all(res, a.out):
                print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
