"""`python -m roman_wheel_node` (or the `roman-wheel-node` command): version and the six reference calls.

With no arguments, or with `demo`, it prints the six original reference calls of `roman_wheel_node(a, scores)`.
Other entry points are their own modules: `python -m roman_wheel_node.shadow`, `.fit`, `.fresh` and
`.bench`.
"""

import argparse
import sys

from . import __version__
from .core import roman_wheel_node

REFERENCE_CALLS = [
    (1, [0.9, 0.92, 0.88]),
    (1, [0.7] * 4),
    (0, [0.95, 0.9]),
    (1, [0.8, 0.0, 0.9]),
    (1, [1.5, 0.9]),
    (1, []),
]


def build_parser():
    p = argparse.ArgumentParser(
        prog="roman-wheel-node",
        description="Roman Wheel Node: exact-copy gate plus learned, auditable evidence fusion.",
        epilog="Shadow mode: python -m roman_wheel_node.shadow --help. Fresh-set labelling and certification: "
        "python -m roman_wheel_node.fresh --help. Fit your own policy: python -m roman_wheel_node.fit --help. "
        "Synthetic benchmark: python -m roman_wheel_node.bench --help.",
    )
    p.add_argument("--version", action="version", version=f"roman-wheel-node {__version__}")
    p.add_argument(
        "command",
        nargs="?",
        default="demo",
        choices=["demo"],
        help="demo (default): print the six reference calls of roman_wheel_node(a, scores)",
    )
    return p


def main(argv=None):
    build_parser().parse_args(argv)
    for a, ws in REFERENCE_CALLS:
        print(" ", a, ws, roman_wheel_node(a, ws))
    return 0


if __name__ == "__main__":
    sys.exit(main())
