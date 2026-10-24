"""Repo-only: the full 20-seed benchmarks are byte-identical to the committed results/ (not shipped in the sdist).

About 1 min each on 8 cores. Excluded from the sdist by MANIFEST.in because results/ is repo-only.
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_full_benchmark_byte_identical_to_committed():
    chk = subprocess.run(
        [sys.executable, "-m", "roman_wheel_node.bench", "--seeds", "20", "--check", os.path.join(ROOT, "results")],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert chk.returncode == 0, chk.stdout + chk.stderr


def test_full_dependence_benchmark_byte_identical_to_committed():
    chk = subprocess.run(
        [
            sys.executable,
            "-m",
            "roman_wheel_node.bench",
            "--seeds",
            "20",
            "--suite",
            "dependence",
            "--check",
            os.path.join(ROOT, "results", "dependence"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert chk.returncode == 0, chk.stdout + chk.stderr
