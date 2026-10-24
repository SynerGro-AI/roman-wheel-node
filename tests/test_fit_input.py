"""1.2.1 R3: a deeply nested --examples line is a clean FitError / usage error, not a RecursionError traceback.
Real files, the real loader and the real CLI in a subprocess."""

import os
import subprocess
import sys

import pytest

from roman_wheel_node import fit as FIT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEEP = "[" * 200000  # deeper than json's C scanner (and the pure-Python one) can recurse


def test_deeply_nested_examples_line_is_a_fit_error(tmp_path):
    f = tmp_path / "ex.jsonl"
    f.write_text('{"claim": "x", "label": true}\n' + DEEP + "\n", encoding="utf-8")
    with pytest.raises(FIT.FitError, match=r"ex\.jsonl:2: not JSON \(RecursionError"):
        FIT.load_examples(str(f))


def test_deeply_nested_examples_line_is_a_cli_usage_error(tmp_path):
    f = tmp_path / "ex.jsonl"
    f.write_text(DEEP + "\n", encoding="utf-8")
    p = subprocess.run(
        [sys.executable, "-m", "roman_wheel_node.fit", str(f), "-o", str(tmp_path / "p.json")],
        capture_output=True,
        env=dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src")),
        timeout=120,
    )
    err = p.stderr.decode("utf-8", "replace")
    assert p.returncode == 2, err[-400:]
    assert "Traceback" not in err and "ex.jsonl:1: not JSON (RecursionError" in err


def test_too_few_examples_message_says_what_is_counted():
    """R4: the count-only check says how many examples the folds need; it does not claim to check classes."""
    rows = [{"claim": f"c{i}", "label": i % 2 == 0, "scores": {"overlap": {"S": 0.5}}} for i in range(12)]
    with pytest.raises(FIT.FitError) as e:
        FIT.fit_policy(rows, profiles={"lex": ("overlap",)}, folds=5)
    assert str(e.value) == "need at least 50 examples for 5-fold cross-fitting; got 12"
