"""1.2.1 F1-F4: the fit CLI and API reject bad fold counts and bad input with clean errors, never a traceback.

Data are fixed and deterministic (a Weyl sequence for the scores; no random generator). Real subprocess runs of
``python -m roman_wheel_node.fit``; no mocks."""

import json
import os
import subprocess
import sys

import pytest

from roman_wheel_node import fit as FIT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rows(n=60, positives=None):
    """n scored examples. By default labels alternate SUPPORT/NEI; ``positives`` gives the SUPPORT indices."""
    pos = set(range(0, n, 2)) if positives is None else set(positives)
    out = []
    for i in range(n):
        u = (i * 0.6180339887498949) % 1.0
        sup = i in pos
        s = 0.35 + 0.65 * u if sup else 0.6 * u
        out.append(
            {"id": f"r{i}", "claim": f"claim {i}", "evidence": ["e."], "label": "SUPPORT" if sup else "NEI",
             "scores": {"overlap": {"S": s}}}
        )  # fmt: skip
    return out


def write(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for r in items:
            f.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
    return str(path)


def cli(*args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "src"), os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run(
        [sys.executable, "-m", "roman_wheel_node.fit", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=120,
    )
    return p.returncode, p.stderr


@pytest.mark.parametrize("folds", ["1", "0", "-3"])
def test_cli_folds_below_two_is_a_usage_error(tmp_path, folds):
    src = write(tmp_path / "ex.jsonl", rows())
    rc, err = cli(src, "-o", str(tmp_path / "p.json"), "--profile", "lex=overlap", "--folds", folds)
    assert rc == 2 and "Traceback" not in err, err
    assert f"--folds {folds}: must be at least 2" in err and err.startswith("usage:")
    assert not (tmp_path / "p.json").exists()


@pytest.mark.parametrize("folds", [1, 0, True, 2.0])
def test_api_rejects_folds_below_two(folds):
    with pytest.raises(FIT.FitError, match="folds must be an integer >= 2"):
        FIT.fit_policy(rows(), profiles={"lex": ("overlap",)}, folds=folds)
    with pytest.raises(FIT.FitError, match="folds must be an integer >= 2"):
        FIT.fit_profile(rows(), ("overlap",), folds=folds)


def test_every_fold_needs_both_classes():
    # 12 SUPPORT out of 60 overall (enough by the old global count), but all at indices = 0 mod 5, so with 5 folds
    # fold 1 is all SUPPORT and folds 2-5 have none.
    bad = rows(60, positives=range(0, 60, 5))
    with pytest.raises(FIT.FitError, match=r"both classes in every fold; fold 1 of 5 has only SUPPORT"):
        FIT.fit_profile(bad, ("overlap",), folds=5)
    FIT.fit_profile(rows(60), ("overlap",), folds=5)  # the same size, balanced per fold: fits


def test_cli_per_fold_failure_is_clean(tmp_path):
    src = write(tmp_path / "ex.jsonl", rows(60, positives=range(0, 60, 5)))
    rc, err = cli(src, "-o", str(tmp_path / "p.json"), "--profile", "lex=overlap")
    assert rc == 1 and "Traceback" not in err and "fit failed:" in err and "every fold" in err, err


@pytest.mark.parametrize(
    "content, expect",
    [
        (["{not json"], "not JSON"),
        ([{"id": "x", "claim": "c", "evidence": ["e."], "label": "MAYBE", "scores": {"overlap": {"S": 0.5}}}], "x"),
    ],
)
def test_cli_bad_examples_file_is_a_clean_error(tmp_path, content, expect):
    src = write(tmp_path / "ex.jsonl", content)
    rc, err = cli(src, "-o", str(tmp_path / "p.json"), "--profile", "lex=overlap")
    assert rc == 2 and "Traceback" not in err and err.startswith("usage:") and expect in err, err


def test_cli_missing_or_non_utf8_examples_file_is_a_clean_error(tmp_path):
    rc, err = cli(str(tmp_path / "absent.jsonl"), "-o", str(tmp_path / "p.json"))
    assert rc == 2 and "Traceback" not in err and "cannot read" in err, err
    latin1 = tmp_path / "latin1.jsonl"
    latin1.write_bytes('{"claim": "caf\u00e9", "evidence": ["e."], "label": "SUPPORT"}\n'.encode("latin-1"))
    rc, err = cli(str(latin1), "-o", str(tmp_path / "p.json"))
    assert rc == 2 and "Traceback" not in err and "not UTF-8 text" in err, err


def test_cli_unwritable_output_is_a_clean_error(tmp_path):
    src = write(tmp_path / "ex.jsonl", rows())
    rc, err = cli(src, "-o", os.path.join(src, "p.json"), "--profile", "lex=overlap")  # parent is a file
    assert rc == 1 and "Traceback" not in err and "cannot write" in err, err
