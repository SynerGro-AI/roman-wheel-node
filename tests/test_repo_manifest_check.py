"""Repo-only (needs phase5/ and results/, which the sdist leaves out): the maintainer script
``python phase5/check_manifest_from_data.py`` keeps the exit codes README.md documents.

The COVID-Fact cache is internal and never distributed, so the 0 path ("manifest reproduces from data") cannot run
here. These tests cover the other two with no COVID-Fact data at all: 2 for a missing, unreadable or malformed cache,
and 1 ("FAILED") for a well-formed cache whose items are invented and so cannot reproduce the committed manifest."""

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "phase5", "check_manifest_from_data.py")
CMD = "python phase5/check_manifest_from_data.py"


def _run(*args):
    p = subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, cwd=ROOT, timeout=120)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def test_missing_cache_exits_2(tmp_path):
    rc, out, err = _run("--items", str(tmp_path / "no-such-items.json"))
    assert (rc, out) == (2, ""), err
    assert "missing" in err and "Traceback" not in err


def test_unreadable_or_malformed_cache_exits_2(tmp_path):
    cases = {
        "bad-utf8.json": b"\xff\xfe[",
        "bad-json.json": b"[{",
        "deep.json": b"[" * 200000,
        "not-a-list.json": b'{"ds": "covidfact"}',
        "no-ds.json": b'[{"claim": "x"}]',
        "no-units.json": b'[{"ds": "covidfact", "claim": "x", "label3": "SUPPORT", "evidence": "e"}]',
    }
    for name, data in cases.items():
        f = tmp_path / name
        f.write_bytes(data)
        rc, out, err = _run("--items", str(f))
        assert (rc, out) == (2, ""), (name, err[-400:])
        assert "Traceback" not in err, name


def test_cache_that_does_not_reproduce_exits_1_and_says_failed(tmp_path):
    items = [  # invented text, not COVID-Fact: a well-formed cache that cannot match the committed manifest
        {
            "ds": "covidfact",
            "claim": f"Invented claim number {i}.",
            "label3": "SUPPORT" if i % 2 else "REFUTE",
            "evidence": f"Invented evidence {i // 2}.",
            "units": [f"Invented evidence {i // 2}."],
        }
        for i in range(8)
    ] + [{"ds": "other", "claim": "ignored"}]
    f = tmp_path / "items.json"
    f.write_text(json.dumps(items), encoding="utf-8")
    rc, out, err = _run("--items", str(f))
    assert rc == 1, err[-400:]
    assert out.strip() == "FAILED"
    assert "manifest sha256 does not reproduce" in err and "Traceback" not in err


def test_readme_gives_the_command_its_output_and_exit_codes():
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    assert f"`{CMD}`" in readme
    i = readme.index(f"`{CMD}`")
    near = readme[i : i + 900]
    assert "`manifest reproduces from data`" in near
    for code, what in (("0", "reproduces"), ("1", "FAILED"), ("2", "cache")):
        assert re.search(rf"exit(s)? {code}\b[^.]*{what}|{what}[^.]*exit(s)? {code}\b", near), (code, near)
