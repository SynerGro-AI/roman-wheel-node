"""Repo-only: refit policy 4.3.0 from SciFact data built by scripts/fetch_test_data.py (not shipped in the sdist).

Without the data these tests fail (they never skip) and print the command that builds it.
"""

import json
import os
import subprocess
import sys

import pytest

from roman_wheel_node.policy import load_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "phase6"))
import fetch_test_data as FT  # noqa: E402

DATA = FT.default_dest()


def _fetch_command() -> str:
    return FT.COMMAND + (f" --dest {DATA}" if os.environ.get("RWN_TEST_DATA") else "")


def _require_data() -> str:
    problems = FT.verify_installed(DATA)
    if problems:
        pytest.fail("SciFact test data not ready: " + "; ".join(problems) + f"\nrun: {_fetch_command()}")
    return os.path.join(DATA, FT.SUBDIR)


def test_shipped_policy_is_reproduced_by_the_fit_api():
    """Refit 4.3.0 with the public fit API from the pinned SciFact data; it must equal the shipped policy."""
    step2 = _require_data()
    import refit_policy_43 as RF

    pool, test = RF.load_rows(step2)
    assert (len(pool), len(test)) == (799, 300)
    doc = RF.build_doc(pool)
    shipped = load_policy()
    strip = ("fitted_with", "sha256")
    assert {k: v for k, v in doc.items() if k not in strip} == {k: v for k, v in shipped.doc.items() if k not in strip}


def test_missing_data_fails_with_the_fetch_command(tmp_path):
    env = dict(os.environ, RWN_TEST_DATA=str(tmp_path / "empty"))
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-rs",
            "-p",
            "no:cacheprovider",
            "-W",
            "error",
            f"{__file__}::test_shipped_policy_is_reproduced_by_the_fit_api",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert "1 failed" in out and "SKIPPED" not in out and "skipped" not in out, out
    assert f"run: python scripts/fetch_test_data.py --dest {tmp_path / 'empty'}" in out, out


def test_tampered_or_stale_data_is_rejected(tmp_path):
    with open(FT.FEATURES, "rb") as f:
        features = f.read()
    with pytest.raises(FT.FetchError, match="scifact_step2_features.json: sha256"):
        FT.build(b"", features.replace(b'"overlap": 0.05', b'"overlap": 0.06', 1))
    d = tmp_path / FT.SUBDIR
    d.mkdir()
    (d / "items.json").write_text("[]\n", encoding="utf-8")
    problems = FT.verify_installed(str(tmp_path))
    assert any("items.json: sha256" in p for p in problems) and any(
        "missing" in p and "local.json" in p for p in problems
    )
    meta = json.loads(features.decode("utf-8"))
    assert meta["source"]["sha256"] == FT.SCIFACT_SHA256 and {r["id"][:3] for r in meta["rows"]} == {"sf:"}
