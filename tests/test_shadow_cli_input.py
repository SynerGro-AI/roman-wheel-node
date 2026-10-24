"""1.2.1: the shadow CLI reads its claims stream robustly.

R1: a claim containing U+2028, U+2029 or U+0085 (JSON written with ensure_ascii=False carries them raw) is one claim,
not split into broken lines. Every run is the real ``python -m roman_wheel_node.shadow`` in a subprocess."""

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _shadow(tmp_path, data: bytes, *args):
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    p = subprocess.run(
        [sys.executable, "-m", "roman_wheel_node.shadow", "--log-dir", str(tmp_path / "d"), *args],
        input=data,
        capture_output=True,
        env=env,
        timeout=120,
    )
    return p.returncode, p.stdout.decode("utf-8"), p.stderr.decode("utf-8", "replace")


def test_stdin_claim_with_unicode_line_separators_is_one_claim(tmp_path):
    claim = "line\u2028sep and NEL\x85 and PS\u2029 caf\u00e9"
    lines = [json.dumps({"claim": claim, "evidence": ["x y"]}, ensure_ascii=False), json.dumps({"claim": "b."})]
    rc, out, err = _shadow(tmp_path, ("\r\n".join(lines) + "\n").encode("utf-8"), "-")
    assert rc == 0, err[-400:]
    assert "Traceback" not in err
    rows = [json.loads(x) for x in out.split("\n") if x]  # shadow mode echoes every claim on stdout
    assert [e["claim"] for e in rows] == [claim, "b."]
    assert '"counts": {"ACCEPT": 0, "NOT_SURE": 0, "BLOCK": 2}' in err  # the run summary goes to stderr
