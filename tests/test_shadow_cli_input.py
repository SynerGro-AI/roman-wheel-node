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


# R2: bad input is a clear usage error (exit 2) naming where it is, never a traceback.
BAD_INPUTS = [
    (b'{"claim": "a."}\n\xff\xfe not utf-8\n', "not UTF-8"),
    (b'{"claim": "a."}\n{"claim": \n', "line 2: not JSON"),
    (b'{"claim": "a."}\n[1, 2]\n', "line 2: expected a JSON object, got list"),
    (b'{"claim": "a."}\n' + b"[" * 200000 + b"\n", "line 2: not JSON"),
    # valid JSON, but a lone surrogate escape is not Unicode text and could never be echoed or logged as UTF-8
    (b'{"claim": "a."}\n{"claim": "x \\ud800 y"}\n', "line 2: not valid Unicode text (lone surrogate \\ud800"),
    (b'{"claim": "a.", "meta": {"k\\udfff": [1, ["\\udc00"]]}}\n', "line 1: not valid Unicode text"),
]


def _assert_usage_error(rc, err, what):
    assert rc == 2, err[-400:]
    assert "Traceback" not in err
    assert what in err


def test_bad_stdin_is_a_clear_error(tmp_path):
    for data, what in BAD_INPUTS:
        rc, _, err = _shadow(tmp_path, data, "-")
        _assert_usage_error(rc, err, "stdin: " + what)


def test_bad_claims_file_is_a_clear_error(tmp_path):
    for n, (data, what) in enumerate(BAD_INPUTS):
        f = tmp_path / f"claims{n}.jsonl"
        f.write_bytes(data)
        rc, _, err = _shadow(tmp_path, b"", str(f))
        _assert_usage_error(rc, err, f"{f}: {what}")


def test_bad_corpus_file_is_a_clear_error(tmp_path):
    claims = tmp_path / "claims.jsonl"
    claims.write_text('{"claim": "a.", "doc_ids": ["1"]}\n', encoding="utf-8")
    no_list = (b'{"doc_id": 1, "sentences": "not a list"}\n', "line 1: needs 'doc_id' and a 'sentences' list")
    cases = BAD_INPUTS + [no_list]
    for n, (data, what) in enumerate(cases):
        f = tmp_path / f"corpus{n}.jsonl"
        f.write_bytes(data)
        rc, _, err = _shadow(tmp_path, b"", "--corpus", str(f), str(claims))
        _assert_usage_error(rc, err, f"{f}: {what}")


def test_unreadable_claims_file_is_a_clear_error(tmp_path):
    rc, _, err = _shadow(tmp_path, b"", str(tmp_path / "no-such-file.jsonl"))
    _assert_usage_error(rc, err, "cannot read")


def test_closed_stdin_is_a_clear_error(tmp_path):
    """1.2.1: reading claims from '-' with file descriptor 0 closed (Python then sets sys.stdin to None) is a usage
    error, exit 2, not an AttributeError traceback. fd 0 is closed in the child before Python starts."""
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    cmd = [sys.executable, "-m", "roman_wheel_node.shadow", "--log-dir", str(tmp_path / "d"), "-"]
    p = subprocess.run(
        ["/bin/sh", "-c", 'exec "$@" 0<&-', "sh", *cmd] if os.name != "nt" else cmd,
        stdin=None if os.name != "nt" else subprocess.DEVNULL,
        capture_output=True,
        env=env,
        timeout=120,
    )
    err = p.stderr.decode("utf-8", "replace")
    if os.name == "nt":  # no portable way to start a child with fd 0 closed there; /dev/null is the empty stream
        assert p.returncode == 0 and "Traceback" not in err, err[-400:]
        return
    _assert_usage_error(p.returncode, err, "stdin is closed")


def test_paired_surrogate_escape_is_one_character_and_echoed(tmp_path):
    """A correctly paired escape (an emoji written as \\ud83d\\ude00) is valid text: accepted, echoed as UTF-8."""
    rc, out, err = _shadow(tmp_path, b'{"claim": "smile \\ud83d\\ude00."}\n', "-")
    assert rc == 0 and "Traceback" not in err, err[-400:]
    assert [json.loads(x)["claim"] for x in out.split("\n") if x] == ["smile \U0001f600."]
