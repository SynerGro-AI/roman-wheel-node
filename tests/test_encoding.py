"""B1 (1.2.1): text I/O never depends on the locale encoding, so non-ASCII claims (≥, α, é) survive on Windows
(cp1252) and logs replay across operating systems.

The subprocess test runs the real package in a child Python whose locale encoding is NOT UTF-8: on Linux/macOS via
LC_ALL=C with UTF-8 mode and locale coercion switched off (the child then uses ASCII), on Windows natively (cp1252).
It asserts the child's encoding really is non-UTF-8, so it cannot pass vacuously."""

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from roman_wheel_node.trace import replay

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "roman_wheel_node"
CLAIM = "IL-6 ≥ 2 pg/mL in α-cells of the café cohort"
EVIDENCE = "In α-cells, IL-6 was ≥ 2 pg/mL across the café cohort."

CHILD = textwrap.dedent(
    """
    import json, locale, os, sys
    from roman_wheel_node.sources import OverlapScorer
    from roman_wheel_node.trace import ClaimTrace, replay
    from roman_wheel_node.shadow import _main as shadow_main
    from roman_wheel_node.fit import load_examples

    enc = locale.getencoding() if hasattr(locale, "getencoding") else locale.getpreferredencoding(False)
    enc = enc if not sys.flags.utf8_mode else "utf-8 (utf8 mode)"
    claim, ev, d = json.loads(sys.argv[1])  # ASCII-escaped: argv itself is locale-decoded
    t = ClaimTrace(os.path.join(d, "audit.jsonl"))
    sc = {"overlap": OverlapScorer().score(claim, [ev])}
    dec = t.submit(claim, sc, a=1)
    rep = replay(os.path.join(d, "audit.jsonl"))
    with open(os.path.join(d, "claims.jsonl"), "wb") as f:  # UTF-8 input file, as written on any other OS
        f.write((json.dumps({"claim": claim, "evidence": [ev]}, ensure_ascii=False) + "\\n").encode("utf-8"))
    with open(os.path.join(d, "corpus.jsonl"), "wb") as f:
        f.write((json.dumps({"doc_id": "d1", "sentences": [ev]}, ensure_ascii=False) + "\\n").encode("utf-8"))
    with open(os.path.join(d, "labels.jsonl"), "wb") as f:
        f.write((json.dumps({"claim": claim, "evidence": [ev], "label": "SUPPORT"}, ensure_ascii=False) + "\\n")
                .encode("utf-8"))
    rows = load_examples(os.path.join(d, "labels.jsonl"))
    rc = shadow_main([os.path.join(d, "claims.jsonl"), "--log-dir", os.path.join(d, "shadow"),
                      "--corpus", os.path.join(d, "corpus.jsonl")])
    print("RESULT " + json.dumps({"enc": enc, "status": dec.status, "n": rep.n_records, "rc": rc,
                                  "fit_claim": rows[0]["claim"]}), file=sys.stderr)
    """
)


def _child_env():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LC_", "PYTHONUTF8", "PYTHONIOENCODING"))}
    env.update(LC_ALL="C", LANG="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0")
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    return env


def test_non_ascii_claims_survive_a_non_utf8_locale(tmp_path):
    flags = ["-X", "warn_default_encoding", "-W", "error::EncodingWarning"] if sys.version_info >= (3, 10) else []
    p = subprocess.run(
        [sys.executable, *flags, "-c", CHILD, json.dumps([CLAIM, EVIDENCE, str(tmp_path)])],
        env=_child_env(),
        capture_output=True,
        timeout=120,
    )
    err = p.stderr.decode("utf-8", "replace")
    assert p.returncode == 0, err
    res = json.loads([ln for ln in err.splitlines() if ln.startswith("RESULT ")][-1][7:])
    assert res["enc"].replace("-", "").lower() not in ("utf8", "cp65001"), res  # the locale really is non-UTF-8
    assert res["n"] == 1 and res["rc"] == 0 and res["fit_claim"] == CLAIM
    # The child's logs are UTF-8 on disk and replay here (another process, the default test locale): portable.
    rep = replay(str(tmp_path / "audit.jsonl"))
    assert rep.n_records == 1
    rec = json.loads((tmp_path / "audit.jsonl").read_bytes().decode("utf-8"))
    assert rec["content"] == CLAIM
    fresh = json.loads((tmp_path / "shadow" / "fresh_claims.jsonl").read_bytes().decode("utf-8"))
    assert fresh["claim"] == CLAIM and not fresh["reason"].startswith("error:Unicode")
    # the shadow CLI echoed the item to stdout as UTF-8 even though the child's stdout encoding is ASCII
    assert json.loads(p.stdout.decode("utf-8").splitlines()[0])["claim"] == CLAIM


def _locale_dependent_io(path: Path):
    """builtin open, Path.open, read_text, write_text in text mode with no encoding=; subprocess text mode likewise."""
    out = []
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(n, ast.Call):
            continue
        kw = {k.arg for k in n.keywords}
        if "encoding" in kw or None in kw:
            continue
        f = n.func
        name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
        if name == "open" and not (isinstance(f, ast.Attribute) and getattr(f.value, "id", "") == "os"):
            idx = 1 if isinstance(f, ast.Name) else 0
            mode = next((k.value for k in n.keywords if k.arg == "mode"), n.args[idx] if len(n.args) > idx else None)
            if mode is None or (isinstance(mode, ast.Constant) and "b" not in str(mode.value)):
                out.append(f"{path.relative_to(ROOT)}:{n.lineno}")
        elif name == "read_text" and not n.args or name == "write_text" and len(n.args) < 2:
            out.append(f"{path.relative_to(ROOT)}:{n.lineno}")
        elif name in ("run", "Popen", "check_output") and any(
            k.arg in ("text", "universal_newlines") and getattr(k.value, "value", None) is True for k in n.keywords
        ):
            out.append(f"{path.relative_to(ROOT)}:{n.lineno}")
    return out


def test_package_has_no_locale_dependent_text_io():
    bad = [b for p in sorted(PKG.rglob("*.py")) for b in _locale_dependent_io(p)]
    assert not bad, f"text I/O without encoding='utf-8': {bad}"


def test_tests_have_no_locale_dependent_text_io():
    bad = [b for p in sorted((ROOT / "tests").rglob("*.py")) for b in _locale_dependent_io(p)]
    assert not bad, f"text I/O without encoding='utf-8': {bad}"
