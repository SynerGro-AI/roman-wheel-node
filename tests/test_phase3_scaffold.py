"""Phase 3 scaffolding: interfaces work offline (no network, no key, no models)."""
import hashlib, json, os, sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase3"))
from factcheck.interfaces import NLIScorer, OverlapScorer, Passage, XAIJudge, exact_copy_gate  # noqa: E402


def test_exact_copy_gate():
    p = Passage("d1", "Aspirin inhibits COX-1.")
    corpus = {p.sha256}
    assert exact_copy_gate([p], corpus) == 1
    assert exact_copy_gate([Passage("d1", "Aspirin inhibits COX-1")], corpus) == 0     # one byte off
    assert exact_copy_gate([], corpus) == 0


def test_overlap_scorer_range_and_order():
    s = OverlapScorer()
    ev = [Passage("d", "Aspirin irreversibly inhibits cyclooxygenase enzymes.")]
    hi, lo = s.score("aspirin inhibits cyclooxygenase", ev), s.score("vitamin d prevents fractures", ev)
    assert 0.05 - 1e-9 <= lo < hi <= 0.95 + 1e-9 and s.score("", ev) == 0.5


def test_judge_uses_cache_without_network_or_key(tmp_path, monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    j = XAIJudge(model="m", cache_path=str(tmp_path / "c.jsonl"))
    ev = [Passage("d", "x")]
    key = hashlib.sha256(f"m\n{j._prompt('c', ev)}".encode()).hexdigest()
    (tmp_path / "c.jsonl").write_text(json.dumps({"key": key, "model": "m", "p": 0.73, "raw": "{}"}) + "\n")
    j2 = XAIJudge(model="m", cache_path=str(tmp_path / "c.jsonl"))
    assert j2.score("c", ev) == 0.73
    with pytest.raises(RuntimeError):
        j2.score("uncached claim", ev)                 # no key -> refuses rather than calling out


def test_stubs_are_explicit():
    with pytest.raises(NotImplementedError):
        NLIScorer("nowhere").score("c", [])
