"""Phase 3 harness: interfaces and statistics work offline (no network, no key, no model download)."""

import hashlib
import json
import os
import sys

import pytest

if not os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase3")):
    pytest.skip("phase3/ research scripts are repo-only (not in the sdist/wheel)", allow_module_level=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase3"))
from factcheck.interfaces import (
    BM25Retriever,
    BudgetExceeded,
    OverlapScorer,
    Passage,
    SpendLedger,  # noqa: E402
    XAIJudge,
    exact_copy_gate,
    parse_judge,
    usage_cost_usd,
)
from factcheck.scifact import load_claims  # noqa: E402


def test_exact_copy_gate():
    p = Passage("d1", "Aspirin inhibits COX-1.")
    corpus = {p.sha256}
    assert exact_copy_gate([p], corpus) == 1
    assert exact_copy_gate([Passage("d1", "Aspirin inhibits COX-1")], corpus) == 0  # one byte off
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
        j2.score("uncached claim", ev)  # no key -> refuses rather than calling out


def test_bm25_ranks_the_matching_doc_first_and_is_deterministic():
    r = BM25Retriever(
        {"1": "aspirin inhibits cyclooxygenase", "2": "vitamin d and bone fractures", "3": "aspirin dosing"}
    )
    assert r.retrieve("does aspirin inhibit cyclooxygenase", 2) == ["1", "3"]
    assert r.retrieve("unrelated words zzz", 3) == []


def test_parse_judge_styles_and_fallback():
    p, q = parse_judge("A", '{"support": 0.7, "contradict": 0.1, "nei": 0.2, "quote": "x"}')
    assert abs(p["SUPPORT"] - 0.7) < 1e-12 and q == "x"
    p, _ = parse_judge("B", '{"verdict": "UNRESOLVED", "confidence": 0.6}')
    assert abs(p["NEI"] - 0.6) < 1e-12 and abs(p["SUPPORT"] - 0.2) < 1e-12
    p, _ = parse_judge("C", '{"p_true": 0.5, "p_false": 0.5}')
    assert p["NEI"] == 0.0
    p, _ = parse_judge("A", "I cannot answer")
    assert p.get("_unparsed") == 1.0 and abs(sum(v for k, v in p.items() if k != "_unparsed") - 1) < 1e-12


def test_cost_matches_price_list_and_cap_is_hard(tmp_path):
    u = {
        "prompt_tokens": 813,
        "completion_tokens": 71,
        "prompt_tokens_details": {"cached_tokens": 128},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    assert abs(usage_cost_usd("grok-4.20-0309-non-reasoning", u) - 10593500 / 1e10) < 1e-12  # = API's cost_in_usd_ticks
    led = SpendLedger(str(tmp_path / "s.jsonl"), cap_usd=0.01)
    led.add("m", 0.006, None)
    led.check()
    led.add("m", 0.006, None)
    with pytest.raises(BudgetExceeded):
        led.check()
    assert SpendLedger(str(tmp_path / "s.jsonl"), cap_usd=0.01).total == pytest.approx(0.012)  # replayed


def test_scifact_label_rule(tmp_path):
    rows = [
        {"id": 1, "claim": "a", "evidence": {}, "cited_doc_ids": [5]},
        {"id": 2, "claim": "b", "evidence": {"5": [{"sentences": [0], "label": "SUPPORT"}]}, "cited_doc_ids": [5]},
        {
            "id": 3,
            "claim": "c",
            "evidence": {"5": [{"sentences": [1, 2], "label": "CONTRADICT"}]},
            "cited_doc_ids": [5],
        },
    ]
    (tmp_path / "claims_dev.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    cs = load_claims(str(tmp_path), "dev")
    assert [c.label3 for c in cs] == ["NEI", "SUPPORT", "CONTRADICT"] and [c.supported for c in cs] == [
        False,
        True,
        False,
    ]
    assert cs[2].rationale == (("5", 1), ("5", 2))
    with pytest.raises(ValueError):
        load_claims(str(tmp_path), "test")


def test_exact_mcnemar_and_platt():
    from evaluate import mcnemar, platt, platt_fit

    assert mcnemar([True] * 6 + [False] * 4, [False] * 6 + [False] * 4) == (6, 0, pytest.approx(2 / 64))
    ab = platt_fit([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.6, 0.4], [1, 1, 1, 0, 0, 0, 0, 1])
    assert platt(ab, 0.9) > 0.5 > platt(ab, 0.1)


def test_quote_gate_is_byte_exact():
    """The judge-quote half of the exact-copy gate (phase3/evaluate.py:quote_verbatim): paraphrase, a one-byte edit,
    a stitched quote across two documents' sentences that are not in the corpus set, or a short quote all close it."""
    from evaluate import quote_verbatim
    from factcheck.scifact import Doc, sha

    d = Doc("1", "Aspirin and the heart", ("Aspirin irreversibly inhibits COX-1 in platelets.", "Dosing was daily."))
    hashes = {sha(s) for s in d.sentences} | {sha(d.title)}
    assert quote_verbatim("Aspirin irreversibly inhibits COX-1 in platelets.", [d], hashes)
    assert quote_verbatim("irreversibly inhibits COX-1 in platelets", [d], hashes)  # substring of one sentence
    assert not quote_verbatim("Aspirin irreversibly blocks COX-1 in platelets.", [d], hashes)  # paraphrase
    assert not quote_verbatim("Aspirin irreversibly inhibits COX-1 in platelets!", [d], hashes)  # one byte
    assert not quote_verbatim("COX-1", [d], hashes)  # too short
    assert not quote_verbatim("Aspirin irreversibly inhibits COX-1 in platelets.", [d], set())  # sentence not in corpus


def test_verifier_prompts_parse_quote_and_probabilities():
    p, q = parse_judge("V", '{"quote": "Aspirin inhibits COX-1.", "support": 0.9, "contradict": 0.05, "nei": 0.05}')
    assert q == "Aspirin inhibits COX-1." and abs(p["SUPPORT"] - 0.9) < 1e-9
    p, q = parse_judge(
        "W", 'Step 2: support: partial\n{"quote": "x y z", "support": 0.1, "contradict": 0.0, "nei": 0.9}'
    )
    assert q == "x y z" and max(p, key=p.get) == "NEI"
