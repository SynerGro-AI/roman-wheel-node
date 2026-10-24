"""SciFact is licensed for non-commercial use: committed Phase 3 outputs carry SciFact IDs only, never dataset text."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R3 = os.path.join(ROOT, "results", "phase3")
TEXT_KEYS = {"claim", "title", "quote", "evidence", "abstract", "sentence", "text"}


def test_label_review_is_ids_only():
    lr = json.load(open(os.path.join(R3, "label_review.json")))
    for split, xs in lr.items():
        for x in xs:
            assert not TEXT_KEYS & set(x), (split, x["key"])
            assert x["key"] == f"{split}:{x['claim_id']}"
            assert all(isinstance(d, str) and isinstance(i, int) for d, i in x["rationale"] + x["quote_sentences"])


def test_scifact_report_candidates_are_ids_only():
    sf = json.load(open(os.path.join(R3, "scifact.json")))
    for C in sf["conditions"].values():
        for x in C.get("label_noise_candidates", []):
            assert not TEXT_KEYS & set(x), x["key"]
            assert isinstance(x["claim_id"], int)


def test_label_review_md_renders_ids_only():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "phase3"))
    from label_review import render_md
    lr = json.load(open(os.path.join(R3, "label_review.json")))
    md = render_md(lr)
    assert md == open(os.path.join(ROOT, "phase3", "label_review.md")).read()
    assert '"' not in md   # no quoted dataset sentences
