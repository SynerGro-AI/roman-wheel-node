"""Committed Phase 3 outputs carry SciFact IDs only, never dataset text (see DATA_LICENSES.md).

`results/phase3/` and `phase3/` are repo-only research files that are not in the sdist or wheel, so this module skips
with that reason when they are absent.
"""

import json
import os
from pathlib import Path

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R3 = os.path.join(ROOT, "results", "phase3")
if not (os.path.isdir(R3) and os.path.isdir(os.path.join(ROOT, "phase3"))):
    pytest.skip("results/phase3/ and phase3/ are repo-only (not in the sdist/wheel)", allow_module_level=True)
TEXT_KEYS = {"claim", "title", "quote", "evidence", "abstract", "sentence", "text"}


def test_label_review_is_ids_only():
    lr = json.loads(Path(os.path.join(R3, "label_review.json")).read_text())
    for split, xs in lr.items():
        for x in xs:
            assert not TEXT_KEYS & set(x), (split, x["key"])
            assert x["key"] == f"{split}:{x['claim_id']}"
            assert all(isinstance(d, str) and isinstance(i, int) for d, i in x["rationale"] + x["quote_sentences"])


def test_scifact_report_candidates_are_ids_only():
    sf = json.loads(Path(os.path.join(R3, "scifact.json")).read_text())
    for C in sf["conditions"].values():
        for x in C.get("label_noise_candidates", []):
            assert not TEXT_KEYS & set(x), x["key"]
            assert isinstance(x["claim_id"], int)


def test_label_review_md_renders_ids_only():
    import sys

    sys.path.insert(0, os.path.join(ROOT, "phase3"))
    from label_review import render_md

    lr = json.loads(Path(os.path.join(R3, "label_review.json")).read_text())
    md = render_md(lr)
    assert md == Path(os.path.join(ROOT, "phase3", "label_review.md")).read_text()
    assert '"' not in md  # no quoted dataset sentences
