"""1.2.1 B2/B3: node errors are never silent and never opaque.

B2: when errors.jsonl cannot be written, the error is reported on stderr instead of being dropped.
B3: a model that cannot be loaded is named (or torch is said to be missing), not reported as 'error:KeyError'.
No test doubles: real ShadowRunner, real LocalSources, real filesystem conditions, real parse errors."""

import json
import os

from roman_wheel_node.shadow import ShadowRunner
from roman_wheel_node.sources import LocalSources


def test_unwritable_error_log_is_reported_on_stderr(tmp_path, capsys):
    r = ShadowRunner(str(tmp_path / "shadow"), sources=LocalSources(), trusted_units=["x."])
    os.mkdir(tmp_path / "shadow" / "errors.jsonl")  # a directory where the log file should be, so appending to it fails
    capsys.readouterr()
    out = r.check_output("{not json", json.loads)  # a real extractor raising a real JSONDecodeError
    assert [v.reason for v in out] == ["error:extract"] and r.errors == 1
    err = capsys.readouterr().err
    assert "could not write" in err and "errors.jsonl" in err
    assert "node error in extract" in err and "JSONDecodeError" in err


def _have(mod):
    import importlib.util

    return importlib.util.find_spec(mod) is not None


def test_unloadable_nli2_model_is_named_not_keyerror(tmp_path, capsys):
    missing = tmp_path / "empty-model-dir"  # an existing local directory with no model in it: no network lookup
    missing.mkdir()
    missing = str(missing)
    capsys.readouterr()
    r = ShadowRunner(str(tmp_path / "shadow"), nli2=missing, trusted_units=["IL-6 rises."])
    err = capsys.readouterr().err
    srcs = r.sources
    assert "nli2" not in srcs.available
    if _have("torch") and _have("transformers"):
        assert f"could not load model {missing!r}" in srcs.errors["nli2"]
    else:
        assert "torch/transformers not installed" in srcs.errors["nli2"]
        assert "roman-wheel-node[local-nli2]" in srcs.errors["nli2"]
    # said once at start-up, on stderr, with the cause ...
    assert "needs source(s) ['nli2']" in err and srcs.errors["nli2"][:60] in err
    # ... and every verdict names the missing source instead of 'error:KeyError'
    v = r.observe("IL-6 rises.", ["IL-6 rises."])
    assert v.outcome == "NOT_SURE" and v.reason == "error:missing_source:nli2"


def test_unloadable_onnx_nli_model_is_named(tmp_path):
    missing = tmp_path / "empty-onnx-dir"
    missing.mkdir()
    missing = str(missing)
    s = LocalSources(nli_dir=missing)
    assert "nli" not in s.available
    if _have("onnxruntime") and _have("tokenizers") and _have("numpy"):
        assert f"could not load the ONNX NLI model from {missing!r}" in s.errors["nli"]
    else:
        assert "onnxruntime/tokenizers/numpy not installed" in s.errors["nli"]
        assert "roman-wheel-node[local-nli]" in s.errors["nli"]


def test_profile_source_never_configured_is_reported(tmp_path, capsys):
    capsys.readouterr()
    r = ShadowRunner(str(tmp_path / "shadow"), sources=LocalSources(), trusted_units=["x."])
    err = capsys.readouterr().err
    assert "needs source(s) ['nli2']" in err and "nli2: not configured (pass nli2=...)" in err
    assert r.observe("x.", ["x."]).reason == "error:missing_source:nli2"


def test_malformed_score_is_labelled_malformed_not_missing(tmp_path):
    """R4: a source that answered with an unusable score is 'malformed_score', not 'missing_source'."""
    cases = [
        ({"overlap": {"S": 0.9}}, "error:missing_source:nli2"),
        ({"overlap": {"S": 0.9}, "nli2": None}, "error:missing_source:nli2"),
        ({"overlap": {"S": 0.9}, "nli2": {"C": 0.1}}, "error:malformed_score:nli2"),
        ({"overlap": {"S": 0.9}, "nli2": {"S": "x"}}, "error:malformed_score:nli2"),
        ({"overlap": {"S": 0.9}, "nli2": {"S": float("nan")}}, "error:malformed_score:nli2"),
        ({"overlap": {"S": 1.5}, "nli2": {"S": 0.9}}, "error:malformed_score:overlap"),
        ({"overlap": [0.9], "nli2": "0.9"}, "error:malformed_score:nli2,overlap"),
    ]
    for n, (raw, why) in enumerate(cases):
        r = ShadowRunner(str(tmp_path / f"s{n}"), sources=lambda c, s, raw=raw: raw, trusted_units=["x."])
        v = r.observe("x.", ["x."])
        assert (v.outcome, v.reason) == ("NOT_SURE", why), raw
