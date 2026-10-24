"""1.2.1 B2/B3: node errors are never silent and never opaque.

B2: when errors.jsonl cannot be written, the error is reported on stderr instead of being dropped.
B3: a model that cannot be loaded is named (or torch is said to be missing), not reported as 'error:KeyError'.
No mocks: real ShadowRunner, real LocalSources, real filesystem conditions, real parse errors."""

import json
import os

from roman_wheel_node.shadow import ShadowRunner
from roman_wheel_node.sources import LocalSources


def test_unwritable_error_log_is_reported_on_stderr(tmp_path, capsys):
    r = ShadowRunner(str(tmp_path / "shadow"), sources=LocalSources(), trusted_units=["x."])
    os.mkdir(tmp_path / "shadow" / "errors.jsonl")  # a directory where the log file should be: open(..., "a") fails
    capsys.readouterr()
    out = r.check_output("{not json", json.loads)  # a real extractor raising a real JSONDecodeError
    assert [v.reason for v in out] == ["error:extract"] and r.errors == 1
    err = capsys.readouterr().err
    assert "could not write" in err and "errors.jsonl" in err
    assert "node error in extract" in err and "JSONDecodeError" in err
