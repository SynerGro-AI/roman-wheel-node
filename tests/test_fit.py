"""The public fit API (roman_wheel_node.fit): labelled JSONL in, a loadable, deterministic policy file out.

The examples here are SYNTHETIC, generated deterministically below (fixed strides through a word list and a Weyl
sequence; no random generator, no dataset text). They test the mechanics only and are not an evaluation of anything.
No mocks: the real local ``overlap`` scorer (:class:`~roman_wheel_node.sources.LocalSources` with no models)
computes the features, and the fitted policy is loaded with the real loader and run through the real FrozenNode and
ShadowRunner.
"""

import json
import os
import sys

import pytest

from roman_wheel_node import fit as FIT
from roman_wheel_node.policy import ACCEPT, BLOCK, NOT_SURE, PolicyError, load_policy, policy_sha256
from roman_wheel_node.shadow import FrozenNode, ShadowRunner
from roman_wheel_node.sources import LocalSources

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORDS = [f"{a}{b}" for a in ("lum", "ver", "tas", "kor", "pex", "dal", "mir", "sof") for b in ("an", "o", "ix", "et")]


def _frac(i, k):
    """Deterministic low-discrepancy value in [0, 1): the fractional part of i times an irrational-ish constant (Weyl
    sequence). Fixed arithmetic, identical on every platform; no random number generator."""
    return (i * (0.6180339887498949, 0.4142135623730951, 0.7320508075688772)[k]) % 1.0


def synthetic(n=160):
    """Deterministic synthetic (claim, evidence, label) rows: SUPPORT evidence repeats most claim words, NEI evidence
    shares few. Some rows are deliberately noisy so the calibrators have something to fit. Words are chosen by fixed
    strides through WORDS (an odd stride over 32 words gives 6 distinct words), not by a random generator."""
    out = []
    for i in range(n):
        start, step = (i * 7) % len(WORDS), 2 * (i % 8) + 1
        claim_w = [WORDS[(start + k * step) % len(WORDS)] for k in range(6)]
        sup = i % 2 == 0
        keep = 4 + (i // 2) % 3 if sup else (i // 2) % 4
        if i % 11 == 0:  # noise: label disagrees with the overlap
            keep = 6 - keep
        other = [w for w in WORDS if w not in claim_w]
        off = (i * 5) % len(other)
        ev = claim_w[:keep] + [other[(off + k) % len(other)] for k in range(6 - keep)]
        r = i % 6
        ev = ev[r:] + ev[:r]
        if i % 4 == 1:
            ev.reverse()
        out.append({"id": f"syn{i}", "claim": " ".join(claim_w) + ".", "evidence": [" ".join(ev) + "."], "label": sup})
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


def test_fit_is_deterministic_loads_and_runs(tmp_path):
    src = write_jsonl(tmp_path / "ex.jsonl", synthetic())
    kw = dict(profiles={"lex": ("overlap",)}, sources=LocalSources(), version="test-1", data_licences="synthetic")
    a = FIT.fit_policy(src, **kw)
    b = FIT.fit_policy(FIT.load_examples(src), **kw)
    ha = FIT.write_policy(a, str(tmp_path / "a.json"))
    hb = FIT.write_policy(b, str(tmp_path / "b.json"))
    assert ha == hb and (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    pol = load_policy(str(tmp_path / "a.json"), expected_sha256=ha)
    prof = pol.profile()
    assert pol.default_profile == "lex" and prof.sources == ("overlap",) and pol.fingerprint == "fp2"
    assert prof.raw["calibration"]["n"] == 160 and prof.raw["certified"]["alpha"] is None
    assert prof.accept_margin > 0 and 0 < prof.cutoff < 1
    node = FrozenNode(str(tmp_path / "audit.jsonl"), policy=pol)
    hi = node.decide("c", {"overlap": {"S": 0.95}}, 1).outcome
    lo = node.decide("c", {"overlap": {"S": 0.05}}, 1).outcome
    assert (hi, lo) == (ACCEPT, NOT_SURE)
    assert node.decide("c", {"overlap": {"S": 0.95}}, 0).outcome == BLOCK
    assert node.decide("c", {}, 1).outcome == NOT_SURE  # missing source: fail closed
    node.trace.verify()
    ev = ["luman vero tasix koret pexan dalo."]
    r = ShadowRunner(str(tmp_path / "shadow"), policy=pol, sources=LocalSources(), trusted_units=ev)
    assert r.observe("luman vero tasix koret pexan dalo.", ev).outcome == ACCEPT
    assert r.observe("mirix sofet miro sofan lumet verix.", ev).outcome == NOT_SURE
    assert r.observe("luman vero tasix koret pexan dalo.", ["not trusted"]).outcome == BLOCK
    r.verify()


def test_fit_with_given_scores_nli2_contradiction_and_certificate(tmp_path):
    rows = []
    for i in range(400):
        y = ["SUPPORT", "NEI", "CONTRADICT"][i % 3]
        u, v, w = _frac(i, 0), _frac(i, 1), _frac(i, 2)
        s = 0.35 + 0.65 * u if y == "SUPPORT" else 0.6 * u  # overlapping class ranges, deterministic
        c = 0.4 + 0.6 * v if y == "CONTRADICT" else 0.35 * v
        o = min(1.0, max(0.0, s + 0.4 * (w - 0.5)))
        rows.append(
            {
                "id": f"g{i}",
                "claim": f"claim {i}",
                "evidence": "Synthetic.",
                "label": y,
                "scores": {"nli2": {"S": s, "C": c}, "overlap": {"S": o}},
            }
        )
    doc = FIT.fit_policy(
        rows,
        profiles={"local": ("nli2", "overlap")},
        alphas=(0.10, 0.01),
        disagreement={"local": {"nli2_contradiction": True}},
    )
    again = FIT.fit_policy(
        rows,
        profiles={"local": ("nli2", "overlap")},
        alphas=(0.10, 0.01),
        disagreement={"local": {"nli2_contradiction": True}},
    )
    assert doc == again and doc["sha256"] == policy_sha256(doc)
    p = doc["profiles"]["local"]
    assert set(p["calibrators"]["nli2"]) == {"S", "C"} and set(p["calibrators"]["overlap"]) == {"S"}
    assert p["disagreement"] == {"nli2_contradiction": True}
    assert set(p["ltt"]["alphas"]) == {"0.1", "0.01"} and p["ltt"]["n_tests"] == 1
    cert = p["certified"]
    if cert["alpha"] is not None:  # whichever way it falls, the file must say so consistently
        assert p["ltt"]["alphas"][str(cert["alpha"])]["certified"] and cert["upper"] <= cert["alpha"]
        assert doc["honesty"].startswith("Certified")
    else:
        assert cert["note"].startswith("NOT certified") and doc["honesty"] == "No profile is certified at any level."
    path = tmp_path / "p.json"
    pol = load_policy(str(path), FIT.write_policy(doc, str(path)))
    node = FrozenNode(str(tmp_path / "a.jsonl"), policy=pol)
    v = node.decide("c", {"nli2": {"S": 0.99, "C": 0.99}, "overlap": {"S": 0.99}}, 1)
    assert v.outcome == NOT_SURE  # strong contradiction: the disagreement rule holds it back
    assert node.decide("c", {"nli2": {"S": 0.99, "C": 0.01}, "overlap": {"S": 0.99}}, 1).outcome == ACCEPT


def test_cli(tmp_path, capsys):
    src = write_jsonl(tmp_path / "ex.jsonl", synthetic())
    out = tmp_path / "cli.json"
    assert FIT._main([src, "-o", str(out), "--profile", "lex=overlap", "--version-tag", "t-2"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["default_profile"] == "lex" and summary["profiles"]["lex"]["sources"] == ["overlap"]
    pol = load_policy(str(out), expected_sha256=summary["sha256"])
    assert pol.version == "t-2"
    assert FIT._main([src, "-o", str(tmp_path / "x.json"), "--profile", "bad=overlap,J1"]) == 1
    assert "profile sources must be" in capsys.readouterr().err


def test_fit_errors(tmp_path):
    good = synthetic()
    with pytest.raises(FIT.FitError):
        FIT.fit_policy(good[:20], profiles={"lex": ("overlap",)}, sources=LocalSources())  # too few
    with pytest.raises(FIT.FitError):
        FIT.fit_policy(good, profiles={"x": ("overlap", "J4")}, sources=LocalSources())  # not a local source
    with pytest.raises(FIT.FitError):
        FIT.fit_policy(good, profiles={"x": ("nli2",)}, sources=LocalSources())  # no nli2 model, no scores
    with pytest.raises(FIT.FitError):
        FIT.fit_policy(good, sources=LocalSources())  # defaults need nli/nli2 scores
    with pytest.raises(FIT.FitError):
        FIT.fit_policy([dict(good[0], label="MAYBE")] + good, profiles={"lex": ("overlap",)}, sources=LocalSources())
    with pytest.raises(FIT.FitError):
        FIT.fit_policy([dict(good[0], scores={"overlap": {"S": 1.5}})] + good, profiles={"lex": ("overlap",)})
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"claim": "x", "label": true}\nnot json\n', encoding="utf-8")
    with pytest.raises(FIT.FitError):
        FIT.load_examples(str(bad))
    doc = FIT.fit_policy(good, profiles={"lex": ("overlap",)}, sources=LocalSources())
    doc["profiles"]["lex"]["cutoff"] = 0.4
    with pytest.raises(FIT.FitError):
        FIT.write_policy(doc, str(tmp_path / "t.json"))  # body no longer matches its hash
    doc["profiles"]["lex"]["sources"] = ["overlap", "J1"]
    doc["sha256"] = policy_sha256(doc)
    (tmp_path / "t.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(PolicyError):
        load_policy(str(tmp_path / "t.json"), doc["sha256"])


def test_shipped_policy_is_reproduced_by_the_fit_api():
    """Repo-only: refit 4.3.0 from the local SciFact feature cache with the public API; must match the shipped file."""
    cache = os.path.join(ROOT, "phase3", "cache", "step2", "local.json")
    if not os.path.isfile(cache):
        pytest.skip("phase3/cache/step2 (SciFact local features) is a local research cache, not in git or the sdist")
    sys.path.insert(0, os.path.join(ROOT, "phase6"))
    import refit_policy_43 as RF

    pool, _ = RF.load_rows()
    doc = RF.build_doc(pool)
    shipped = load_policy()
    strip = ("fitted_with", "sha256")
    assert {k: v for k, v in doc.items() if k not in strip} == {k: v for k, v in shipped.doc.items() if k not in strip}
