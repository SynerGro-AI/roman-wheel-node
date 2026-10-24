"""Build the SciFact test data that the repo-only refit test needs, from pinned public sources.

    python scripts/fetch_test_data.py                   # writes ~/.cache/roman_wheel_node/test-data/scifact-step2
    python scripts/fetch_test_data.py --dest DIR        # or set RWN_TEST_DATA=DIR (the tests read the same variable)
    python scripts/fetch_test_data.py --scifact-tarball data.tar.gz   # use a local copy instead of downloading

Inputs, both pinned by sha256:
  * SciFact ``data.tar.gz`` from the Allen Institute's S3 release (claims CC BY 4.0, abstracts ODC-By 1.0);
  * ``phase6/data/scifact_step2_features.json`` in this repository: for each SciFact claim used by the 4.3.0 fit, its
    id, split, gold doc ids and the five local feature scores (no text).
Output: ``items.json`` (claim, label, gold evidence sentences) and ``local.json`` (scores), byte-identical on every
run; their sha256 values are pinned below and checked before anything is written. HealthVer, Climate-FEVER and
COVID-Fact are not part of this data.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SCIFACT_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
SCIFACT_SIZE = 3115079
FEATURES = os.path.join(ROOT, "phase6", "data", "scifact_step2_features.json")
FEATURES_SHA256 = "99c3ee8395f1a5b2789863c581aa4159979ca6eba9ef4013b07186e444544cf9"
OUT_SHA256 = {
    "items.json": "9b62c7ec235fb07afe043dec28dc4f3c0d5b88800bd67d9e0d0b6928332bec53",
    "local.json": "bb7f2af49b7c80bad5dcbb27da2d5c0b5dc0afb9febc861fbbc61bfe99a1bcbb",
}
SUBDIR = "scifact-step2"
COMMAND = "python scripts/fetch_test_data.py"


class FetchError(Exception):
    pass


def default_dest() -> str:
    return os.environ.get("RWN_TEST_DATA") or os.path.join(
        os.path.expanduser("~"), ".cache", "roman_wheel_node", "test-data"
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check(name: str, data: bytes, want: str) -> None:
    got = sha256(data)
    if got != want:
        raise FetchError(f"{name}: sha256 {got}, expected {want}")


def read_scifact(tarball: str | None) -> bytes:
    if tarball:
        with open(tarball, "rb") as f:
            data = f.read()
    else:
        with urllib.request.urlopen(SCIFACT_URL, timeout=120) as r:
            data = r.read()
    if len(data) != SCIFACT_SIZE:
        raise FetchError(f"SciFact data.tar.gz: {len(data)} bytes, expected {SCIFACT_SIZE}")
    _check("SciFact data.tar.gz", data, SCIFACT_SHA256)
    return data


def _jsonl(raw: bytes) -> list:
    return [json.loads(line) for line in raw.decode("utf-8").split("\n") if line.strip()]


def rebuild(tar_bytes: bytes, features_bytes: bytes) -> dict[str, bytes]:
    """Return {"items.json": bytes, "local.json": bytes}; a pure function of the two inputs (no pin checks)."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        files = {
            n: tf.extractfile(f"data/{n}").read() for n in ("claims_train.jsonl", "claims_dev.jsonl", "corpus.jsonl")
        }
    corpus = {str(d["doc_id"]): d["abstract"] for d in _jsonl(files["corpus.jsonl"])}
    claims = {}
    for split in ("train", "dev"):
        for c in _jsonl(files[f"claims_{split}.jsonl"]):
            labels = {e["label"] for evs in c["evidence"].values() for e in evs}
            if len(labels) > 1:
                raise FetchError(f"SciFact {split}:{c['id']}: mixed evidence labels {sorted(labels)}")
            label3 = labels.pop() if labels else "NEI"
            claims[f"sf:{split}:{c['id']}"] = (c["claim"], label3)
    feats = json.loads(features_bytes.decode("utf-8"))
    items, local = [], {}
    for row in feats["rows"]:
        claim, label3 = claims[row["id"]]
        items.append(
            {
                "id": row["id"],
                "ds": row["ds"],
                "claim": claim,
                "label3": label3,
                "sents": [s for d in row["docs"] for s in corpus[d]],
            }
        )
        local[row["id"]] = row["f"]
    return {
        "items.json": (json.dumps(items, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"),
        "local.json": (json.dumps(local, sort_keys=True) + "\n").encode("utf-8"),
    }


def build(tar_bytes: bytes, features_bytes: bytes) -> dict[str, bytes]:
    """rebuild() with every input and output checked against the pinned sha256 values."""
    _check("phase6/data/scifact_step2_features.json", features_bytes, FEATURES_SHA256)
    out = rebuild(tar_bytes, features_bytes)
    for name, data in out.items():
        _check(f"built {name}", data, OUT_SHA256[name])
    return out


def install(dest: str, out: dict[str, bytes]) -> str:
    target = os.path.join(dest, SUBDIR)
    os.makedirs(target, exist_ok=True)
    for name, data in out.items():
        fd, tmp = tempfile.mkstemp(dir=target, prefix=f".{name}.")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, os.path.join(target, name))
    return target


def verify_installed(dest: str) -> list[str]:
    """Problems with an installed copy ([] if it matches the pins)."""
    problems = []
    for name, want in OUT_SHA256.items():
        p = os.path.join(dest, SUBDIR, name)
        if not os.path.isfile(p):
            problems.append(f"missing {p}")
            continue
        with open(p, "rb") as f:
            got = sha256(f.read())
        if got != want:
            problems.append(f"{p}: sha256 {got}, expected {want}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--dest", default=default_dest(), help="default: $RWN_TEST_DATA or ~/.cache/roman_wheel_node/test-data"
    )
    ap.add_argument("--scifact-tarball", help="local copy of SciFact data.tar.gz (checked against the pinned sha256)")
    a = ap.parse_args(argv)
    try:
        with open(FEATURES, "rb") as f:
            features = f.read()
        target = install(a.dest, build(read_scifact(a.scifact_tarball), features))
    except (FetchError, OSError) as e:
        print(f"fetch_test_data: {e}", file=sys.stderr)
        return 1
    print(f"wrote {target} (items.json {OUT_SHA256['items.json'][:12]}…, local.json {OUT_SHA256['local.json'][:12]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
