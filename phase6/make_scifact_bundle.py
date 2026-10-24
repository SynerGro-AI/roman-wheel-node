"""Maintainer-only: write phase6/data/scifact_step2_features.json from the local research caches.

    python phase6/make_scifact_bundle.py --cache /path/to/phase3/cache --scifact-tarball data.tar.gz --write-pins

Reads phase3/cache/step2/{items,local}.json and phase3/cache/local_features.json (gold doc ids), keeps the SciFact
rows only (HealthVer, Climate-FEVER and COVID-Fact never enter the bundle), and stores no text: id, split, gold doc ids
and the five rounded scores. It then rebuilds items/local with scripts/fetch_test_data.py from the pinned SciFact
tarball and checks that every rebuilt claim, label, evidence list and score equals the research cache exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "phase6"))
import fetch_test_data as FT  # noqa: E402
import refit_policy_43 as RF  # noqa: E402

KEEP = (*RF.FIT_DS, RF.TEST_DS)
FEATURE_KEYS = ("nli_entail", "nli_contra", "nli2_entail", "nli2_contra", "overlap")


def _load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "phase3", "cache"))
    ap.add_argument("--scifact-tarball", help="local SciFact data.tar.gz (else downloaded)")
    ap.add_argument("--out", default=FT.FEATURES)
    ap.add_argument("--write-pins", action="store_true", help="update the sha256 pins in scripts/fetch_test_data.py")
    a = ap.parse_args()
    items = _load(os.path.join(a.cache, "step2", "items.json"))
    local = _load(os.path.join(a.cache, "step2", "local.json"))
    gold = _load(os.path.join(a.cache, "local_features.json"))
    rows = []
    for it in items:
        if it["ds"] not in KEEP:
            continue
        assert it["id"].startswith("sf:"), it["id"]
        f = local[it["id"]]
        rows.append(
            {
                "id": it["id"],
                "ds": it["ds"],
                "docs": [str(d) for d in gold[it["id"][3:]]["gold"]["docs"]],
                "f": {k: f[k] for k in FEATURE_KEYS},
            }
        )
    doc = {
        "format": "rwn-scifact-step2-features/1",
        "source": {"url": FT.SCIFACT_URL, "sha256": FT.SCIFACT_SHA256, "bytes": FT.SCIFACT_SIZE},
        "licence": "Derived from SciFact (claims/annotations CC BY 4.0; abstracts ODC-By 1.0). See DATA_LICENSES.md.",
        "scores": "nli = cross-encoder/nli-deberta-v3-small, nli2 = MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, "
        "overlap = lexical; rounded to 6 decimals",
        "rows": rows,
    }
    body = json.dumps({k: v for k, v in doc.items() if k != "rows"}, sort_keys=True)[:-1]
    text = body + ', "rows": [\n' + ",\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n]}\n"
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    data = text.encode("utf-8")
    print(f"{a.out}: {len(rows)} rows, {len(data)} bytes, sha256 {FT.sha256(data)}")

    out = FT.rebuild(FT.read_scifact(a.scifact_tarball), data)
    built = json.loads(out["items.json"])
    built_local = json.loads(out["local.json"])
    want = [{k: it[k] for k in ("id", "ds", "claim", "label3", "sents")} for it in items if it["ds"] in KEEP]
    if built != want or built_local != {r["id"]: r["f"] for r in rows}:
        print("rebuild does NOT equal the research cache", file=sys.stderr)
        return 1
    print(f"rebuild equals the research cache for all {len(want)} SciFact rows")
    pins = {"FEATURES_SHA256": FT.sha256(data), "ITEMS_SHA256": FT.sha256(out["items.json"])}
    pins["LOCAL_SHA256"] = FT.sha256(out["local.json"])
    for k, v in pins.items():
        print(f"{k} = {v}")
    if a.write_pins:
        path = os.path.join(ROOT, "scripts", "fetch_test_data.py")
        with open(path, encoding="utf-8") as f:
            src = f.read()
        src = re.sub(r'FEATURES_SHA256 = "[^"]*"', f'FEATURES_SHA256 = "{pins["FEATURES_SHA256"]}"', src)
        src = re.sub(r'"items.json": "[^"]*"', f'"items.json": "{pins["ITEMS_SHA256"]}"', src)
        src = re.sub(r'"local.json": "[^"]*"', f'"local.json": "{pins["LOCAL_SHA256"]}"', src)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(src)
        print(f"updated pins in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
