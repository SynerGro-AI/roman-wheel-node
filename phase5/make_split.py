"""Phase 5: write the committed split manifest (hashed group IDs + roles only, no text). Box or desktop.

    python3 phase5/make_split.py [--items phase3/cache/phase4/items.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairs_common as C  # noqa: E402


def covidfact_items(path: str):
    return [i for i in json.load(open(path, encoding="utf-8")) if i["ds"] == "covidfact"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=os.path.join(C.ROOT, "phase3", "cache", "phase4", "items.json"))
    ap.add_argument("--out", default=C.MANIFEST)
    a = ap.parse_args()
    items = covidfact_items(a.items)
    roles = C.build_split(items)
    doc = C.manifest_doc(items, roles)
    if os.path.exists(a.out):
        old = C.load_manifest(a.out)
        if old["sha256"] != doc["sha256"]:
            sys.exit(f"{a.out} exists with a different split ({old['sha256'][:12]} vs {doc['sha256'][:12]}): refusing")
        print("manifest unchanged", doc["sha256"][:12]); return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(doc, open(a.out, "w", encoding="utf-8"), indent=1, sort_keys=True)
    print(json.dumps(doc["counts"], indent=1)); print("sha256", doc["sha256"])


if __name__ == "__main__":
    main()
