"""Maintainer-only: check that the committed Phase 5 split manifest reproduces from the internal COVID-Fact cache.

    python phase5/check_manifest_from_data.py [--items phase3/cache/phase4/items.json]

COVID-Fact has no stated licence, so its text is never committed, published or fetched by any script in this repo,
and this check is not part of the pytest suite (it cannot run on a clean checkout). Exit 0 = reproduces and no claim
text or raw group key leaks into the manifest; 1 = mismatch or leak; 2 = cache missing.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phase5"))
import pairs_common as C  # noqa: E402

ITEMS = os.path.join(ROOT, "phase3", "cache", "phase4", "items.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=ITEMS)
    a = ap.parse_args()
    if not os.path.isfile(a.items):
        print(f"missing {a.items}: the internal COVID-Fact cache (licence unclear, never distributed)", file=sys.stderr)
        return 2
    items = [i for i in json.loads(Path(a.items).read_text(encoding="utf-8")) if i["ds"] == "covidfact"]
    problems = []
    if C.manifest_doc(items, C.build_split(items))["sha256"] != C.load_manifest()["sha256"]:
        problems.append("manifest sha256 does not reproduce")
    blob = Path(C.MANIFEST).read_text(encoding="utf-8")
    if any(it["claim"] in blob for it in items if len(it["claim"]) > 15):
        problems.append("claim text found in the manifest")
    if any(C.group_of(it["evidence"]) in blob for it in items):
        problems.append("raw group key found in the manifest")
    for p in problems:
        print(p, file=sys.stderr)
    print("manifest reproduces from data" if not problems else "FAILED")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
