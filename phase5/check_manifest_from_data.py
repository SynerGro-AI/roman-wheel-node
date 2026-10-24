"""Maintainer-only: check that the committed Phase 5 split manifest reproduces from the internal COVID-Fact cache.

    python phase5/check_manifest_from_data.py [--items phase3/cache/phase4/items.json]

COVID-Fact has no stated licence, so its text is never committed, published or fetched by any script in this repo,
and the real check is not part of the pytest suite (it cannot run on a clean checkout; tests/test_repo_manifest_check.py
only checks the exit codes on a missing, broken or invented cache). Exit codes:
  0 = reproduces and no claim text or raw group key leaks into the manifest; prints "manifest reproduces from data"
  1 = mismatch or leak; prints each problem on stderr, then "FAILED"
  2 = cache missing, unreadable or malformed (not a list of item objects with the keys the split reads)
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
    try:
        data = json.loads(Path(a.items).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as e:  # ValueError covers bad UTF-8 and bad JSON
        print(f"cannot read {a.items}: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        return 2
    keys = ("claim", "label3", "evidence", "units")  # what build_split / manifest_doc read from a COVID-Fact item
    if not isinstance(data, list) or not all(isinstance(i, dict) and "ds" in i for i in data):
        print(f"{a.items}: expected a JSON list of item objects, each with a 'ds' key", file=sys.stderr)
        return 2
    items = [i for i in data if i["ds"] == "covidfact"]
    if not all(k in i for i in items for k in keys):
        print(f"{a.items}: every covidfact item needs the keys {', '.join(keys)}", file=sys.stderr)
        return 2
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
