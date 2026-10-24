"""Phase 5 / Task B: derive rwn-policy-4.1.0 from the frozen 4.0.0 file (no refit, no data, no API call).

4.1.0 = every 4.0.0 profile unchanged (same calibrators, reliabilities, cut-offs, margins) + the fp2 exact-copy
fingerprint (``roman_wheel_node.fingerprint``: Unicode NFC, invisible format chars dropped, quotes/dashes unified,
whitespace collapsed, version-tagged). 4.0.0 is not touched and stays the default. The script refuses to write if the
4.0.0 body no longer matches its pinned hash, and it pins the new hash in policy.py.

    python3 phase5/export_policy_41.py
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from roman_wheel_node import policy as P                         # noqa: E402
from roman_wheel_node.fingerprint import NORMALIZE_V2_TABLE_SHA256  # noqa: E402


def main() -> None:
    base = P.load_policy()                                       # verifies the 4.0.0 pin
    assert base.sha256 == P.DEFAULT_POLICY_SHA256 and base.version == "4.0.0"
    doc = {k: v for k, v in json.load(open(P.DEFAULT_POLICY_PATH, encoding="utf-8")).items() if k != "sha256"}
    doc["version"] = "4.1.0"
    doc["parent_policy"] = {"version": "4.0.0", "sha256": base.sha256}
    doc["gate"] = {"fingerprint": "fp2", "normalize": "NFC; drop U+200B-200D,2060,FEFF,00AD; curly/typographic quotes "
                   "-> ' and \"; hyphen/dash/minus variants -> -; whitespace runs -> one space; strip. Not NFKC.",
                   "normalize_table_sha256": NORMALIZE_V2_TABLE_SHA256,
                   "superset_of": "fp1 (a unit passes if its raw sha256 or its fp2 fingerprint is trusted)"}
    doc["honesty"] = doc["honesty"] + " 4.1.0 changes only the exact-copy gate fingerprint (fp2); the certificate is " \
        "unchanged and was earned under the raw-byte gate."
    doc["sha256"] = P.policy_sha256(doc)
    with open(P.POLICY_4_1_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    src = os.path.join(ROOT, "src", "roman_wheel_node", "policy.py")
    s = open(src, encoding="utf-8").read()
    s = re.sub(r'POLICY_4_1_SHA256 = "[^"]*"', f'POLICY_4_1_SHA256 = "{doc["sha256"]}"', s)
    open(src, "w", encoding="utf-8").write(s)
    print("wrote", P.POLICY_4_1_PATH, "sha256", doc["sha256"])


if __name__ == "__main__":
    main()
