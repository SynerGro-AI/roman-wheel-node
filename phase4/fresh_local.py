"""Phase 4: free local sources for every fresh item (same models and features as Phase 3d s2_features.py:
nli = deberta-v3-small int8 ONNX, nli2 = DeBERTa-v3-base mnli-fever-anli (torch, CPU), overlap).
Output: phase3/cache/phase4/local.json (gitignored). Zero API calls."""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase3"))
import s2_features as F  # noqa: E402
from features import CACHE  # noqa: E402

if __name__ == "__main__":
    if os.environ.get("XAI_API_KEY"):
        sys.exit("refusing to run with XAI_API_KEY set (local features need no key): use env -u XAI_API_KEY")
    F.main(os.path.join(CACHE, "phase4", "items.json"), os.path.join(CACHE, "phase4", "local.json"))
