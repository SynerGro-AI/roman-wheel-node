"""What fits on this box for Phase 3? Prints RAM, disk, CPU features, runtimes, and candidate model sizes.

python phase3/check_box.py [--online]      # --online queries the Hugging Face API for file sizes (no downloads)
"""

import importlib.util
import json
import os
import shutil
import sys
import urllib.request

CANDIDATES = [
    "cross-encoder/nli-deberta-v3-xsmall",
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/nli-MiniLM2-L6-H768",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
]


def main():
    mem = {
        line.split(":")[0]: int(line.split()[1])
        for line in open("/proc/meminfo", encoding="utf-8")
        if line.split(":")[0] in ("MemTotal", "MemAvailable")
    }
    du = shutil.disk_usage(os.getcwd())
    flags = set(next(line for line in open("/proc/cpuinfo", encoding="utf-8") if line.startswith("flags")).split())
    print(f"RAM total {mem['MemTotal'] / 1e6:.1f} GB, available {mem['MemAvailable'] / 1e6:.1f} GB")
    print(
        f"disk free {du.free / 1e9:.0f} GB;  CPUs {os.cpu_count()};  "
        f"avx2={'avx2' in flags} avx512f={'avx512f' in flags} avx512_vnni={'avx512_vnni' in flags};  GPU: none expected"
    )
    print(
        "python packages:",
        {m: bool(importlib.util.find_spec(m)) for m in ("onnxruntime", "numpy", "tokenizers", "transformers", "torch")},
    )
    print("XAI_API_KEY set:", bool(os.environ.get("XAI_API_KEY")))
    if "--online" in sys.argv:
        for m in CANDIDATES:
            d = json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{m}?blobs=true", timeout=20))
            sizes = {s["rfilename"]: s.get("size") for s in d.get("siblings", [])}
            pick = {
                k: round(v / 1e6)
                for k, v in sizes.items()
                if v
                and k
                in (
                    "model.safetensors",
                    "onnx/model.onnx",
                    "onnx/model_qint8_avx512_vnni.onnx",
                    "onnx/model_quantized.onnx",
                )
            }
            print(f"  {m}: {pick} MB")


if __name__ == "__main__":
    main()
