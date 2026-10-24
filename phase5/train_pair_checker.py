"""Phase 5 Task A: fine-tune a small NLI cross-encoder as a meaning-change checker on COVID-Fact true/fake pairs.

Input (evidence units joined, claim) -> SUPPORT (entailment) / REFUTE (contradiction), on the 3-way MNLI head, so the
neutral class keeps its meaning. Trains on A_train ONLY. A_dev is scored for information; B_test is never loaded
for training (the loader refuses it). Fixed seeds, deterministic kernels where torch allows, CUDA if available.

    # desktop (CUDA):
    python phase5/train_pair_checker.py --out phase5/models/pairs-v1
    # box smoke (CPU, tiny random model, tiny subset):
    python phase5/train_pair_checker.py --tiny-random --limit 48 --epochs 1 --out phase5/runs/smoke/model

The model directory is gitignored (phase5/models/, phase5/runs/).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pairs_common as C  # noqa: E402

BASE = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def seed_all(seed: int):
    import numpy as np
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def label_ids(config):
    l2i = {k.lower(): v for k, v in config.label2id.items()}
    ent = [v for k, v in l2i.items() if k.startswith("entail")]
    con = [v for k, v in l2i.items() if k.startswith("contra")]
    if len(ent) != 1 or len(con) != 1:
        raise SystemExit(f"cannot find entailment/contradiction in label2id {config.label2id}")
    return ent[0], con[0]


def load_model(a, device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.base, cache_dir=a.hf_cache)
    if a.tiny_random:
        from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

        cfg = DebertaV2Config(
            vocab_size=len(tok),
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=512,
            relative_attention=True,
            position_buckets=64,
            pos_att_type=["p2c", "c2p"],
            num_labels=3,
            id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
            label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
            pad_token_id=tok.pad_token_id,
        )
        torch.manual_seed(a.seed)
        model = DebertaV2ForSequenceClassification(cfg)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(a.base, cache_dir=a.hf_cache)
    return tok, model.to(device)


def encode(tok, rows, max_len):
    return tok(
        [C.premise(r["units"]) for r in rows],
        [r["claim"] for r in rows],
        truncation="only_first",
        max_length=max_len,
        padding=True,
        return_tensors="pt",
    )


def score(model, tok, rows, device, max_len, bs, ent, con):
    """P(entailment) under softmax over all 3 classes, fp32. Returns list aligned with rows."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            enc = {k: v.to(device) for k, v in encode(tok, rows[i : i + bs], max_len).items()}
            p = torch.softmax(model(**enc).logits.float(), dim=-1)
            out += p[:, ent].tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=C.BUNDLE)
    ap.add_argument("--base", default=BASE)
    ap.add_argument(
        "--hf-cache", default=os.environ.get("RWN_HF_CACHE", os.path.join(C.ROOT, "phase3", "models", "hf"))
    )
    ap.add_argument("--out", default=os.path.join(C.HERE, "models", "pairs-v1"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--amp", choices=["auto", "on", "off"], default="auto", help="fp16 autocast (auto = on with CUDA)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="smoke only: use the first N A_train and N A_dev items")
    ap.add_argument("--tiny-random", action="store_true", help="smoke only: tiny randomly initialised DeBERTa-v2")
    a = ap.parse_args()

    import torch

    seed_all(a.seed)
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    amp = (a.amp == "on") or (a.amp == "auto" and dev.type == "cuda")
    if os.path.exists(os.path.join(a.out, "train_meta.json")):
        sys.exit(f"{a.out} already holds a trained model; pick a new --out")

    b = C.load_bundle(a.bundle)
    man = C.load_manifest()
    if b["manifest_sha256"] != man["sha256"]:
        sys.exit("bundle was built from a different split manifest")
    train = [r for r in b["covidfact"] if r["role"] == "A_train"]
    devset = [r for r in b["covidfact"] if r["role"] == "A_dev"]
    assert not any(r["role"] == "B_test" for r in train + devset)
    if a.limit:
        rng = random.Random(a.seed)
        train = rng.sample(train, min(a.limit, len(train)))
        devset = rng.sample(devset, min(a.limit, len(devset)))

    tok, model = load_model(a, dev)
    ent, con = label_ids(model.config)

    def target(r):
        return ent if r["y"] else con

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps = math.ceil(len(train) / (a.batch_size * a.grad_accum)) * a.epochs
    warm = int(a.warmup * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (steps - s) / max(1, steps - warm))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dev.type == "cuda")
    print(
        f"device {dev} amp {amp} train {len(train)} dev {len(devset)} steps {steps} base "
        f"{'tiny-random' if a.tiny_random else a.base}",
        flush=True,
    )

    hist, t0, g = [], time.time(), random.Random(a.seed)
    for ep in range(a.epochs):
        model.train()
        order = list(range(len(train)))
        g.shuffle(order)
        tot = n = 0
        for bi, i in enumerate(range(0, len(order), a.batch_size)):
            rows = [train[j] for j in order[i : i + a.batch_size]]
            enc = {k: v.to(dev) for k, v in encode(tok, rows, a.max_len).items()}
            y = torch.tensor([target(r) for r in rows], device=dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                loss = torch.nn.functional.cross_entropy(model(**enc).logits.float(), y) / a.grad_accum
            scaler.scale(loss).backward()
            if (bi + 1) % a.grad_accum == 0 or i + a.batch_size >= len(order):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            tot += loss.item() * a.grad_accum * len(rows)
            n += len(rows)
        p = score(model, tok, devset, dev, a.max_len, a.batch_size * 4, ent, con)
        acc = sum((pp >= 0.5) == r["y"] for pp, r in zip(p, devset)) / max(1, len(devset))
        hist.append(
            {
                "epoch": ep + 1,
                "train_loss": tot / max(1, n),
                "dev_acc_at_0.5": acc,
                "elapsed_s": round(time.time() - t0, 1),
            }
        )
        print(json.dumps(hist[-1]), flush=True)

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    wfile = os.path.join(a.out, "model.safetensors")
    import transformers

    meta = {
        "base": "tiny-random" if a.tiny_random else a.base,
        "weights_sha256": C.file_sha(wfile),
        "manifest_sha256": man["sha256"],
        "bundle_sha256": C.file_sha(a.bundle),
        "train_items": len(train),
        "dev_items": len(devset),
        "smoke": bool(a.tiny_random or a.limit),
        "label_ids": {"entailment": ent, "contradiction": con},
        "hparams": {
            k: getattr(a, k)
            for k in ("epochs", "lr", "batch_size", "grad_accum", "max_len", "warmup", "weight_decay", "seed")
        },
        "amp": amp,
        "device": str(dev),
        "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "history": hist,
    }
    json.dump(meta, open(os.path.join(a.out, "train_meta.json"), "w"), indent=1)
    if not meta["smoke"]:  # aggregates only (no text): a committed record of what was trained
        os.makedirs(C.RESULTS, exist_ok=True)
        json.dump(
            meta,
            open(os.path.join(C.RESULTS, f"train_meta_{os.path.basename(os.path.normpath(a.out))}.json"), "w"),
            indent=1,
        )
    print("saved", a.out, "weights sha256", meta["weights_sha256"][:16], f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
