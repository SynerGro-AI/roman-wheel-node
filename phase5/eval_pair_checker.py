"""Phase 5 Task A: pre-register the checker's veto thresholds on A_dev, then score the held-out half B ONCE.

    python phase5/eval_pair_checker.py prereg  --model phase5/models/pairs-v1     # A_dev only; writes prereg_pairs.json
    python phase5/eval_pair_checker.py heldout --model phase5/models/pairs-v1     # B once; writes heldout_B_pairs.*

Both refuse to overwrite. 'heldout' refuses unless prereg_pairs.json exists and matches the protocol, the split
manifest, the bundle and the model weights. Add --smoke for the box smoke test: it uses a stand-in for B made from
A_dev (B is never read), writes only to phase5/runs/smoke/, and never touches results/.
Committed outputs are aggregates only (COVID-Fact and HealthVer are internal, unlicensed). Per-item scores stay in
the gitignored phase5/runs/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import pairs_common as C  # noqa: E402

from roman_wheel_node import fresh as FR  # noqa: E402

SMOKE = os.path.join(C.HERE, "runs", "smoke")


def now():
    return dt.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def load_protocol():
    p = json.load(open(C.PROTOCOL))
    if p["sha256"] != C.canon_sha({k: v for k, v in p.items() if k != "sha256"}):
        sys.exit("protocol.json does not match its sha256")
    return p


def check_model(model_dir, bundle_path, manifest):
    meta = json.load(open(os.path.join(model_dir, "train_meta.json")))
    w = C.file_sha(os.path.join(model_dir, "model.safetensors"))
    if w != meta["weights_sha256"]:
        sys.exit("model weights do not match train_meta.json")
    if meta["manifest_sha256"] != manifest["sha256"]:
        sys.exit("model was trained on a different split")
    if meta["bundle_sha256"] != C.file_sha(bundle_path):
        sys.exit("bundle differs from the one the model was trained with")
    return meta


class Scorer:
    def __init__(self, model_dir, device, batch_size, max_len):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        sys.path.insert(0, C.HERE)
        from train_pair_checker import label_ids, score

        self.dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.dev)
        self.ent, self.con = label_ids(self.model.config)
        self._score, self.bs, self.max_len = score, batch_size, max_len

    def __call__(self, rows):
        return (
            self._score(self.model, self.tok, rows, self.dev, self.max_len, self.bs, self.ent, self.con) if rows else []
        )


def judged(rows, base):
    return [r for r in rows if r.get(base) is not None]


def table(rows, ps, taus, bases, alphas, delta):
    """rows aligned with ps (P(support) or None). -> {base: {setting: stats}} plus correct accepts lost."""
    out = {}
    for base in bases:
        idx = [i for i, r in enumerate(rows) if r.get(base) is not None]
        if not idx:
            continue
        before = [(rows[i][base], rows[i]["y"]) for i in idx]
        s0 = FR.stats(before, alphas, delta)
        out[base] = {"none": s0}
        for name, tau in taus:
            after = [(C.apply_veto(rows[i][base], ps[i], tau), rows[i]["y"]) for i in idx]
            s = FR.stats(after, alphas, delta)
            s["correct_accepts_lost"] = sum(
                1 for (o0, y), (o1, _) in zip(before, after) if y and o0 == "ACCEPT" and o1 != "ACCEPT"
            )
            s["false_accepts_removed"] = sum(
                1 for (o0, y), (o1, _) in zip(before, after) if not y and o0 == "ACCEPT" and o1 != "ACCEPT"
            )
            out[base][name] = s
    return out


def auc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y]
    neg = [p for p, y in zip(ps, ys) if not y]
    if not pos or not neg:
        return None
    w = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return w / (len(pos) * len(neg))


def line(name, s0, s):
    def f(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"

    return (
        f"{name}: labelled {s0['labelled']}; ACCEPT {s0['accepted']} -> {s['accepted']}; false accepts "
        f"{s0['false_accepts']} -> {s['false_accepts']}; precision {f(s0['precision'])} -> {f(s['precision'])}; "
        f"FA-share UB {f(s0['false_accept_share_upper'])} -> {f(s['false_accept_share_upper'])}; coverage "
        f"{f(s0['coverage'])} -> {f(s['coverage'])}; correct accepts lost {s.get('correct_accepts_lost', 0)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prereg", "heldout"])
    ap.add_argument("--model", default=os.path.join(C.HERE, "models", "pairs-v1"))
    ap.add_argument("--bundle", default=C.BUNDLE)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=None, help="default: the training max_len")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-n", type=int, default=60)
    a = ap.parse_args()

    P = load_protocol()
    man = C.load_manifest()
    b = C.load_bundle(a.bundle)
    if P["split_manifest_sha256"] != man["sha256"] or b["manifest_sha256"] != man["sha256"]:
        sys.exit("protocol / bundle / manifest disagree")
    meta = check_model(a.model, a.bundle, man)
    if meta.get("smoke") and not a.smoke:
        sys.exit("this is a smoke-test model; the real run needs a model trained without --tiny-random/--limit")
    max_len = a.max_len or meta["hparams"]["max_len"]
    alphas, delta = P["certification"]["alphas"], P["certification"]["delta"]
    out_dir = SMOKE if a.smoke else C.RESULTS
    runs = SMOKE if a.smoke else os.path.join(C.HERE, "runs", "heldout")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(runs, exist_ok=True)
    prereg_path = os.path.join(out_dir, "prereg_pairs.json") if a.smoke else C.PREREG
    held_path = os.path.join(out_dir, "heldout_B_pairs.json") if a.smoke else C.HELDOUT
    adev = [r for r in b["covidfact"] if r["role"] == "A_dev"]
    if a.smoke:  # disjoint halves of A_dev: one plays A_dev, the other plays B. B itself is never read.
        adev.sort(key=lambda r: C.sha("smoke:" + r["id"]))
        stand_in_b = adev[a.smoke_n : 2 * a.smoke_n]
        adev = adev[: a.smoke_n]
    sc = Scorer(a.model, a.device, a.batch_size, max_len)

    if a.step == "prereg":
        if os.path.exists(prereg_path):
            sys.exit(f"{prereg_path} exists: thresholds are fixed once. Refusing to overwrite.")
        if os.path.exists(held_path):
            sys.exit("held-out results already exist; a prereg written now would not be a pre-registration")
        ps = sc(adev)
        true_ps = [p for p, r in zip(ps, adev) if r["y"]]
        taus = [(t["name"], C.tau_for_retention(true_ps, t["retention"])) for t in P["tau_sequence"]]
        dev_tab = table(adev, ps, taus, ["panel_4b", "panel", "local"], alphas, delta)
        doc = {
            "format": "rwn-phase5-prereg",
            "written": now(),
            "smoke": a.smoke,
            "protocol_sha256": P["sha256"],
            "manifest_sha256": man["sha256"],
            "bundle_sha256": C.file_sha(a.bundle),
            "model_weights_sha256": meta["weights_sha256"],
            "model_base": meta["base"],
            "max_len": max_len,
            "taus": {n: t for n, t in taus},
            "tau_order": [n for n, _ in taus],
            "primary": P["primary"],
            "a_dev": {
                "items": len(adev),
                "true": len(true_ps),
                "auc": auc(ps, [r["y"] for r in adev]),
                "by_base": {
                    base: {
                        k: {kk: v[kk] for kk in ("labelled", "accepted", "false_accepts", "precision", "coverage")}
                        for k, v in d.items()
                    }
                    for base, d in dev_tab.items()
                },
            },
            "note": C.NOTE,
        }
        doc["sha256"] = C.canon_sha(doc)
        json.dump(doc, open(prereg_path, "w"), indent=1, sort_keys=True)
        print("prereg", prereg_path, "taus", {n: round(t, 4) for n, t in taus}, "A_dev AUC", doc["a_dev"]["auc"])
        return

    # heldout
    if os.path.exists(held_path):
        sys.exit(f"{held_path} exists: the held-out half is scored once. Refusing.")
    if not os.path.exists(prereg_path):
        sys.exit("no prereg_pairs.json: run the prereg step first")
    pre = json.load(open(prereg_path))
    if pre["sha256"] != C.canon_sha({k: v for k, v in pre.items() if k != "sha256"}):
        sys.exit("prereg_pairs.json was edited")
    for k, v in (
        ("protocol_sha256", P["sha256"]),
        ("manifest_sha256", man["sha256"]),
        ("bundle_sha256", C.file_sha(a.bundle)),
        ("model_weights_sha256", meta["weights_sha256"]),
    ):
        if pre[k] != v:
            sys.exit(f"prereg {k} does not match the current file")
    taus = [(n, pre["taus"][n]) for n in pre["tau_order"]]
    B = stand_in_b if a.smoke else [r for r in b["covidfact"] if r["role"] == "B_test"]
    started = now()
    psB = sc(B)
    noreg = [r for r in b["noreg"] if "ACCEPT" in (r["panel"], r["panel_4b"], r["local"])] if not a.smoke else []
    if a.smoke:
        noreg = [r for r in b["noreg"] if "ACCEPT" in (r["panel"], r["panel_4b"])][:20]
    psN = sc(noreg)
    # per-item scores stay local (gitignored)
    with open(os.path.join(runs, "scores.jsonl"), "w") as f:
        for r, p in list(zip(B, psB)) + list(zip(noreg, psN)):
            f.write(json.dumps({"id": r["id"], "p_support": p}) + "\n")

    bases = ["panel_4b"] + P["secondary_bases"]
    res = {
        "format": "rwn-phase5-heldout",
        "smoke": a.smoke,
        "prereg_sha256": pre["sha256"],
        "prereg_written": pre["written"],
        "scored_started": started,
        "scored_finished": now(),
        "protocol_sha256": P["sha256"],
        "model_weights_sha256": meta["weights_sha256"],
        "taus": pre["taus"],
        "primary": P["primary"],
        "covidfact_B": {
            "items": len(B),
            "auc": auc(psB, [r["y"] for r in B]),
            "by_base": table(B, psB, taus, bases, alphas, delta),
        },
    }
    prim = res["covidfact_B"]["by_base"]["panel_4b"]
    res["ltt_fixed_sequence_panel_4b"] = C.fixed_sequence_ltt([(n, prim[n]) for n, _ in taus], alphas)
    # no regression on the other datasets: rows not accepted by any base need no score (the veto cannot change them)
    pmap = {r["id"]: p for r, p in zip(noreg, psN)}
    res["no_regression"] = {}
    for ds in C.NOREG_SETS:
        rows = [r for r in b["noreg"] if r["ds"] == ds]
        if a.smoke:
            rows = [r for r in rows if r["id"] in pmap or "ACCEPT" not in (r["panel"], r["panel_4b"])][:200]
        ps = [pmap.get(r["id"]) for r in rows]
        t = table(rows, ps, [(P["primary"]["setting"], pre["taus"][P["primary"]["setting"]])], bases, alphas, delta)
        s0, s1 = t["panel_4b"]["none"], t["panel_4b"][P["primary"]["setting"]]
        correct0 = s0["accepted"] - s0["false_accepts"]
        ok = (s1["precision"] if s1["precision"] is not None else 1.0) >= (s0["precision"] or 0.0) and s1[
            "correct_accepts_lost"
        ] <= max(1, 0.05 * correct0)
        res["no_regression"][ds] = {"pass": bool(ok), "by_base": t}
    ps_name = P["primary"]["setting"]
    res["success"] = bool(
        prim[ps_name]["alphas"]["0.05"]["certified"]
        and ps_name in res["ltt_fixed_sequence_panel_4b"]["0.05"]["certified"]
        and all(v["pass"] for v in res["no_regression"].values())
    )
    res["note"] = C.NOTE
    json.dump(res, open(held_path, "w"), indent=1, sort_keys=True)

    L = [
        f"Phase 5 pair checker{' (SMOKE: stand-in B from A_dev, tiny model; numbers mean nothing)' if a.smoke else ''}",
        f"prereg {pre['sha256'][:12]} written {pre['written']}; scored {started}; model {meta['weights_sha256'][:12]} "
        f"({meta['base']}); taus " + ", ".join(f"{n}={t:.4f}" for n, t in taus),
        "",
    ]
    L.append(f"covidfact_B AUC {res['covidfact_B']['auc']}")
    for base in bases:
        d = res["covidfact_B"]["by_base"].get(base)
        if not d:
            continue
        for n, _ in taus:
            L.append(line(f"  {base} + checker {n}", d["none"], d[n]))
            L.append("     certified: " + ", ".join(f"{al}: {d[n]['alphas'][f'{al}']['certified']}" for al in alphas))
    L.append("LTT fixed sequence (panel_4b): " + json.dumps(res["ltt_fixed_sequence_panel_4b"]))
    for ds, v in res["no_regression"].items():
        d = v["by_base"]["panel_4b"]
        L.append(f"no-regression {ds}: {'PASS' if v['pass'] else 'FAIL'}  " + line("panel_4b", d["none"], d[ps_name]))
    L += ["", f"SUCCESS (primary certified at 5% and no regression): {res['success']}", "", C.NOTE]
    open(held_path.replace(".json", ".txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
