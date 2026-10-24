"""Shadow-mode runtime: run the frozen policy on an agent's claims WITHOUT changing what the agent does.

    from roman_wheel_node.shadow import ShadowRunner
    rwn = ShadowRunner("rwn_shadow/")                       # free local sources, shadow mode
    answer = rwn.wrap(agent_fn, extract=my_claims)(prompt)   # answer is the agent's own object, unchanged

* **Shadow (default).** Every claim is decided by the frozen policy through a hash-chained :class:`ClaimTrace`, and a
  row is appended to ``fresh_claims.jsonl`` (claim, evidence, raw source outputs, outcome) for later labelling and
  certification. The wrapped function's return value is returned as-is (the same object), its exceptions propagate
  unchanged, and any failure inside the node is caught, counted and logged as NOT_SURE: the node can never change,
  delay by error, or break the agent's output.
* **Enforce (opt-in, ``enforce=True``).** The output is released only if every extracted claim is ACCEPTed; otherwise
  ``on_withhold(output, verdicts)`` is returned (default: None). Fail closed: an error inside the node withholds.
* **Sources.** Default: the exact-copy gate + free local sources (overlap, ONNX NLI if available, nli2 if torch is
  installed) and BM25 retrieval over a trusted corpus when the agent gives no evidence. Nothing is sent to any
  third-party API. To decide with weights fitted on your own labelled data, pass ``policy=load_policy(path, sha)``
  for a file written by :mod:`roman_wheel_node.fit`.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .core import roman_wheel_node
from .fingerprint import exact_gate_v2, trusted_set
from .policy import ACCEPT, BLOCK, NOT_SURE, Policy, Profile, load_policy
from .sources import BM25, exact_gate, sha256
from .trace import ClaimTrace

__all__ = ["FrozenNode", "ShadowRunner", "Verdict"]


@dataclass(frozen=True)
class Verdict:
    outcome: str  # ACCEPT | NOT_SURE | BLOCK
    eff: Optional[float]
    reason: str
    claim_id: str = ""
    record_hash: str = ""
    seq: int = -1
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def admitted(self) -> bool:
        return self.outcome == ACCEPT


class FrozenNode:
    """The frozen policy on top of a hash-chained ClaimTrace. ``decide`` never raises for bad source data."""

    def __init__(
        self,
        log_path: str,
        policy: Optional[Policy] = None,
        profile: Optional[str] = None,
        hmac_key: Optional[bytes] = None,
    ) -> None:
        self.policy = policy or load_policy()
        self.prof: Profile = self.policy.profile(profile)
        fr = self.prof.frozen()
        self._thr = self.prof.accept_threshold()
        self.trace = ClaimTrace(
            log_path,
            live=fr,
            mode="bayes",
            variant=self.prof.variant,
            threshold=fr,
            prior=self.prof.prior,
            not_sure_band=(0.0, self._thr),
            hmac_key=hmac_key,
        )
        self._lock = threading.Lock()

    def decide(self, claim: str, raw: Mapping[str, Mapping[str, float]], a: int) -> Verdict:
        """raw = {source: {"S": .., "C": ..}} (uncalibrated). a = exact-copy gate bit."""
        a = 1 if (a is True or (type(a) in (int, float) and a == 1)) else 0
        with self._lock:
            try:
                sc = self.prof.calibrate(raw)
                eff, _, _ = roman_wheel_node(
                    a,
                    sc,
                    self.prof.frozen(),
                    mode="bayes",
                    variant=self.prof.variant,
                    threshold=self.prof.frozen(),
                    prior=self.prof.prior,
                )
                dis = a == 1 and eff > 0.0 and self.prof.disagrees(raw, eff)
                expect = (
                    BLOCK
                    if a != 1
                    else (ACCEPT if (not dis and eff >= self._thr and eff >= self.prof.cutoff) else NOT_SURE)
                )
                d = self.trace.submit(claim, sc, a=a, not_sure=dis)
                got = {"ACCEPT": ACCEPT, "BLOCK": BLOCK}.get(d.status, NOT_SURE)
                if got != expect:  # never trust a disagreement between the two paths
                    got, reason = (BLOCK if a != 1 else NOT_SURE), "internal_mismatch"
                else:
                    reason = {"ACCEPT": "accept_margin", "BLOCK": "exact_gate"}.get(
                        got, "disagreement" if dis else "below_margin"
                    )
                cid = d.claim.id if d.claim is not None else ""
                return Verdict(got, d.eff, reason, cid, self.trace.head(), d.seq)
            except Exception as e:  # missing source, bad score, anything: fail closed
                d = self.trace.submit(claim if isinstance(claim, str) else repr(claim), {}, a=a, not_sure=True)
                out = BLOCK if a != 1 else NOT_SURE
                miss = _missing_sources(self.prof, raw) if isinstance(raw, Mapping) else []
                bad = _malformed_scores(self.prof, raw) if isinstance(raw, Mapping) and not miss else []
                why = (
                    f"error:missing_source:{','.join(miss)}"
                    if miss
                    else (f"error:malformed_score:{','.join(bad)}" if bad else f"error:{type(e).__name__}")
                )
                return Verdict(out, None, why, "", self.trace.head(), d.seq)


def _jsonable(x: Any) -> Any:
    """JSON-safe copy for the fresh-claims log (non-finite floats and odd objects become strings)."""
    if isinstance(x, float):
        return x if math.isfinite(x) else repr(x)
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return repr(x)


def _stderr(line: str) -> None:
    """Best-effort report on stderr (the last place an error can go; it must not raise into the agent)."""
    try:
        print(line, file=sys.stderr, flush=True)
    except Exception:  # stderr itself is closed or unencodable: retry with ASCII escapes, then give up
        try:
            print(line.encode("ascii", "backslashreplace").decode("ascii"), file=sys.stderr, flush=True)
        except Exception:
            return


def _missing_sources(prof: Profile, raw: Mapping[str, Any]) -> List[str]:
    """Sources the profile needs that are absent from ``raw`` (no entry, or None)."""
    return [s for s in prof.sources if raw.get(s) is None]


def _malformed_scores(prof: Profile, raw: Mapping[str, Any]) -> List[str]:
    """Sources the profile needs that are present in ``raw`` but give no score ``Profile.calibrate`` accepts
    (an entry that is not ``{"S": ...}``, or an S that is not a number in [0, 1])."""

    def ok(v: Any) -> bool:
        try:
            x = float(v["S"])
        except Exception:  # any failure means unusable (an int past float range raises OverflowError): never raise
            return False
        return math.isfinite(x) and 0.0 <= x <= 1.0

    return [s for s in prof.sources if raw.get(s) is not None and not ok(raw[s])]


def _read_text_lines(stream: Any) -> List[str]:
    """Lines of a text stream decoded as UTF-8 regardless of the locale (stdin on Windows is cp1252 by default)."""
    buf = getattr(stream, "buffer", None)
    data = buf.read().decode("utf-8") if buf is not None else stream.read()
    # split on "\n" only: str.splitlines() also splits on U+2028, U+2029 and U+0085, which JSON written with
    # ensure_ascii=False carries raw inside strings (json.loads ignores a trailing "\r")
    return data.split("\n")


def _write_line(line: str) -> None:
    """Write one UTF-8 line to stdout regardless of the locale encoding."""
    buf = getattr(sys.stdout, "buffer", None)
    if buf is None:
        print(line)
        return
    sys.stdout.flush()
    buf.write((line + "\n").encode("utf-8"))
    buf.flush()


def _file_lines(ap: Any, path: str) -> List[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().split("\n")
    except UnicodeDecodeError as e:
        ap.error(f"{path}: not UTF-8 ({e})")
    except OSError as e:
        ap.error(f"cannot read {path}: {e.strerror or e}")


def _jsonl_objects(ap: Any, name: str, lines: List[str]) -> List[Tuple[int, dict]]:
    """(line number, object) for each non-blank line; a bad line is a usage error (exit 2), never a traceback."""
    out = []
    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except (ValueError, RecursionError) as e:
            ap.error(f"{name}: line {i}: not JSON ({type(e).__name__}: {str(e)[:200]})")
        if not isinstance(d, dict):
            ap.error(f"{name}: line {i}: expected a JSON object, got {type(d).__name__}")
        bad = _lone_surrogate(d)
        if bad is not None:
            ap.error(f"{name}: line {i}: not valid Unicode text (lone surrogate \\u{ord(bad):04x} from a JSON escape)")
        out.append((i, d))
    return out


_SURROGATE = re.compile("[\ud800-\udfff]")


def _lone_surrogate(obj: Any) -> Optional[str]:
    """The first unpaired surrogate (U+D800..U+DFFF) in any key or string of a parsed JSON value, else None.

    UTF-8 input cannot carry one, but a JSON escape such as ``"\\ud800"`` can; such a string can never be written out
    as UTF-8 (claims are echoed on stdout and logged), so it is rejected as invalid text. Iterative: values nested as
    deep as ``json.loads`` accepts must not hit the recursion limit."""
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, str):
            m = _SURROGATE.search(x)
            if m:
                return m.group()
        elif isinstance(x, dict):
            stack.extend(x.keys())
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
    return None


def _read_claims(ap: Any, claims: str) -> List[dict]:
    if claims == "-":
        if sys.stdin is None:  # fd 0 closed when Python started: there is no stdin object at all
            ap.error("stdin is closed: pipe the claims JSONL in, or give a claims file instead of '-'")
        try:
            lines = _read_text_lines(sys.stdin)
        except UnicodeDecodeError as e:
            ap.error(f"stdin: not UTF-8 ({e})")
        except (OSError, ValueError) as e:  # fd closed or invalid after start (ValueError: I/O on closed file)
            ap.error(f"cannot read stdin: {getattr(e, 'strerror', None) or e}")
        name = "stdin"
    else:
        lines, name = _file_lines(ap, claims), claims
    return [d for _, d in _jsonl_objects(ap, name, lines)]


ClaimSpec = Union[str, Tuple[str, Sequence[str]], Mapping[str, Any]]


class ShadowRunner:
    """Wrap an agent's claim stream. See the module docstring.

    Args:
        log_dir: directory for ``audit.jsonl`` (hash-chained ClaimTrace), ``fresh_claims.jsonl`` and ``errors.jsonl``.
            Keep it out of git (``*.jsonl`` is gitignored here); it holds claim and evidence text.
        policy / profile: frozen policy (default: the pinned bundled file) and profile (default ``local``).
        sources: callable ``(claim, sentences) -> raw`` (default: :class:`~roman_wheel_node.sources.LocalSources`
            built from ``nli_dir`` / ``nli2``; with neither, only ``overlap`` is available, so profiles fail closed).
        corpus: trusted ``{doc_id: [sentence, ...]}``; gives the exact-copy gate its hash set and BM25 retrieval.
        trusted_hashes: extra sha256 set for the gate (e.g. evidence the agent quoted from a verified store).
            Under an fp2 policy (the shipped default) entries may also be ``fingerprint.fp2`` values.
        trusted_units: extra trusted sentences (hashed with the policy's fingerprint version).
        enforce: False (default) = shadow; True = release outputs only when every claim is ACCEPTed.
        on_withhold: enforce-mode replacement, ``(output, verdicts) -> object`` (default returns None).
        Other keyword options are accepted only as None / False (options removed in 1.2.0, switched off).
    """

    def __init__(
        self,
        log_dir: str,
        policy: Optional[Policy] = None,
        profile: Optional[str] = None,
        sources: Optional[Callable] = None,
        corpus: Optional[Mapping[str, Sequence[str]]] = None,
        trusted_hashes: Optional[Iterable[str]] = None,
        enforce: bool = False,
        on_withhold: Optional[Callable] = None,
        nli_dir: Optional[str] = None,
        nli2: Optional[str] = None,
        hf_cache: Optional[str] = None,
        device: str = "cpu",
        retrieve_k: int = 1,
        hmac_key: Optional[bytes] = None,
        clock: Callable[[], float] = time.time,
        trusted_units: Optional[Iterable[str]] = None,
        **removed: object,
    ) -> None:
        # Options removed in 1.2.0 are still accepted when switched off (None / False), so callers written for the
        # pre-release API keep working; any other value would silently change behaviour, so it is refused.
        bad = sorted(k for k, v in removed.items() if v is not None and v is not False)
        if bad:
            raise TypeError(f"ShadowRunner: unsupported option(s) {bad} (removed in 1.2.0; only None/False accepted)")
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir, self.enforce, self.clock = log_dir, bool(enforce), clock
        self.node = FrozenNode(os.path.join(log_dir, "audit.jsonl"), policy, profile, hmac_key)
        if sources is None:
            from .sources import LocalSources

            sources = LocalSources(nli_dir=nli_dir, nli2=nli2, hf_cache=hf_cache, device=device)
        self.sources = sources
        self._warn_missing_sources()
        self.bm25 = BM25(corpus) if corpus else None
        self.fingerprint = self.node.policy.fingerprint
        hs = set(trusted_hashes or ())
        if self.bm25 is not None:
            hs |= self.bm25.hashes()
            if self.fingerprint == "fp2":
                hs |= trusted_set((s for v in self.bm25.docs.values() for s in v), "fp2")
        if trusted_units is not None:
            hs |= trusted_set(trusted_units, self.fingerprint)
        given = corpus is not None or trusted_hashes is not None or trusted_units is not None
        self.hashes = hs if (hs or given) else None
        self.on_withhold = on_withhold or (lambda output, verdicts: None)
        self.retrieve_k = retrieve_k
        self.fresh_path = os.path.join(log_dir, "fresh_claims.jsonl")
        self.errors = 0
        self.counts = {ACCEPT: 0, NOT_SURE: 0, BLOCK: 0}
        self._lock = threading.Lock()

    # ---- one claim ---------------------------------------------------------------------------
    def observe(self, claim: str, evidence: Optional[Sequence[str]] = None, meta: Optional[Mapping] = None) -> Verdict:
        """Decide and log one claim. Never raises (errors -> NOT_SURE, or BLOCK if the gate already failed)."""
        try:
            return self._observe(claim, evidence, meta)
        except Exception as e:  # last-resort guard: the agent must never see a node error
            self._error("observe", e)
            with self._lock:
                self.counts[NOT_SURE] += 1
            return Verdict(NOT_SURE, None, f"error:{type(e).__name__}")

    def _observe(self, claim, evidence, meta) -> Verdict:
        if not isinstance(claim, str):
            claim = str(claim)
        sents = [s for s in (evidence or []) if isinstance(s, str) and s.strip()]
        retrieved = False
        if not sents and self.bm25 is not None:
            sents, retrieved = self.bm25.retrieve(claim, self.retrieve_k), True
        fp2 = self.fingerprint == "fp2"
        a = exact_gate_v2(sents, self.hashes) if fp2 else exact_gate(sents, self.hashes)
        raw: Dict[str, Dict[str, float]] = {}
        try:
            raw.update(self.sources(claim, sents))
        except Exception as e:
            self._error("sources", e)
        v = self.node.decide(claim, raw, a)
        rec = {
            "id": f"{v.seq}:{v.record_hash[:16]}",
            "ts": float(self.clock()),
            "seq": v.seq,
            "record_hash": v.record_hash,
            "policy": self.node.policy.sha256,
            "profile": self.node.prof.name,
            "claim": claim,
            "evidence": sents,
            "retrieved": retrieved,
            "a": a,
            "raw": _jsonable(raw),
            "eff": v.eff,
            "outcome": v.outcome,
            "reason": v.reason,
            "meta": _jsonable(dict(meta or {})),
        }
        with self._lock:
            with open(self.fresh_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False, default=str) + "\n")
            self.counts[v.outcome] += 1
        return Verdict(v.outcome, v.eff, v.reason, v.claim_id, v.record_hash, v.seq, raw)

    # ---- wrapping an agent -------------------------------------------------------------------
    def check_output(self, output: Any, extract: Callable[[Any], Iterable[ClaimSpec]]) -> List[Verdict]:
        verdicts: List[Verdict] = []
        try:
            specs = list(extract(output))
        except Exception as e:
            self._error("extract", e)
            return [Verdict(NOT_SURE, None, "error:extract")]
        for sp in specs:
            if isinstance(sp, str):
                verdicts.append(self.observe(sp))
            elif isinstance(sp, Mapping):
                verdicts.append(self.observe(sp.get("claim", ""), sp.get("evidence"), sp.get("meta")))
            else:
                try:
                    c, ev = sp
                except Exception:
                    verdicts.append(Verdict(NOT_SURE, None, "error:bad_claim_spec"))
                    continue
                verdicts.append(self.observe(c, ev))
        return verdicts

    def wrap(self, agent_fn: Callable, extract: Callable[[Any], Iterable[ClaimSpec]] = lambda out: [out]) -> Callable:
        """Return ``agent_fn`` wrapped. Shadow: returns exactly what ``agent_fn`` returned (same object)."""
        runner = self

        def wrapped(*args, **kwargs):
            output = agent_fn(*args, **kwargs)  # agent exceptions propagate untouched
            verdicts = runner.check_output(output, extract)
            if not runner.enforce:
                return output
            if verdicts and all(v.outcome == ACCEPT for v in verdicts):
                return output
            return runner.on_withhold(output, verdicts)

        wrapped.__wrapped__ = agent_fn  # type: ignore[attr-defined]
        wrapped.rwn = runner  # type: ignore[attr-defined]
        return wrapped

    def stream(self, claims: Iterable[ClaimSpec]) -> Iterable[Tuple[ClaimSpec, Verdict]]:
        """Shadow a claim stream: yields every item unchanged together with its verdict."""
        for sp in claims:
            vs = self.check_output(sp, lambda x: [x])
            yield sp, vs[0]

    def verify(self):
        return self.node.trace.verify()

    def _warn_missing_sources(self) -> None:
        """Say once, on stderr, which source the profile needs but did not load (and why), instead of leaving every
        claim to fail closed with no visible cause."""
        avail = getattr(self.sources, "available", None)
        if avail is None:
            return
        miss = [s for s in self.node.prof.sources if s not in avail]
        if not miss:
            return
        errs = getattr(self.sources, "errors", {}) or {}
        why = "; ".join(errs.get(s, f"{s}: not configured (pass {'nli_dir' if s == 'nli' else s}=...)") for s in miss)
        _stderr(
            f"roman-wheel-node: profile {self.node.prof.name!r} needs source(s) {miss} that are not available: {why}. "
            f"Every claim will fail closed (NOT_SURE, reason error:missing_source:{','.join(miss)})."
        )

    def _error(self, where: str, e: BaseException) -> None:
        self.errors += 1
        msg = f"{type(e).__name__}: {e}"
        if any(t in msg for t in ("API_KEY", "TOKEN", "Bearer", "Authorization")):  # never log a credential
            msg = type(e).__name__
        try:
            with open(os.path.join(self.log_dir, "errors.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.time(), "where": where, "error": msg[:300]}) + "\n")
        except Exception as w:  # errors.jsonl unwritable: never lose the error silently, report it on stderr
            _stderr(
                f"roman-wheel-node: could not write {os.path.join(self.log_dir, 'errors.jsonl')} "
                f"({type(w).__name__}); node error in {where}: {msg[:300]}"
            )


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m roman_wheel_node.shadow",
        description="Shadow a JSONL claim stream ({claim, evidence?: [..]} per line) with the frozen policy.",
    )
    ap.add_argument("claims", help="input JSONL; '-' for stdin")
    ap.add_argument("--log-dir", default="rwn_shadow")
    ap.add_argument("--profile", default=None, help="profile of the policy (bundled: local (default) or local-lite)")
    ap.add_argument(
        "--policy",
        default="4.3.0",
        choices=["4.3.0"],
        help="bundled frozen policy (only 4.3.0 ships: local open models, fitted on SciFact train only)",
    )
    ap.add_argument("--policy-file", default=None, help="your own policy file (from python -m roman_wheel_node.fit)")
    ap.add_argument("--policy-sha256", default=None, help="expected sha256 of --policy-file (recommended)")
    ap.add_argument("--corpus", default=None, help="trusted corpus JSONL {doc_id, sentences:[..]} (gate + retrieval)")
    ap.add_argument(
        "--trust-given-evidence",
        action="store_true",
        help="treat evidence given in the stream as trusted exact copies (only if your pipeline verified it)",
    )
    ap.add_argument("--nli-dir", default=os.environ.get("RWN_NLI_DIR"))
    ap.add_argument("--nli2", default=os.environ.get("RWN_NLI2"))
    ap.add_argument("--hf-cache", default=os.environ.get("RWN_HF_CACHE"))
    ap.add_argument("--device", default=os.environ.get("RWN_DEVICE", "cpu"), help="cpu (default) or cuda (GPU)")
    ap.add_argument("--enforce", action="store_true", help="print only ACCEPTed claims (default: shadow, echo all)")
    args = ap.parse_args(argv)
    corpus = None
    if args.corpus:
        corpus = {}
        for i, d in _jsonl_objects(ap, args.corpus, _file_lines(ap, args.corpus)):
            if "doc_id" not in d or not isinstance(d.get("sentences"), list):
                ap.error(f"{args.corpus}: line {i}: needs 'doc_id' and a 'sentences' list")
            corpus[str(d["doc_id"])] = list(d["sentences"])
    items = _read_claims(ap, args.claims)
    trusted = {sha256(s) for it in items for s in it.get("evidence") or []} if args.trust_given_evidence else None
    pol = load_policy(args.policy_file, args.policy_sha256)
    if trusted is not None and pol.fingerprint == "fp2":
        from .fingerprint import trusted_set

        trusted |= trusted_set((s for it in items for s in it.get("evidence") or []), "fp2")
    r = ShadowRunner(
        args.log_dir,
        policy=pol,
        profile=args.profile,
        corpus=corpus,
        trusted_hashes=trusted,
        enforce=args.enforce,
        nli_dir=args.nli_dir,
        nli2=args.nli2,
        hf_cache=args.hf_cache,
        device=args.device,
    )
    for it, v in r.stream(items):
        if not args.enforce or v.outcome == ACCEPT:
            _write_line(json.dumps(it, ensure_ascii=False))
    r.verify()
    print(
        json.dumps(
            {
                "counts": r.counts,
                "errors": r.errors,
                "profile": r.node.prof.name,
                "available_sources": list(getattr(r.sources, "available", ())),
                "log_dir": args.log_dir,
            }
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
