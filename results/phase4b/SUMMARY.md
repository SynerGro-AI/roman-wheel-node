# Phase 4b: deterministic structural veto (held-out result)

The veto is deterministic and only ever changes ACCEPT to NOT_SURE (`src/roman_wheel_node/structure.py`).
- **Rule:** 4b.0, sha256 `dc2299c9a92c…`.
- **Veto fields:** identifier_mutation, fragment, near_copy_swap.
- **Tuning:** designed on COVID-Fact half A only.
- **Pre-registration:** 2026-10-22T10:50:34-0600, committed in 46e8f50 before scoring.
- **Scoring:** half B was scored once at 10:50:52-0600.
- **Panel policy:** 254804f8ac9b, unchanged.

Claims, evidence and judge quotes are compared as a per-field structural vector (numbers, negations, identifiers, content words, absolutes, hedges, directions). The fields are never summed into a scalar. The Rule 24 export is a per-field tuple plus a digest.

| set | ACCEPT before → after | false accepts | precision | correct accepts lost |
|---|---|---|---|---|
| COVID-Fact half B (held out) | 301 → 289 | 85 → 77 | 71.8% → 73.4% | 4 |
| COVID-Fact half A (design, optimistic) | 254 → 229 | 63 → 40 | 75.2% → 82.5% | 2 |
| SciFact dev | 63 → 63 | 1 → 1 | 98.4% → 98.4% | 0 |
| HealthVer test | 22 → 22 | 0 → 0 | 100% → 100% | 0 |
| Climate-FEVER | 36 → 32 | 0 → 0 | 100% → 100% | 4 |

**Half B in detail**
- **Coverage lost:** 0.65 points (16.2% → 15.6%).
- **False-accept share, one-sided 95% Clopper-Pearson upper bound:** 32.8% → 31.3%.
- **Certification:** nothing is certified at 1%, 2% or 5%. Those levels need 299, 149 and 59 error-free accepts respectively.
- **Caught on B, by field (false accepts / correct accepts lost):**
  - identifier mutation: 4 / 2
  - near-copy word swap: 3 / 2
  - fragment: 1 / 0
- **Missed false accepts on B:** 72 of 77 show no structural signal, and 5 show only a missing number. Numbers do not veto, because on half A they cost 7 correct accepts to catch 2.

**Reading.** The precision on half A did not carry over to half B: the catch ratio fell from 23:2 to 8:4. Most COVID-Fact false accepts are not surface-structural, so meaning-level errors need another signal.

Data handling: aggregates only. No COVID-Fact or HealthVer text or per-item output is committed.
