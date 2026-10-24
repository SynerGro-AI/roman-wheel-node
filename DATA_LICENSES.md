# Data licences

Rule for anything that ships in the `roman-wheel-node` package (wheel or sdist): every policy, calibration or weights
file is fitted **only** on data whose licence permits commercial use, checked at the dataset's primary source. Data with
a non-commercial, missing or unclear licence may appear in the research history (`phase3/` to `phase5/`, aggregates
only in `results/`), but nothing fitted on it ships.

Checked on 2026-10-24 (America/Edmonton). Licence text is quoted from the primary source as of that date.

## Datasets

| Dataset | Licence (primary source) | Source URL | Commercial use OK? | Used in shipped files |
|---|---|---|---|---|
| **SciFact** claims + evidence annotations (`claims_*.jsonl`) | CC BY 4.0 | https://github.com/allenai/scifact/blob/master/LICENSE.md | **Yes**, with attribution | `policies/rwn-policy-4.3.0.json`: both profiles (`local`, `local-lite`) are fitted on SciFact train (799 claims, gold evidence). SciFact dev (300) is the held-out evaluation. |
| **SciFact** abstracts (`corpus.jsonl`, from S2ORC) | ODC-By 1.0 | https://github.com/allenai/scifact/blob/master/LICENSE.md | **Yes**, with attribution | Same: the evidence sentences scored for the 4.3.0 fit. |
| HealthVer | CC BY 4.0 (stated in the README since 2026-10-21) | https://github.com/sarrouti/HealthVer | Appears yes, but **excluded by the owner's decision** | None. It was half of the calibration pool of policies 4.0.0 / 4.1.0, which no longer ship. |
| Climate-FEVER | None stated (no LICENSE file; README silent; Hugging Face card says "unknown") | https://github.com/tdiggelm/climate-fever-dataset | **No (unclear)** | None. Research history only (Phase 3 transfer test). |
| COVID-Fact | None stated (no LICENSE file; README silent) | https://github.com/asaakyan/covidfact | **No (unclear)** | None as data. See "Structural veto" below for the one rule choice made on it. |

Licence text, verbatim:

* SciFact, `LICENSE.md` (commits 7e07702 and a5254a1, 2023-09-28; earlier versions said CC BY-NC 2.0, and the
  Hugging Face card `allenai/scifact` still shows that old licence, but the repository is the primary source):
  > All claims and evidence annotations -- in the files `claims_*.jsonl` -- are released under
  > [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  >
  > The abstracts in the corpus -- in the file `corpus.jsonl` -- are part of the Semantic Scholar S2ORC dataset and are
  > licensed under [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/).
  >
  > The code in this repository is licensed under the Apache 2.0 license.
* HealthVer, `README.md` (commit e4de7e8, 2026-10-21, by co-author Asma Ben Abacha):
  > This work is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0). Please cite
  > our paper if you use the HealthVer dataset or code.

  Its evidence sentences come from CORD-19 abstracts, whose licences vary by paper. HealthVer is excluded from every
  shipped file by the owner's decision. Adding it back would need the owner's approval.

## Attribution (required by CC BY 4.0 and ODC-By 1.0)

Policy 4.3.0 is fitted on SciFact: David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman
Cohan, Hannaneh Hajishirzi. "Fact or Fiction: Verifying Scientific Claims." EMNLP 2020. Claims and annotations are
CC BY 4.0. Abstracts are from S2ORC under ODC-By 1.0 (Allen Institute for AI). No SciFact text is redistributed in this
package: the policy holds only calibration breakpoints, reliabilities and thresholds.

## Models (not shipped; downloaded by the user at run time)

| Model | Licence (Hugging Face model card) | URL | Commercial use OK? | Used by |
|---|---|---|---|---|
| MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | MIT | https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | Yes | `nli2` source (`local` profile) |
| cross-encoder/nli-deberta-v3-small (`onnx/model_qint8_avx512_vnni.onnx`) | Apache 2.0 | https://huggingface.co/cross-encoder/nli-deberta-v3-small | Yes | `nli` source (`local-lite` profile) |

The models' own training data (MNLI, SNLI, FEVER, ANLI) is upstream of these models and governed by their licences.
This project fits nothing on those datasets.

## Hosted-model (xAI/Grok) outputs: not used in shipped files

No shipped file is fitted on, or contains, output of xAI/Grok or any other hosted model (LLM API). By the owner's
decision, the earlier `panel` profile, whose judge calibrators were fitted on xAI API outputs for SciFact train
claims, and all code that called the xAI API were removed from the package before release; that profile was never
published. Its measured results are kept in the repository only, as a benchmark comparison that is not used for
fitting anything (`results/phase6/xai_panel_ledger.md`), and `tests/test_no_xai_in_package.py` fails if any
reference to it reaches the installed package. The xAI Enterprise Terms of Service
(https://x.ai/legal/terms-of-service-enterprise, section 3.2, last updated 2026-08-14) restrict using Output "to
train any foundation models, large language models, or other artificial intelligence systems"; with no xAI output in
any shipped fit, that question no longer applies to this package.

## Fitting your own policy

`roman_wheel_node.fit` fits a policy from labelled examples you supply. The licence of those examples is yours to
check; the fit records whatever you pass as `data_licences` in the policy file. It uses only the local sources above
and makes no API calls.

## Structural veto (`structure.FROZEN`)

`roman_wheel_node.structure` is opt-in and is not used by the policy. Its three veto field names
(`identifier_mutation`, `fragment`, `near_copy_swap`) and integer constants were chosen in Phase 4b on half A of
COVID-Fact, whose licence is unclear. No COVID-Fact text or statistic is stored in the package. Re-checked on SciFact
train with the same pre-declared bar (`phase6/structural_scifact_43.py`, `results/phase6/structural_scifact_43.json`):
no field meets the bar there, and the frozen rule catches 1 false accept while losing 3 correct accepts of the 4.3.0
`local` profile's out-of-fold ACCEPTs. The rule is unchanged in 1.2.0. Changing or removing it is the owner's decision.

## Superseded policies

`rwn-policy-4.0.0.json` (sha256 254804f8…) and `rwn-policy-4.1.0.json` (sha256 2156f584…) were fitted on SciFact train
plus HealthVer train. `rwn-policy-4.2.0.json` (sha256 6f7bb9e3…) was fitted on SciFact train only but its `panel`
profile used xAI judge outputs. None of them was ever published; all are removed from the package in 1.2.0 and remain
in git history only (4.0.0 / 4.1.0: commit 3857154; 4.2.0: commit 69aa8db).
