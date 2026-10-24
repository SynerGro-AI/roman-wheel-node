# SciFact label review (for a human; nothing here changes any label or any reported number)

Claims where **every** Grok judge (8-9 judges, 5 models, 6 prompt styles) reading the *cited* abstracts gave the
same verdict, and that verdict differs from the SciFact gold label. Unanimity is not proof: the panel is
correlated (error φ 0.4-0.9), so a shared misreading is possible. Each case needs a domain expert.

SciFact is licensed for non-commercial use, so this file lists SciFact IDs only (claim id, abstract doc id,
0-based abstract sentence index), never the claim, title or abstract text. Look them up in your own copy of
SciFact (`claims_{split}.jsonl` by `id`, `corpus.jsonl` by `doc_id`, sentence index into `abstract`).

Dev (held-out): 9 of 300 claims (3.0%). Train subset: 8 of 350 (2.3%).

## dev

### dev:1385: gold **SUPPORT**, panel **CONTRADICT** (min judge confidence 0.90)
- Claim: SciFact dev claim id 1385
- Cited abstract: doc 306006
- Judge quote (verbatim): doc 306006 sentence 4
- Gold rationale: doc 306006 sentence 4
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### dev:593: gold **SUPPORT**, panel **CONTRADICT** (min judge confidence 0.90)
- Claim: SciFact dev claim id 593
- Cited abstract: doc 19675911
- Judge quote (verbatim): doc 19675911 sentence 5
- Gold rationale: doc 19675911 sentence 5
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### dev:808: gold **SUPPORT**, panel **CONTRADICT** (min judge confidence 0.85)
- Claim: SciFact dev claim id 808
- Cited abstract: doc 36606083
- Judge quote (verbatim): doc 36606083 sentence 3
- Gold rationale: doc 36606083 sentence 1
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### dev:907: gold **NEI**, panel **SUPPORT** (min judge confidence 0.85)
- Claim: SciFact dev claim id 907
- Cited abstract: doc 6923961
- Judge quote (verbatim): doc 6923961 sentence 1
- Gold rationale: none (NEI)
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### dev:1368: gold **SUPPORT**, panel **NEI** (min judge confidence 0.80)
- Claim: SciFact dev claim id 1368
- Cited abstract: doc 2425364
- Gold rationale: doc 2425364 sentence 9, doc 2425364 sentence 10
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### dev:783: gold **NEI**, panel **CONTRADICT** (min judge confidence 0.80)
- Claim: SciFact dev claim id 783
- Cited abstract: doc 40632104
- Judge quote (verbatim): doc 40632104 sentence 4
- Gold rationale: none (NEI)
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### dev:879: gold **CONTRADICT**, panel **SUPPORT** (min judge confidence 0.75)
- Claim: SciFact dev claim id 879
- Cited abstract: doc 8426046
- Judge quote (verbatim): doc 8426046 sentence 6
- Gold rationale: doc 8426046 sentence 3
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### dev:1292: gold **NEI**, panel **CONTRADICT** (min judge confidence 0.70)
- Claim: SciFact dev claim id 1292
- Cited abstract: doc 56893404
- Judge quote (verbatim): doc 56893404 sentence 3
- Gold rationale: none (NEI)
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### dev:693: gold **SUPPORT**, panel **CONTRADICT** (min judge confidence 0.55)
- Claim: SciFact dev claim id 693
- Cited abstract: doc 24088502
- Judge quote (lightly reworded): doc 24088502 sentence 8
- Gold rationale: doc 24088502 sentence 8, doc 24088502 sentence 9
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

## train

### train:44: gold **SUPPORT**, panel **NEI** (min judge confidence 0.85)
- Claim: SciFact train claim id 44
- Cited abstract: doc 56893404
- Gold rationale: doc 56893404 sentence 3
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### train:1386: gold **CONTRADICT**, panel **SUPPORT** (min judge confidence 0.82)
- Claim: SciFact train claim id 1386
- Cited abstract: doc 306006
- Judge quote (verbatim): doc 306006 sentence 4
- Gold rationale: doc 306006 sentence 4
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### train:1214: gold **NEI**, panel **CONTRADICT** (min judge confidence 0.80)
- Claim: SciFact train claim id 1214
- Cited abstract: doc 6493422
- Judge quote (verbatim): doc 6493422 sentence 1
- Gold rationale: none (NEI)
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### train:989: gold **NEI**, panel **CONTRADICT** (min judge confidence 0.77)
- Claim: SciFact train claim id 989
- Cited abstract: doc 9988425
- Judge quote (verbatim): doc 9988425 sentence 5
- Gold rationale: none (NEI)
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### train:651: gold **NEI**, panel **CONTRADICT** (min judge confidence 0.75)
- Claim: SciFact train claim id 651
- Cited abstract: doc 9433958
- Judge quote (verbatim): doc 9433958 sentence 3
- Gold rationale: none (NEI)
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### train:1193: gold **NEI**, panel **CONTRADICT** (min judge confidence 0.75)
- Claim: SciFact train claim id 1193
- Cited abstract: doc 20532591
- Judge quote (verbatim): doc 20532591 sentence 3
- Gold rationale: none (NEI)
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### train:169: gold **CONTRADICT**, panel **NEI** (min judge confidence 0.70)
- Claim: SciFact train claim id 169
- Cited abstract: doc 5824985
- Judge quote (verbatim): doc 5824985 sentence 11
- Gold rationale: doc 5824985 sentence 11
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

### train:610: gold **NEI**, panel **CONTRADICT** (min judge confidence 0.63)
- Claim: SciFact train claim id 610
- Cited abstract: doc 40096222
- Judge quote (verbatim): doc 40096222 sentence 0
- Gold rationale: none (NEI)
- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous

