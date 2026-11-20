# Pre-registration — written before any test-set number existed

**Question (the one the user ranked first): is *latent* agreement a usable reference-free quality
signal for machine translation — and does it beat the *string* version of the same idea?**

Everything below was fixed at ~22:55, before the 10 k test pool finished generating. Deviations are
logged at the bottom, not silently fixed.

## Design in one paragraph

One frozen translator `A = opus-mt-vi-en` (77 M) produces an 8-best pool per test sentence. One frozen
back-translator `B = opus-mt-en-vi` (77 M) turns every candidate back into Vietnamese. Two utilities
are then computed on exactly the same candidates:

- **string round-trip**: chrF and char-3-gram Jaccard between the back-translation and the source;
- **latent round-trip**: both Vietnamese strings are embedded with **A's encoder** (same model, same
  Vietnamese tokenizer → *one shared space*, unlike the cross-model spaces the project measured
  before), and agreement is measured as pooled cosine plus token-level nearest-neighbour
  precision / recall / F1.

Reranking = pick the candidate with the highest utility. No reference is used by any arm.

## Hypotheses (decided now, scored later)

| id | hypothesis | operational test | what would make it false |
|---|---|---|---|
| **H1** *(primary)* | latent round-trip ranks candidates better than string round-trip | Spearman(utility, COMET) and top-1 accuracy on the 10 k test pool; paired bootstrap over sentences for BLEU/chrF/COMET of the *selections* | ΔSpearman ≤ 0 **and** ΔBLEU CI straddles 0 |
| **H2** | reference-free **consensus** (chrF-MBR) is a stronger utility than either round-trip signal | same three diagnostics | chrF-MBR loses on ≥ 2 of the 3 |
| **H3** | adding latent features to a tiny learned ranker helps beyond seed noise | ΔSpearman(string+latent vs string) > sd over 3 seeds | Δ ≤ sd |
| **H4** *(control)* | 4 M LoRA parameters buy more than any training-free selection | BLEU of `A5_lora` vs best training-free arm | any training-free arm ≥ LoRA |

## Metrics and inference

- BLEU (`sacrebleu`, `13a`), chrF, COMET (`Unbabel/wmt22-comet-da`), all corpus-level on the full
  10 k test; **paired** bootstrap (1 000 resamples, same sentence indices for every arm).
- 2 k `core` subset reported separately so the historical `22.27 BLEU` for `opus-mt-vi-en` stays
  comparable.
- "Headroom captured" = (BLEU(arm) − BLEU(beam top-1)) / (BLEU(oracle) − BLEU(beam top-1)), where the
  oracle picks the best candidate by true COMET. This is the number that says whether a utility is
  *usable*, not merely correlated.

## Known threats to validity (stated before looking)

1. **Correlated candidates.** Beam-8 hypotheses share prefixes; a reranker's ceiling is small if the
   pool is redundant. Report the unique-string rate per row so the reader can judge.
2. **COMET as teacher and as judge.** The oracle, the learned ranker's labels and the final metric are
   the same model. Any "latent ≈ COMET" claim is therefore an upper bound on usefulness, not proof of
   human agreement. (Cheap partial check: report chrF too; it is a different instrument.)
3. **Domain.** MTet is law/news/medical, and earlier phases found ~1 in 12 misaligned pairs, so every
   absolute number is depressed; only *differences between arms* are interpretable.
4. **Sentence-level only.** Span-level food/culture phenomena are out of scope tonight.
5. **The back-translation is itself a model output.** A good hypothesis can back-translate badly
   (domain shift vi→en→vi). This is a property of the signal, not a bug in the code — but it caps
   H1's ceiling and is worth measuring: report the correlation of each signal with hypothesis length.

## Deviation log

*(completed 2026-11-20 04:40. Every change to the plan above, with its reason. Nothing here changes a
pre-registered verdict; the two places where it would are marked.)*

| # | when | deviation | reason | consequence |
|---|---|---|---|---|
| D1 | 22:53 | the `lp` utility (teacher-forced model confidence) was **dropped** from the feature set | `compute_transition_scores` materialises `batch × len × vocab` in fp32 → OOM on 4 GB; `sequences_scores` is absent for greedy Marian. Recomputing teacher-forced log-probs for 80 k candidate pairs costs ~20 min GPU for a baseline MBR already covers | the "model confidence" reference-free baseline is **missing** from the arm table. This is the largest gap in the pre-registered design |
| D2 | 23:37 | test/val sizes: **10 000 / 10 000**, not the historical 2 000 | the historical split was calibrated at 2 k, where ±0.5 BLEU is invisible; the pre-registration already assumed 10 k. `core=True` keeps the historical 2 000 a strict subset | the published 22.27 stays quotable; measurement noise drops to ~±0.3 BLEU |
| D3 | 00:05 | first feature run crashed (`MarianMTModel` has no `.encoder`) and was re-run | code bug, not design; back-translations were cached afterwards so a repeat costs 10 k strings, not 80 k | 24 min lost; no methodological change |
| D4 | 01:00 | bootstrap resamples: **1 000 on the 10 k runs, 300 on the 2 k replication runs** | re-tokenising 10 k sentences per resample cost ~5 min/arm; switching to sacrebleu sentence statistics made 1 000 resamples ~1 s/arm. The 2 k runs use 300 because they are secondary | CIs on the main table are exactly as pre-registered; the replication intervals are wider |
| D5 | 01:57 | H4 said "4 M LoRA parameters"; the run used **1 179 648** (r = 16 on 6+6 layers × 512) | r = 32 would have exceeded what a 6-layer 512-d model can absorb usefully; the mission's constraint was ≤ 5 M, and smaller is the conservative choice | H4 is tested with a *weaker* intervention than announced, so its confirmation is a lower bound. See also D6 |
| D6 | 04:26 | H4 was run at **two budgets** (60 k and 150 k pairs) instead of one | after the first run the returns were visibly not saturated; a single 0.4-epoch point would have invited "undertrained control" | both reported; the pre-registered verdict is unchanged and stronger |
| D7 | 03:05 | the reference-free QE baseline `Unbabel/wmt22-cometkiwi-da` was **not run** | gated repo, `GatedRepoError 401` (no HF credentials on this machine) | the strongest learned reference-free comparator is absent; `src/kiwi.py` is written and will run once the licence is accepted |

**No pre-registered hypothesis was altered, dropped, or re-defined after the fact.** The exploratory
work below was added *after* the four verdicts were computed, and is labelled as such.

## Verdicts (scored 2026-11-20 ~02:00, before any post-hoc analysis)

| id | verdict | evidence |
|---|---|---|
| **H1** | **holds as a scorer; fails as a usable signal** | see below |
| **H2** | **confirmed** | chrF-MBR beats latent by **+0.736 BLEU [+0.593, +0.885]** and string by **+0.748 [+0.612, +0.898]** (paired, 1 000 resamples, n = 10 000); it also wins chrF (46.59 vs 46.02 / 45.98) and loses COMET by 0.0005 (0.7609 vs 0.7614). Falsifier was "loses on ≥ 2 of 3" → it loses on 1 of 3 |
| **H3** | **confirmed statistically, useless practically** | ρ **0.557 ± 0.001** (string+latent) vs **0.496 ± 0.001** (string), Δ = +0.061 ≫ sd 0.001; but top-1 accuracy 0.355 vs 0.353 and BLEU 22.15 vs 22.09 (**Δ +0.055, CI [−0.056, +0.168]**), both far below plain beam (22.85) |
| **H4** | **confirmed, and larger than pre-registered** | LoRA **26.50 BLEU** at 60 k pairs / 7.5 min and **27.02** at 150 k / 19.3 min, vs best training-free arm 23.04 → gaps of **+3.46** and **+3.98**. No training-free arm came within 3 BLEU |

### H1 in detail (the primary hypothesis)

Pre-registered falsifier: *ΔSpearman ≤ 0 **and** ΔBLEU CI straddles 0*. Only the second conjunct holds,
so H1 is not falsified — but the pre-registration asked whether the signal is *usable*, and the answer
splits cleanly:

| diagnostic | string | latent | contrast |
|---|---|---|---|
| Spearman vs COMET (80 k candidates) | +0.394 | **+0.478** | **Δ +0.083 [+0.075, +0.092]**, bootstrap over sentences |
| top-1 accuracy | 0.356 | **0.370** | +0.014 |
| BLEU of the selection (10 k) | 22.29 | 22.31 | **Δ +0.012 [−0.101, +0.123]**, P(Δ≤0) = 0.397 |
| chrF of the selection | 46.02 | 45.98 | −0.04 |
| COMET of the selection (2 k core) | 0.7604 | 0.7614 | +0.0010 |
| headroom captured | −17 % | −16 % | both *below* plain beam |

**Reading.** The representation-space signal is a significantly better *predictor* and an
indistinguishable *decision rule*. The pre-registered "usability" clause therefore fails even though
the correlation clause passes — which is exactly the distinction the project keeps re-learning
(similarity ≠ usability), now at hypothesis level. The result replicates on the adapted model's pool
(2 k / 16 k candidates): string ρ +0.286, latent ρ +0.362, Δ +0.076; selection BLEU 25.71 vs 25.89,
both below that model's own top-1 26.66.

## Post-hoc (exploratory) results — added after the verdicts, not pre-registered

| # | experiment | result | status |
|---|---|---|---|
| P1 | decode-time length penalty (lp 1.2 / 1.5, beam-5, 2 k core) | +0.11 [−0.09, +0.34] / +0.06 [−0.20, +0.33] BLEU | negative; under-length is real but not caused by lp |
| P2 | chrF-MBR on the *adapted* pool | 25.81 vs 26.50 (−0.69 BLEU) | negative; a selection rule's value is a property of the pool |
| P3 | diverse pool (beam-8 + 8 samples, 2 k) | ceiling 26.40 vs 26.54; round-trip rules −1.4 BLEU; MBR +0.05 | negative; candidate *quality*, not count, sets the ceiling |
| P4 | idf-weighted latent agreement | ρ +0.458 vs +0.478; BLEU 22.28 vs 22.31 | negative; the shallow format component is part of the signal |
| P5 | consensus + agreement mixture `(1−w)·z(MBR) + w·z(latent)` | 23.04 → 23.06 → 22.87 → 22.53 → 22.31 for w = 0, .25, .5, .75, 1 | negative; not complementary |
| P6 | replication on the adapted pool | Δρ +0.076 (vs +0.083); selection also loses | **positive**: the H1 scoring gap is a property of the signal |

All six are reported with the same numbers in `REPORT.md` §8–§12 and in `PROGRESS.md` with timestamps.

## What this pre-registration got right, and what it missed

**Right.** Fixing the *pool* before the utility meant every contrast is paired, and pre-committing to
"headroom captured" prevented the most likely self-deception: +0.083 Spearman would have been written up
as a win if BLEU and top-1 accuracy had not been fixed as the deciding instruments in advance.

**Missed.** The design assumed the interesting failure would be *statistical* (does the signal
correlate?) and therefore under-specified the *decision* side: top-1 accuracy turned out to be the
binding quantity, and it was only added as a diagnostic, not as the primary test. A better
pre-registration would have made **headroom captured** the primary endpoint and correlation the
secondary one.

**A claim I would not make from this design.** That latent agreement is *not* a quality signal. Nine
rules, one pool family, one direction, one teacher (COMET) — the honest statement is the narrow one in
`REPORT.md`: better predictor, equally poor selector, and 3.38 BLEU of headroom that no reference-free
rule reached.

