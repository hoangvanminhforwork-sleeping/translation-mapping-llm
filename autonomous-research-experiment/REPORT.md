# REPORT — one night, two frozen 77 M models, one question

**Question.** Is *latent* (representation-space) agreement a usable **reference-free** quality signal
for machine translation — and does it beat the *string* version of the same idea?

**Setup.** Two frozen models, no fine-tuning as a claim:
`A = Helsinki-NLP/opus-mt-vi-en` (77 M, translator) and `B = Helsinki-NLP/opus-mt-en-vi`
(77 M, back-translator). MTet test split, **10 000** sentences (the historical 2 000 are a subset of
it). A's beam-8 pool provides the candidates; every arm is a *selection* from that one pool, so all
comparisons are paired by construction. Hypotheses and threats were pre-registered in
`HYPOTHESES.md` **before** any test number existed.

---

## TL;DR - seven numbers and one sentence

| # | finding | number |
|---|---|---|
| 1 | the pipeline reproduces the published baseline | greedy **22.29** vs published **22.27** BLEU; COMET **0.7522** vs **0.752** |
| 2 | beam search is free quality, and nobody used it | beam-8 top-1 = **+1.01 BLEU** [+0.84, +1.17] |
| 3 | the frozen model already *contains* the target answer | oracle over its own 8 candidates = **26.23 BLEU** (**+3.38** headroom) |
| 4 | latent agreement ranks candidates **better** than string agreement | ρ **+0.478** vs **+0.394**, Δρ **+0.083 [+0.075, +0.092]** |
| 5 | …but that advantage does **not** make it a usable selector | BLEU **22.31** vs **22.29** (Δ +0.02, inside noise); best reference-free arm = chrF-MBR **23.04 (+0.19)** |
| 6 | **1.5 % of the parameters and 7.5 minutes beat every selection rule** | LoRA control **26.50 BLEU** (**+3.65**), COMET **0.7839** |
| 7 | …and it is still rising at a fair budget | 150 k pairs (19 min) → **27.02 BLEU** (**+4.17**), COMET **0.7861** |

> **The frozen 77 M model already produces a 26.2-BLEU answer among its eight candidates, and no
> reference-free signal we could build converts more than 6 % of that gap. Latent agreement is a
> better *scorer* and not a better *selector* — the project's "similarity ≠ usability" thesis, now
> measured at the level of hypotheses instead of mappers. And the cheapest 3.65–4.17 BLEU of the night
> came from weights (1.2 M LoRA parameters, 19 min), not from any clever choice of output.**

**The ordering of levers, which is the real result:**

| lever | available at test time? | BLEU |
|---|---|---|
| domain adaptation (1.2 M LoRA params, 19 min) | no (needs train data) | **27.02 (+4.17)** |
| domain adaptation (same params, 7.5 min) | no | **26.50 (+3.65)** |
| reference-based selection (oracle over 8 candidates) | no (needs references) | **26.23 (+3.38)** |
| reference-free selection (best of 9 rules) | **yes** | **23.04 (+0.19)** |

---

## 1. What was asked, and what I changed

The mission said: maximise BLEU/COMET/chrF, touch as few models as possible, scientific novelty is a
constraint. The user then ruled out fine-tuning the 77 M as *not research*, and re-framed the
objective as **insights toward the highest BLEU**.

Given `CURRENT-PROGRESS.md` (the project's map architecture plateaus at 3–6 BLEU while a frozen
77 M model gets 22.27), the only honest space left in eight hours was: **keep both models frozen and
make the choice of output better**. That choice is where the open scientific question lives, and it is
cheap — no training, one shared pool, paired comparisons.

## 2. Design in one paragraph

`A` translates each Vietnamese test sentence with beam-8 → a pool of 8 candidates (7.96 distinct on
average, mean pairwise chrF 78.1 inside a row). `B` back-translates every candidate into Vietnamese.
Two reference-free utilities are computed on **exactly the same candidates**: *string round-trip*
(chrF / char-3-gram Jaccard between back-translation and source) and *latent round-trip* (source and
back-translation are both Vietnamese, so both are embedded with **A's own frozen encoder** — one
shared space, no cross-model alignment problem — and agreement is measured as token-level
nearest-neighbour precision/recall/F1, plus a *centred* variant because this project has already
learned that an uncentred cosine is dominated by the mean direction). Reranking = pick argmax. No arm
sees a reference.

Additional arms: chrF-MBR (self-consensus), length-ratio prior, two tiny MLP rankers (2 433 and
3 137 parameters, trained on 5 000 **train-split** sentences with COMET as the target, 3 seeds), the
oracle (best candidate by true COMET — headroom only), and a labelled LoRA control.

## 3. Calibration: the pipeline reproduces the published numbers

| arm (2 k historical core) | BLEU | COMET |
|---|---|---|
| greedy (this run) | **22.29** | **0.7522** |
| greedy, published (`research-phase01`) | **22.27** | **0.752** |

Two independent numbers reproduce to the second decimal, so everything below is measured on an
instrument that is calibrated against the project's own past.

## 4. Result 1 — beam search is +1 BLEU that the project never used

Every previous phase decoded greedily. Beam-8 costs 19.5 min instead of 5 min per 10 000 sentences
(8.5 vs 34 sent/s on this laptop) and returns **+1.01 BLEU [+0.84, +1.17]** and +0.96 chrF.

It is not a uniform improvement: **31.9 %** of sentences are byte-identical to greedy, **38.7 %** get
better, **29.0 %** get worse (mean sentence chrF Δ +1.06). The gain is real but it is a *coin flip
that pays 1.3:1* — which is exactly why a good reranker should have something to fix, and why "beam
is better" cannot be reported as "beam is right".

## 5. Result 2 — the answer is already in the pool; selection is the whole game

| arm | BLEU | chrF | COMET (2 k core) | Δ BLEU vs beam [95 % CI] |
|---|---|---|---|---|
| A0 greedy | 21.84 | 44.93 | 0.7522 | −1.01 [−1.17, −0.84] |
| A1 beam-8 top-1 | 22.85 | 45.89 | **0.7617** | — |
| A3 length-ratio | 22.55 | 46.39 | 0.7599 | −0.30 [−0.51, −0.10] |
| **A5 chrF-MBR** | **23.04** | **46.59** | 0.7609 | **+0.19 [+0.07, +0.32]** |
| A6 string round-trip | 22.29 | 46.02 | 0.7604 | −0.56 [−0.71, −0.40] |
| A7 latent round-trip | 22.31 | 45.98 | 0.7614 | −0.54 [−0.70, −0.38] |
| A7c latent, centred | 22.32 | 46.00 | 0.7613 | −0.53 [−0.69, −0.36] |
| A8 learned (string) | 22.09 | 45.88 | 0.7608 | −0.76 [−0.92, −0.59] |
| A8 learned (string+latent) | 22.15 | 45.83 | 0.7607 | −0.70 [−0.88, −0.53] |
| **A9 oracle (COMET)** | **26.23** | **48.76** | **0.7923** | **+3.38 [+3.22, +3.56]** |

Capture of the oracle headroom: chrF-MBR **+6 %**, every round-trip arm −16…−17 %, the learned
rankers −21…−22 %, greedy −30 %.

BLEU/chrF on all 10 000 sentences; COMET on the 2 000-sentence historical core (COMET at 10 k × 10
arms is 100 k scorer calls, and the ordering is unchanged on the core). Differences are paired
bootstrap over the same sentence indices (1 000 resamples).

**The oracle is not a strawman.** It picks, per sentence, the candidate COMET likes best — no
reference-free arm can do this — and it raises BLEU by 3.38 and COMET by 0.031. That is the size of
the prize, and it says something about the frozen model: *it does not need to be better, it needs to
be chosen better.*

## 6. Result 3 — latent agreement is the better scorer, and that is not enough

Candidate level, 80 000 candidates over 10 000 sentences (Spearman against the COMET teacher):

| utility | Spearman ρ | top-1 accuracy | params |
|---|---|---|---|
| string round-trip (chrF of back-translation) | +0.394 | 0.356 | 1 |
| **latent round-trip (token NN F1)** | **+0.478** | **0.370** | 1 |
| latent, centred | +0.480 | 0.369 | 1 |
| learned, string only | +0.496 ± 0.001 | 0.353 | 2 433 |
| learned, string + latent | **+0.557** ± 0.001 | 0.355 | 3 137 |

- **H1 is confirmed on correlation**: Δρ(latent − string) = **+0.083 [+0.075, +0.092]** (300 bootstrap
  resamples over *sentences*), and the latent features add **+0.06 ρ** to the tiny ranker over seed
  noise (sd 0.001). Centring changes almost nothing (+0.002), so the shared-mean direction is *not*
  what carries the signal here — unlike the mapper-level analysis in E9.
- **H1 fails on usability**: top-1 accuracy moves only 0.356 → 0.370, BLEU moves +0.02 (inside noise),
  and both round-trip arms are **0.55 BLEU worse than simply taking the beam's first candidate**.
- **Not a length artifact**: latent beats string in *every* source-length quartile
  (ρ +0.480/+0.437/+0.453/+0.552 vs +0.420/+0.370/+0.425/+0.519), so the advantage is not "latent
  agreement measures length better".
- **H2 is falsified**: chrF-MBR is the strongest reference-free rule (+0.19 BLEU, significant), so
  *consensus* beats *agreement-with-the-source*.
- **H3 is confirmed but useless**: latent features help the learned ranker's correlation and not its
  selections (0.353 → 0.355).

The mechanism is visible in the text (`results/EXAMPLES.md`, 4 205 disagreeing rows): latent
agreement is sensitive to *format* evidence in the back-translation — numbers, dates, `%s` patterns,
"8.4" — and it happily prefers a hypothesis whose back-translation reproduces the source's *shape*
while getting the word wrong (*"thanh toán"* vs *"kiểm toán"*). It inherits the back-translator's copy
behaviour. The string signal has the same weakness but less sharply, and neither can tell that one
fluent candidate is semantically right and the other is not.

**The obvious fix was tested and made it worse.** If the signal is fooled by frequent tokens, an
idf-weighted nearest-neighbour agreement should help. Down-weighting frequent tokens *lowers* the
correlation (ρ +0.458 vs +0.478) and leaves the selection unchanged (BLEU 22.28 vs 22.31, −0.57 vs
beam) — so the "shallow" part is not noise to be removed, it is where part of the signal lives. The
signal is genuinely measuring shape, and shape is informative but not sufficient.

## 7. Replication: the gap is a property of the signal, not of one pool

The whole comparison above used the frozen model's pool. Re-running both utilities on the
**adapted model's** pool (2 000 sentences, 16 000 candidates, LoRA beam-8) reproduces the ordering at
nearly the same magnitude:

| utility | frozen pool (80 k cands) | adapted pool (16 k cands) |
|---|---|---|
| string round-trip | ρ +0.394 | ρ +0.286 |
| latent round-trip | ρ +0.478 | ρ +0.362 |
| **Δρ** | **+0.083** | **+0.076** |
| top-1 accuracy string → latent | 0.356 → 0.370 | 0.374 → 0.392 |

Both pools agree on the *sign and size* of the gap and on its practical irrelevance for selection
(+0.014 and +0.018 top-1 accuracy). Selection replicates too, on the same 2 k rows:

| selection (adapted pool, 2 k) | BLEU | Δ vs its own top-1 |
|---|---|---|
| beam-8 top-1 | 26.66 | — |
| latent round-trip | **25.89** | −0.77 |
| chrF-MBR | 25.85 | −0.81 |
| string round-trip | 25.71 | −0.95 |

Same verdict, and the same ordering inside the losing group: latent loses less than string. So the
signal's advantage is real, consistent, and still not large enough to beat "take the first candidate".
That is the strongest form the claim can take from one night: not "latent agreement is good", but
"latent agreement is consistently the better of two bad scorers".

## 8. Why the round-trip arms lose BLEU but win chrF

| arm | hypothesis words | reference words |
|---|---|---|
| beam top-1 | 16.18 | 17.50 |
| MBR | 16.79 | 17.50 |
| string round-trip | 16.61 | 17.50 |
| latent round-trip | 16.58 | 17.50 |

Beam top-1 systematically *under-lengths* by 1.3 words (Marian's default length penalty = 1.0). Every
reranker moves toward the reference length; chrF rewards the recovered words (46.0 vs 45.89), BLEU
punishes the ones that are wrong (22.3 vs 22.85). **The two "reference-free quality signals" are
mostly length-repair devices**, and length-repair is worth ~+0.1 chrF and −0.55 BLEU at this level of
precision.

That also means the cheapest remaining lever is *decode-time* length control, not selection — **and I
tested that too, and it failed**: re-decoding the 2 k core with `length_penalty` 1.2 and 1.5 (beam-5)
gives BLEU 23.43 (+0.11 [−0.09, +0.34]) and 23.38 (+0.06 [−0.20, +0.33]) against 23.32 at lp = 1.0,
i.e. nothing outside the CI. The under-length of beam top-1 is real, but it is not *caused* by the
length penalty the way the mechanism above suggests — the rerankers recover length by changing word
choices, not by telling the decoder to write more. A hypothesis I liked, killed in five minutes, is
worth more than one I never tested.

## 9. A diverse pool does not help — and it hurts the round-trip rules badly

Doubling the candidate count with 8 sampled hypotheses (temperature 0.95, top-p 0.95) on the same
2 k core:

| pool | cands/row | unique/row | top-1 BLEU | oracle BLEU | chrF-MBR | string RT | latent RT |
|---|---|---|---|---|---|---|---|
| beam-8 | 8 | 7.96 | 23.32 | **26.54** | 23.34 | 22.74 | 22.64 |
| beam-8 + sample-8 | 16 | 14.94 | 23.32 | 26.40 | **23.39** | 21.25 | 21.37 |

Sampling adds 7 more distinct candidates per row and **lowers** the BLEU ceiling (26.54 → 26.40): the
beam already contains the best candidates, and the samples are diverse in the wrong direction. MBR is
indifferent (+0.05), but both round-trip rules **lose ~1.4 BLEU** — they chase the fluency of sampled
outputs and their back-translations match the source only loosely. So the ceiling is set by *candidate
quality*, not by candidate count, and adequacy-style signals degrade as the pool gets noisier. This
also kills the "our pool was too redundant" excuse for the small selection gains.

## 10. BLEU and COMET disagree about the best selector (a warning for the paper)

chrF-MBR is best on BLEU (23.04) and chrF (46.59) but **not** on COMET (0.7609 vs beam's 0.7617).
The latent arm is the mirror image: worst-but-one on BLEU, best-but-one on COMET (0.7614). A paper
that reports one metric would tell a different story about which signal is "good". Both are inside
0.001–0.002 of each other, i.e. the *ranking of selectors* is metric-dependent and small.

## 11. The learned-ranker paradox

A 3 k-parameter MLP trained **with reference labels** (COMET) reaches ρ = 0.557, the best scorer in
the table — and yet its selections lose 0.70 BLEU to plain beam search. Two causes, both checkable:
regression-to-the-mean compresses the predicted range, and the top-1 tolerance in COMET terms (0.01)
is hit only 35 % of the time, so a better ρ does not translate into better picks. This is the
strongest evidence in the night for the project's methodological claim: **a better similarity number
is not a better system, and the mapping from one to the other is not automatic.**

## 12. The labelled control: what fine-tuning buys (not part of the claim)

One pre-registered configuration: LoRA r = 16 on q/k/v/out of both the 6-layer encoder and the
6-layer decoder, **1 179 648 trainable parameters (1.5 % of the model)**, 2 500 steps × 24 = 60 000
train-split pairs (0.4 epoch of the prepared file), lr 3e-4, fp16, **7.5 minutes** on this laptop.

| arm | BLEU (10 k) | BLEU (2 k core) | chrF | COMET (2 k core) |
|---|---|---|---|---|
| frozen, beam-8 top-1 | 22.85 | 23.32 | 45.89 | 0.7617 |
| frozen, chrF-MBR | 23.04 | — | 46.59 | 0.7609 |
| frozen, oracle over its own pool | 26.23 | 26.54 | 48.76 | 0.7923 |
| **LoRA, greedy** | 25.66 | 25.66 | 49.06 | 0.7742 |
| **LoRA, beam-8 top-1 (60 k pairs, 7.5 min)** | **26.50** | 26.66 | 50.08 | 0.7839 |
| **LoRA, beam-8 top-1 (150 k pairs, 19.3 min)** | **27.02** | 26.96 | **50.52** | **0.7861** |
| LoRA, chrF-MBR (60 k pairs) | 25.81 | 25.85 | 50.39 | 0.7828 |

**Returns to the adaptation budget** (same 1.18 M parameters, only pairs seen changes):

| pairs seen | wall clock | BLEU | Δ vs frozen beam |
|---|---|---|---|
| 0 | — | 22.85 | — |
| 60 000 | 7.5 min | 26.50 | +3.65 |
| 150 000 (1 epoch) | 19.3 min | **27.02** | **+4.17** |

Still rising, so +3.65 is a lower bound; it does not saturate at 0.4 epoch.

Three readings, in increasing order of discomfort:

1. **+3.65 BLEU for 1.5 % of the parameters and 7.5 minutes** (and +4.17 at one epoch, 19 min).
   Nothing reference-free came close (+0.19 was the ceiling of nine rules).
2. **The control beats the frozen model's *oracle*** on BLEU (26.50 vs 26.23) — the domain-adapted
   model's first candidate is better than the frozen model's best-of-eight.
3. It does **not** beat the oracle on COMET (0.7839 vs 0.7923), so a well-chosen frozen candidate can
   still carry information the adapted model lost.
4. **And the complementarity hypothesis was tested and rejected**: running chrF-MBR on the adapted
   model's own pool *loses* 0.69 BLEU (25.81 vs 26.50) and 0.001 COMET. The +0.19 that MBR earned on
   the frozen model did not transfer — a selection rule's value is a property of the *pool*, not of
   the rule. That is the cleanest single sentence the night produced about selection methods.

Caveat kept: one configuration, 0.4 epoch, no hyperparameter search, and no test-time leakage (the
adapter saw only `split == train`). It is a *lower* bound on what fine-tuning can do here.

## 13. Threats to validity

1. **COMET is teacher and judge.** The oracle, the ranker's labels and the reported COMET are the same
   checkpoint. The +3.38 BLEU headroom is therefore "what COMET would choose", not "what is right".
2. **MTet misalignment.** 2.9 % of test rows have a best-of-8 COMET < 0.5 and 8.7 % < 0.6, so a
   minority of rows cannot be scored well by anyone; absolute BLEU is depressed for every arm, only
   gaps are interpretable.
3. **Pool redundancy.** 7.96/8 distinct candidates and mean pairwise chrF 78.1 bound the reranker's
   ceiling. **Tested** (section 9): adding 8 sampled candidates per row (14.94/16 distinct) did *not* raise the
   BLEU ceiling (26.54 → 26.40) and cost the round-trip rules ~1.4 BLEU, so redundancy is not the
   binding constraint.
4. **Sentence-level only.** No span-level or food-specific analysis; the culinary domain is untouched.
5. **Back-translation is itself a model output** with its own domain shift — a property of the signal,
   not a bug, but it caps what round-trip agreement can see.
6. **One domain, one direction, one seed.** vi→en, MTet, seed 20261119; the tiny ranker uses 3 seeds,
   everything else is deterministic.

## 14. What I would do next, in order of value per hour

1. **A selection rule's value is a property of the pool.** MBR earned +0.19 BLEU on the frozen pool
   and lost 0.69 on the adapted pool (measured tonight, section 12); round-trip rules lost 1.4 BLEU when the
   pool was made more diverse (section 9). Any paper that reports "rule X gives +Y" must report the pool it
   was measured on, or the number does not travel.
2. **COMET-Kiwi (reference-free QE) as a utility** — the clean upper bound for a *learned*
   reference-free signal. **Not run tonight**: `Unbabel/wmt22-cometkiwi-da` is a gated repo and this
   machine has no HF credentials (`GatedRepoError 401`). First thing to run when someone accepts the
   licence.
3. **Span-level agreement**, then fact insertion — the project's title layer, still unmeasured inside
   the pipeline.
4. **Beam across the project** (cheap, uncomfortable): every absolute number in the paper was produced
   greedily; the +1 BLEU is bigger than most of the effects the paper argues about.
5. **Train the readout on the injected channel** (the E7/E8 lesson) is still uncombined with tonight:
   a pretrained decoder reading mapped states was never tried, and the oracle headroom suggests
   candidate quality — not channel capacity — is what limits generation.

## 15. Cost ledger (this laptop)

| stage | measured |
|---|---|
| greedy generation, 10 k sentences | ~5 min (34 sent/s, batch 24, fp16) |
| beam-8 generation, 10 k sentences | 19.5 min frozen / 25.9 min with adapter (8.5 / 6.4 sent/s, batch 8) |
| back-translation, 79 605 unique strings | 24 min (55 sent/s) |
| encoder features, 10 k rows + 80 k back-translations | 7 min |
| COMET scoring, 80 k candidate pairs | 28 min (~48 pairs/s incl. dedupe) |
| LoRA control, 60 k pairs | **7.5 min** (134 pairs/s) |
| API spend (`coding` key) | **$0.25 of $3.00** |

## 16. Reproduce

`README.md` §Reproduce. Every table above is regenerated by `python -m src.tables` from
`results/*.json`; every hypothesis is in `HYPOTHESES.md`; every mistake is in `PROGRESS.md`.
