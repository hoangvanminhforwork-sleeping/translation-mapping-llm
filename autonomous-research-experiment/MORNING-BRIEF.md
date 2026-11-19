# MORNING BRIEF — read this first (2 minutes)

You asked for insight toward the highest BLEU, with novelty as a constraint, fully autonomous, 8 h,
$3, two models, no fine-tuning-as-a-claim. Here is what the night produced.

## The question I chose

The project's map architecture plateaus at 3–6 BLEU; a frozen 77 M model already gets 22.27. So the
only honest space left was: **keep both models frozen and make the choice of output better** — and the
open question there is the one your neural-mining intro circles: *is representation-space (latent)
agreement a usable reference-free quality signal?*

## Six numbers

| what | number |
|---|---|
| calibration: greedy on the 2 k core | **22.29 BLEU / 0.7522 COMET** vs published 22.27 / 0.752 |
| beam-8 instead of greedy (every earlier phase used greedy) | **+1.01 BLEU** [+0.84, +1.17] |
| oracle over the frozen model's own 8 candidates | **26.23 BLEU** (+3.38) |
| best of nine reference-free selection rules (chrF-MBR) | 23.04 (**6 % of the headroom**) |
| **latent agreement vs string agreement** | ρ **+0.478** vs +0.394, Δ **+0.083** [+0.075, +0.092] — better *scorer* |
| …as a selector | 22.31 vs 22.29 BLEU — **no better**, both 0.55 below plain beam |
| control: 1.2 M-param LoRA, 7.5 min | **26.50 BLEU** (+3.65), COMET 0.7839 |

**The one-sentence answer:** latent agreement is a significantly better *predictor* of quality than
string agreement and an equally poor *selector* — and the frozen model already contains a 26.2-BLEU
answer among its eight candidates, which no reference-free rule comes close to extracting. Seven and a
half minutes of LoRA beat the entire selection exercise by 19×.

## Your top-ranked insight: "is latent agreement usable?"

- **Yes as a ranking/estimation signal** (Δρ +0.083, robust across source-length quartiles, better than
  the string version and better than a 2.4 k-param MLP trained on string features).
- **No as a decision rule** (top-1 accuracy 0.370 vs 0.356 → +0.02 BLEU, i.e. nothing).
- **Why** (`results/EXAMPLES.md`): it inherits the back-translator's copy behaviour — it rewards a
  candidate whose Vietnamese back-translation reproduces the source's *format* (numbers, `%s`, dates)
  while getting the word wrong. It measures shape, not meaning.
- **Replication on a second pool** (the adapted model's, 2 k sentences / 16 k candidates): string
  ρ +0.286, latent ρ +0.362, **Δ +0.076** — nearly the same gap, so this is a property of the *signal*,
  not of one pool (`results/replication.json`). Selection replicates too: latent 25.89, MBR 25.85,
  string 25.71, all below the adapted model's own top-1 (26.66).

## What failed (reported, not hidden)

1. **Length-penalty hypothesis**: lp 1.2/1.5 → +0.11/+0.06 BLEU, CI straddles 0. Under-length was real
   but not caused by the penalty.
2. **"Adaptation + selection compose"**: chrF-MBR on the adapted pool **loses** 0.69 BLEU. A selection
   rule's value is a property of the pool.
3. **"A more diverse pool will help"**: beam-8 + 8 samples (14.94 distinct/row) **lowered** the ceiling
   26.54 → 26.40 and cost the round-trip rules ~1.4 BLEU.
4. **COMET-Kiwi as a learned QE baseline**: not run — `Unbabel/wmt22-cometkiwi-da` is a gated repo
   (`GatedRepoError 401`, no HF credentials here). Accept the licence and `python -m src.kiwi` runs.

## What you should check before believing me

- `HYPOTHESES.md` — written **before** any test number, with the deviation log (the `lp` feature was
  dropped; stated there).
- `results/metrics_*.json` + `results/sel_*.jsonl` — every table is recomputable from these.
- `PROGRESS.md` — including three self-inflicted process-tree stalls (the symptom is halved throughput,
  not an error) and one 24-minute loss from not checkpointing.
- `NOTEBOOK_executed.ipynb` — runs end-to-end, 5 sections, no hand-typed numbers.

## Honest framing for the paper

The defensible sentence is narrow: *for two frozen 77 M models on a 10 k vi→en test set,
representation-space round-trip agreement is a significantly better predictor of COMET than
string-space round-trip agreement, while being an equally poor selector; the frozen model's own 8-best
pool contains 3.38 BLEU that no reference-free rule (and no 3 k-param learned ranker) extracts.*

The uncomfortable but useful companion sentence: **a 7.5-minute, 1.2 M-parameter LoRA beats every
selection rule by an order of magnitude**, so any paper whose contribution is a selector has to be
compared against "just adapt the model".

## Cost

**$0.25 of $3.00** (reasoning only; all compute was local GPU). GPU: 4 GB RTX 3050, peak 66 °C, one job
at a time. Total wall clock ≈ 5 h 15 min of machine work, hard stop respected.
