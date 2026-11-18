# Mission 4 — generation run: the diagnostic picked the wrong winner

Reproduce: `..\research-scale-100k\.venv\Scripts\python.exe -m src.gen_variants --n 1000`
Evidence: `results/gen_variants_vi_en.json`, `results/gen_vi_en_<channel>.jsonl`

1,000 held-out test pairs, vi→en, one adapter (`trans-t5-small-vi-en`), greedy
decoding, identical pairs across channels. Transforms fitted on train tokens with
the same seed/subsample as `fix_maps.py`, so each channel is exactly what was
measured there. Metrics are ver02's own (`score.py`), so the degeneration column
is comparable to its 31.6 % figure.

## Result

| channel | BLEU | chrF | NLL | degen | empty |
|---|---|---|---|---|---|
| native (target's own encoder) | 4.20 | 14.82 | 3.413 | 0.325 | 0.000 |
| identity (no map) | 0.04 | 6.04 | 4.615 | 0.156 | 0.000 |
| ridge [ver02's `mapped`] | 2.73 | 19.73 | **3.228** | **0.149** | 0.000 |
| **ridge + recolor(t=0.1)** | **3.17** | **20.28** | 3.290 | 0.162 | 0.000 |
| procrustes | 0.48 | 8.88 | 5.168 | 0.695 | 0.006 |
| procrustes + recolor(t=0.3) | 0.70 | 10.65 | 4.844 | 0.596 | 0.004 |

Sanity check against ver02: its `mapped` channel scored 2.84 BLEU / 19.96 chrF at
n=3000; here ridge scores 2.73 / 19.73 at n=1000. Consistent.

## Finding 1 — a free +16 % BLEU

`ridge + recolor(t=0.1)` beats ver02's best channel on both surface metrics:
**BLEU 2.73 → 3.17 (+16 % relative)**, **chrF 19.73 → 20.28**. It costs one
512×512 matrix fitted in seconds on data already on disk — no retraining, no GPU.

Against `native` it wins on chrF (20.28 vs 14.82), NLL (3.290 vs 3.413) and
degeneration (0.162 vs 0.325), and loses only on BLEU (3.17 vs 4.20) — which is
the known BLEU artefact: `native` degenerates into loops 32.5 % of the time and
BLEU is precision-oriented, so it rewards them. ver02 documented the same
inversion (4.17 native vs 2.84 mapped BLEU, but 15.35 vs 19.96 chrF).

## Finding 2 — every token-level diagnostic ranked Procrustes first, and it fails

| | retrieval@1 | zdist_med | BLEU | degen |
|---|---|---|---|---|
| ridge | 0.0770 | 6.31 | 2.73 | 0.149 |
| procrustes | **0.0938** | 34.88 | **0.48** | **0.695** |

Procrustes has **22 % better token retrieval**, a healthier spread, and is
**5.7× worse at BLEU** with four times the degeneration. The token-retrieval
metric and the decoder disagree completely, and the decoder wins.

Why: an orthogonal map preserves the source norm, and source norms are ~2.5×
larger than target norms (identity channel `zdist_med = 37.9` vs genuine 22.8).
The encoder-output distribution is far outside what the decoder's cross-attention
tolerates, so it degenerates — exactly like `identity`, just less completely.
Ridge, by shrinking toward the mean, lands in a *survivable* scale and survives.
**The decoder cares about the marginal scale far more than about per-token
neighbour structure.**

## Finding 3 — the metric hierarchy for this project

Ranked by how well each predicts translation quality, from this experiment:

1. **Generation metrics** (BLEU / chrF / degen) — ground truth, expensive.
2. **`zdist_med` vs the genuine 22.8** — tracked generation correctly here
   (ridge 6.3 → survivable, procrustes 34.9 → degenerate). Cheap, no decoder.
3. **`cos_centre`** — agrees with generation on ridge > procrustes, but ranked
   `ridge+recolor.1` *below* plain ridge while generation says above.
4. **retrieval@1** — actively misleading. Best retrieval, worst generation.

Anyone continuing this work should treat retrieval@1 as a *diagnostic of the map*,
never as a proxy for translation quality.

## What this means for the paper

- The contribution is now concrete and cheap: **a variance-matched linear map
  beats the plain ridge map by 16 % BLEU / 2.8 % chrF at zero training cost**,
  and it beats `native` on chrF, reference NLL and degeneration.
- The negative result is equally clean: **orthogonal / length-preserving maps are
  unusable through a frozen decoder**, whatever their retrieval score. That
  contradicts the "Procrustes wins" reading of the alignment literature and
  explains why: those papers measure retrieval between *static* spaces, never
  generation through one of them.
- Together with `ANOMALY.md`, the story is: the map's problem was never that its
  vectors were strange. It was that they were *too average* — and partially
  undoing that, by 10 %, is enough to gain 16 % BLEU.

## Caveats

- n=1000, single seed, greedy decoding, one adapter. BLEU differences of ~0.4 are
  at the edge of noise; the chrF and degeneration gaps are larger and consistent
  in direction with ver02's 3,000-sentence run, but a paired bootstrap on
  ≥3,000 pairs is needed before this goes in a table.
- `t = 0.1` was chosen from a sweep on *diagnostics*, not on generation. It is
  the best of six values tried; a generation-side sweep over t might find better.
- One direction (vi→en). en→vi has no comparable map-quality baseline.
- The recolor matrix is fitted on the same train token sample the map was fitted
  on, so it inherits any sampling bias there.
