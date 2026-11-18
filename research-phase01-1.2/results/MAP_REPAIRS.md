# Mission 4 — Repairing the map: what the diagnosis bought us

Reproduce: `..\research-scale-100k\.venv\Scripts\python.exe -m src.fix_maps`
Evidence: `results/map_repairs.json`

Protocol: every transform is **fitted on 30,000 train tokens** and scored on
**30,000 held-out val tokens**. Nothing is fitted on what it is measured against.
Direction vi→en. `ret@1` uses a 4,000-vector bank, so **chance = 0.00025**.

## Result

| map | cos_abs | cos_centre | const_cos | zdist_med | zdist_p05 | zdist_p95 | ret@1 |
|---|---|---|---|---|---|---|---|
| constant (μ_Y) | 0.3541 | 0.0001 | 0.3541 | 0.00 | 0.00 | 0.00 | 0.0003 |
| identity (no map) | 0.0903 | 0.0412 | 0.1680 | 37.89 | 33.46 | 45.90 | 0.0063 |
| **ridge [ver02]** | **0.4433** | **0.2703** | 0.3542 | **6.31** | 0.99 | 9.85 | 0.0770 |
| ridge + recolor | 0.1423 | 0.0706 | 0.3529 | 61.59 | 9.59 | 95.68 | 0.0475 |
| **procrustes** | 0.2279 | 0.1465 | 0.3540 | 34.88 | 16.07 | 41.77 | **0.0938** |
| procrustes + recolor | 0.2268 | 0.1290 | 0.3540 | 31.96 | 11.76 | 39.90 | 0.0845 |

Genuine target vectors: `zdist_med = 22.77`, p05/p95 = 16.65 / 27.46.

## Three findings

### 1. The reported metric is broken, and this table proves it

`cos_abs` — the number ver02 reports as "token cosine" — gives a **constant
vector 0.3541** and the orthogonal Procrustes map **0.2279**. The constant scores
**55 % higher than a map with 22 % better retrieval**. Any alignment claim based
on uncentred cosine can be satisfied by predicting the mean, which is precisely
what ridge does (it beats the constant by only 0.089 while collapsing to
‖z‖ = 6.3).

**Consequence for the paper:** report `cos_centre` *and* `const_cos` next to
every cosine. A map that does not beat `const_cos` has not demonstrated transfer.

### 2. Naive covariance matching fails, and the failure is instructive

`recolor` (`Σ_Y^½ Σ_M^{-½}`) was meant to undo the collapse. It overcorrected
catastrophically: `zdist_med` went from **6.31 to 61.59**, far past the genuine
22.77, and retrieval *dropped* from 0.0770 to 0.0475.

Cause: ridge's output covariance is near-singular in the collapsed directions,
and `Σ_M^{-1/2}` inverts those near-zero eigenvalues, amplifying noise. Whitening
an ill-conditioned matrix inflates rather than repairs.

So *"rescale the output to match the target covariance"* is not a fix. The
direction structure has to be right before a rescaling can help.

### 3. Procrustes wins retrieval while losing on every cosine

The orthogonal map has **+22 % relative retrieval@1** over ridge (0.0938 vs
0.0770, both far above the 0.00025 chance floor) and a far healthier spread —
but a *lower* `cos_abs` and `cos_centre`.

This is consistent with Zhou et al. 2026, who found Procrustes winning retrieval@1
for **every one of 36 model pairs** while affine won MSE and CKA — two metrics
ranking the same two maps in opposite orders.

Ridge's advantage on centred cosine with a collapsed distribution is the classic
signature of a fit that captures the dominant shared component but not the
token-level variation. Procrustes, being length-preserving, cannot collapse, and
that shows up as better retrieval.

## What I would do next, in order

### Addendum — the partial sweep has been run, and it revises point 1

`IN_PROGRESS` below was replaced by a real sweep (`src/recolor_sweep.py`,
`results/recolor_sweep.json`, 66 s). `M_t = (1−t)·M + t·recolor(M)`:

| ridge t | cos_centre | zdist_med | ret@1 |
|---|---|---|---|
| 0.0 | 0.2703 | 6.31 | 0.0770 |
| **0.1** | 0.2132 | 9.35 | **0.0915** |
| 0.2 | 0.1581 | 14.41 | 0.0870 |
| 0.3 | 0.1273 | 20.01 | 0.0810 |
| 0.5 | 0.0967 | 31.70 | 0.0685 |
| 1.0 | 0.0706 | 61.59 | 0.0475 |

| procrustes t | cos_centre | zdist_med | ret@1 |
|---|---|---|---|
| 0.0 | 0.1465 | 34.88 | 0.0938 |
| **0.3** | 0.1520 | 31.10 | **0.0955** |
| 0.6 | 0.1492 | 29.72 | 0.0907 |
| 1.0 | 0.1290 | 31.96 | 0.0845 |

Two things follow, and the first is a correction to what I recommended above:

1. **Matching the marginal distance distribution is *not* the fix.** The sweep
   that brings ridge's `zdist_med` closest to the genuine 22.77 (t ≈ 0.3) has
   *worse* retrieval than the uncorrected map. Recovering the marginal spread by
   blending in a whitened version adds variance but not signal. The defect is in
   the joint structure, not the marginal variance — the collapse was a symptom,
   not the cause.
2. **A small recolor does help, just not for the reason assumed.** Ridge at
   `t = 0.1` gains **+19 % relative retrieval** (0.0770 → 0.0915) while keeping
   `cos_centre` at 0.21. And `procrustes + t = 0.3` is the best configuration in
   the whole study at **ret@1 = 0.0955** — 382× the 0.00025 chance floor.

**Best two candidates to take to a generation test:**
`procrustes + recolor(t=0.3)` (best retrieval) and `ridge + recolor(t=0.1)`
(best cosine, near-best retrieval). They disagree on which is "better" by the
same metric conflict already documented above — so the generation test must
decide, not the diagnostic.

Original plan follows, now superseded only on point 1:

1. ~~Partial recolor, not full recolor.~~ **Done, see addendum.**
2. **Then** test generation with the best of `{ridge, procrustes, partial recolor}`
   through the frozen decoder. Do not spend GPU hours before the distance and
   retrieval diagnostics look right — ver02's lesson is that a scalar metric
   improving does not imply better text.
3. **If that fails**, attack the supervision: the ~14 % of source tokens
   whose alignment falls back to the nearest target token are wrong *training
   targets*, and wrong targets force exactly this kind of shrinkage. The corpus
   has a ready replacement — *Cross-Lingual Representation Alignment by Token-Level
   Optimal Transport in a Language-Agnostic Space* (2026).

## Honest summary

The de-shrinking idea was reported as a **diagnostic**, per the agreed framing,
and the diagnostic has now done its job twice: it identified collapse as the
defect, and it showed that the obvious repair is not sufficient. Linear algebra
alone did not fix the map. What it did produce is a measurement protocol that
cannot be gamed by a constant, and a clear ordering of what to try next.
