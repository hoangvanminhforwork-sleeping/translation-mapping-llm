# Mission 5 — One-class anomaly detection on the mapped representation

**Result in one line.** The distribution-shift hypothesis is *half* right, and the
half that is wrong is more useful than the half that is right. Mapped vectors are
**not** outliers — they are **over-dispersed and directionless**: the map gets the
*scale* roughly right while producing a displacement from the target mean that is
**essentially orthogonal** to the true one (cosine **0.020**).

Reproduce: `..\research-scale-100k\.venv\Scripts\python.exe -m src.anomaly`
Evidence: `results/anomaly_detection.json`, `results/anomaly_shrinkage.json`,
cached embeddings in `data/anomaly_vectors.npz`.

---

## 1. Protocol (as specified in missions.md §5)

- 6,000 held-out sentences (`chosen.parquet`, `split == "val"`), vi→en,
  embedded with the frozen pair the map was fitted on (`t5-small-vietnamese` →
  `t5-small`); **145,298 aligned token pairs**, 15 s on the RTX 3050.
- Detectors **fit on 90 % of genuine target-side states only** (one-class).
- Threshold set so exactly **10 % of the fitting set** is flagged.
- Test A: the held-out 10 % of genuine states → must come back at ~10 % (a
  detector that cannot pass this is broken).
- Test B: **all** mapped vectors → the number of interest.

## 2. Three detectors disagree, and the disagreement is the finding

| Detector | held-out normal | **mapped** | identity (no map) | target mean |
|---|---|---|---|---|
| Isolation Forest | 0.099 | **0.000** | 0.846 | 0.000 |
| Gaussian Mixture (8 comp, diag) | 0.103 | **0.366** | 1.000 | 0.000 |
| Autoencoder (512→256→64→256→512) | 0.102 | **0.737** | 1.000 | 0.000 |

Calibration holds in every case (≈0.10 on held-out genuine states), so no
detector is simply broken. Isolation Forest says the mapped vectors are
*perfectly normal*. The autoencoder says three quarters of them are anomalies.

**Why.** Isolation Forest scores by path length to isolation — it measures
*how far from the mass* a point sits. GMM and the autoencoder score *density and
reconstruction*, which measure *whether a point lies on the manifold*. A set can
be perfectly ordinary in the first sense and thoroughly off-manifold in the
second, and that is exactly what the mapped vectors are.

## 3. The direct measurement explains it

Distances are in the z-scored target frame (genuine-state mean and std).
`in-inner-90 %` = fraction of points falling inside the radius that contains 90 %
of genuine states.

| Representation | mean ‖dist‖ | median | p05 | p95 | per-dim var ratio | in-inner-90 % |
|---|---|---|---|---|---|---|
| target **Y** (genuine) | 22.37 | 22.78 | 16.65 | 27.46 | 1.000 | **90.0 %** |
| **mapped M** | 25.85 | 25.67 | **7.71** | **45.10** | 1.276 | **53.6 %** |
| source **X** (identity) | 38.72 | 37.89 | 33.43 | 45.76 | 2.555 | 0.0 % |
| target mean | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 | 100 % |

Two numbers carry the argument:

- **`p05 = 7.71` and `p95 = 45.10`.** Genuine states occupy a *narrow shell*
  (16.7 → 27.5). Mapped vectors are spread from deep inside the cloud to far
  outside it. The map does not land vectors in the target region; it produces a
  **bimodal, over-dispersed** population, half inside and half beyond.
- **variance along the target's principal axes = 0.296× target**, while
  **per-dimension variance = 1.276× target**. Variance is being pushed into
  directions the target does not use.

And the decisive one:

> **cosine(M − μ, Y − μ) = 0.020**

where the mapped vector lands *relative to the target mean* tells you **nothing**
about where the true target state lies relative to that mean. The map reproduces
the scale of the target cloud and almost none of its direction.

## 4. What this means for the project

1. **The "distribution shift" hypothesis is refined, not confirmed.**
   `missions.md` §5 proposed that mapped vectors are not in the decoder's
   familiar distribution. They are not *outliers* (Isolation Forest: 0.000) —
   they are *directionless*. The decoder's problem is not that the input looks
   strange, it is that the input carries almost no target-relevant signal.
2. **This explains the copy-readout result in ver02.** The copy gate passed at
   85.9 % from the model's *own* encoder, yet the mapped readout produced 0.37
   BLEU / 95 % degenerate output. A vector whose displacement is orthogonal to the
   true one is statistically indistinguishable from a generic "plausible" vector
   — which is precisely what makes a decoder fall back on fluent, content-free
   text rather than producing garbage. **Degeneration is the expected outcome of
   an orthogonal map, not an unexplained failure.**
3. **It kills the "maximise cosine similarity" objective for the right reason.**
   If token-level cosine can be raised while the displacement direction stays
   orthogonal, then cosine is measuring the wrong thing. Any new objective should
   be evaluated on **direction** (e.g. cosine of the residual after removing the
   mean, or retrieval@1 on held-out pairs), not on pooled cosine.
4. **Mission 4's "reduce the ambiguity" now has a concrete target.** The
   ambiguity to remove is the map's anisotropy: 0.296× variance on target axes
   while overall variance rises to 1.276×. A whitening / Procrustes-style
   constraint (rather than plain ridge least squares) targets exactly this
   defect, and it is cheap to test with the cached vectors in
   `data/anomaly_vectors.npz` — no GPU pass needed.

## 5. Caveats, stated plainly

- **Isolation Forest at 0.000 is not evidence of "no shift".** It is evidence
  that IF's notion of anomaly (distance from the mass) is the wrong instrument
  here. I keep the number precisely because it would be easy to report only the
  autoencoder's 0.737 and claim a clean confirmation.
- **The autoencoder ran on CPU, 40 epochs, fixed architecture, one seed.** The
  0.737 could move by a few points with a different capacity; the ordering
  (mapped ≫ normal, identity ≫ mapped) is what is being claimed, not the third
  decimal.
- **`cosine = 0.020` is measured on character-span-aligned token pairs**, the
  same noisy alignment ver02 used (~14 % of source tokens fall back to the
  nearest target token). That noise can only *deflate* the cosine, so the true
  token-level alignment may be somewhat higher than 0.020 — but the same
  alignment produced the positive retrieval numbers in ver02, so it is not
  uniformly destructive.
- **One direction only (vi→en).** en→vi has no equivalent map quality and was
  not run.
