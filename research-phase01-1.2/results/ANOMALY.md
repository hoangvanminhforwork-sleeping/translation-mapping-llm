# Mission 5 — One-class anomaly detection on the mapped representation

**Result in one line.** All three one-class detectors flag **0.000** of the mapped
vectors while flagging **0.85–1.00** of unmapped source vectors. The mapped
representation is not out of distribution — it is *too* in distribution: it sits
in the densest core of the target cloud. Mission 5's hypothesis, taken
literally, is **falsified**.

Reproduce: `..\research-scale-100k\.venv\Scripts\python.exe -m src.anomaly`
Evidence: `results/anomaly_detection.json`, `results/map_repairs.json` (distance
statistics), embeddings cached in `data/anomaly_vectors.npz`.

> **Correction notice.** The first run of this experiment applied the map as
> `X @ W.T + b`. ver02 applies `X @ W + b` (`eval_geometry.py:75`,
> `fit_lgbm.py:124`, `mine.py:123`). The transposed run produced the opposite
> conclusion ("over-dispersed and directionless"). The tell was that a *constant*
> vector beat the map on the reported cosine — an impossible result for a
> correctly-oriented least-squares map. Everything below uses `X @ W + b`,
> which reproduces ver02's token cosine to within 0.016 (0.4433 here vs 0.4595
> in `results/lgbm_metrics.json`).

---

## 1. Protocol (missions.md §5, as specified)

- 6,000 held-out sentences (`chosen.parquet`, `split == "val"`), vi→en, embedded
  with the frozen pair the map was fitted on (`t5-small-vietnamese` → `t5-small`):
  **145,298 aligned token pairs**, 15 s on the RTX 3050.
- Detectors **fit on 90 % of genuine target-side states only** (one-class).
- Threshold set so exactly **10 % of the fitting set** is flagged.
- Test A: the held-out 10 % of genuine states → must return ~0.10, else the
  detector is broken.
- Test B: **all** mapped vectors.

## 2. Result

| Detector | held-out normal | **mapped** | identity (no map) |
|---|---|---|---|
| Isolation Forest | 0.099 | **0.000** | 0.846 |
| Gaussian Mixture (8 comp, diag) | 0.103 | **0.000** | 1.000 |
| Autoencoder (512→256→64→256→512) | 0.102 | **0.000** | 1.000 |

Calibration holds everywhere (≈0.10 on held-out genuine states), so this is not a
broken detector. Three detectors built on three different notions of "anomalous" —
path length, density, reconstruction error — agree that the mapped vectors are
**more normal than normal vectors are**.

## 3. Why: the mapped vectors are collapsed toward the mean

Distances in the z-scored target frame, held-out tokens:

| Representation | median ‖z‖ | p05 | p95 |
|---|---|---|---|
| genuine target **Y** | 22.77 | 16.65 | 27.46 |
| **mapped M** (ridge) | **6.31** | **0.99** | **9.85** |
| identity (raw source X) | 37.89 | 33.46 | 45.90 |
| constant (μ_Y) | 0.00 | 0.00 | 0.00 |

The mapped cloud is **3.6× tighter than the genuine cloud**, and its lower tail
reaches essentially the mean (`p05 = 0.99`). Genuine states live in a shell
between 16.7 and 27.5 from the centre; mapped states live between 1 and 10.

That is textbook ridge shrinkage: the least-squares solution pulls predictions
toward the mean, hardest in the directions where the source carries little
signal. The mapped vector is therefore an excellent "average plausible
activation" — which is exactly why density- and reconstruction-based detectors
score it as maximally normal.

## 4. What this means

1. **Mission 5's hypothesis is falsified in its literal form.** The decoder does
   not receive an unfamiliar vector. It receives a vector from the *most* familiar
   part of its space.
2. **It explains the 31.6 % token loops in ver02.** If every injected vector is
   within ‖z‖ ≈ 6 of the same point, the decoder gets almost no
   sentence-specific signal. A decoder with no conditioning signal produces
   fluent, repetitive, generic text — `the sex of the sex of …` is the expected
   output of an input that barely varies. Degeneration is not a mysterious
   failure mode; it is the signature of a collapsed conditioning vector.
3. **It also explains why `mapped` beats `identity` but loses to `native`.**
   Identity vectors are far outside the manifold (0.85–1.00 flagged) and produce
   nothing at all. Collapsed vectors are inside it, so the decoder at least
   emits language. A real encoder (`native`) emits language *and* carries
   content.
4. **The collapsed distribution is why anomaly detection was the wrong
   instrument** for this question, and that is worth reporting: the test the
   mission proposed cannot detect the defect the project actually has.

## 5. Caveats

- The autoencoder is CPU, 40 epochs, one architecture, one seed. The claim is the
  ordering (mapped ≪ normal ≪ identity), not the third decimal.
- `0.000` is a floor, not a precise estimate: at most a handful of mapped tokens
  exceed the 90th-percentile threshold.
- Token alignment is character-span based (~14 % of source tokens fall back to
  the nearest target token, per the ver02 handoff). That noise deflates every
  directional metric here, so the true directional signal is *at least* what is
  reported.
- vi→en only.
