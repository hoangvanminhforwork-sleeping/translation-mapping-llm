# Learning note — how the coincidence experiment was built, and what it cost

For the human. `REPORT.md` has the findings; this file has the reasoning, the trade-offs and the
mistakes, so the next person does not pay for them again.

---

## 1. The decision that made the night possible: cache the encoder states

The map is fitted on 20 000 pairs for up to 6 epochs. Re-encoding each epoch would put a frozen
60 M encoder in the inner loop; instead `src/cache_states.py` writes `E_vi`, `E_nat`, `E_ref` and the
decoder labels once (**150 s, 6.5 GB**), and every later step is a *decoder-only* job.

- Cost: 6.5 GB on disk, 150 s once.
- Benefit: one epoch = 20 000 pairs through a 60 M decoder ≈ **110 s**, so 6 epochs fit in 11 min and
  the whole night's GPU time is 75 min.
- Trade-off I accepted: the cache is a schema. Changing `MAX_LEN` or the tokenisation invalidates it,
  so the cache is keyed by set name only. If you change tokenisation, delete `agent-work/cache/`.
- Engineering detail that matters on 2.25 GB of free RAM: the arrays are `np.lib.format.open_memmap`
  files, read by fancy index per batch. Nothing is ever fully materialised — a 20 000×64×512 float32
  tensor would be 2.6 GB.

I also cached `E_ref` and `E_nat` for the *fit* split (2.6 GB) as insurance for a distillation arm
that was never run. Honest accounting: that was 1 minute of GPU and 2.6 GB for an arm that stayed
specified-only. Cheap insurance, but it is unused.

## 2. Two oracles, and why the geometric analysis only uses one

| oracle | what it is | strength | defect |
|---|---|---|---|
| `ref` | target encoder reading the **English reference** | 74 BLEU / CE 0.30 — a real behavioural ceiling | different tokenisation and length from the map's memory |
| `nat` | target encoder reading the **Vietnamese source** | in-distribution, and *position-aligned* with the map | weak content (4 BLEU), already degenerating |

The mission's mental model needs "a memory the decoder reads well" → that is `ref`. But any
position-wise distance between `ref` and the map's memory is meaningless (vi token 3 and en token 3
are unrelated slots). So:

- **perturbation curves** (which need no correspondence) → run on both oracles;
- **subspace projection of the map's error** (which needs corresponding slots) → run on `nat` only.

This is the single most important methodological decision in the package, and it is why
`analyse_oracle(..., aligned=True)` exists. A reviewer would have caught a position-wise
"distance to the oracle"; the fix is structural, not cosmetic.

## 3. fp32 for analysis, and how I proved it did not break comparability

Gradients through bf16 are noisy, so the decoder runs in fp32. That changes greedy decoding slightly:
ridge came out 2.50 BLEU where E9 published 2.73, while CE matched to the third decimal (3.228).

Two checks resolved it:
1. CE is precision-insensitive and matched exactly → the channels are right.
2. Re-scoring at E9's `max_new=48` gives ridge **2.71** vs published **2.73** → the gap was the
   decoding budget, not the precision.

**Rule extracted: `max_new` is worth 0.2–0.35 BLEU in this pipeline.** Every table now reports both
settings, and cross-phase comparisons must state it.

## 4. The mistake I nearly published

The arm-name convention is integer percent: `ridge_recolor10` is t = 0.10, `ridge_recolor01` is
t = 0.01. I generated `ridge_recolor01` intending E9's t = 0.1, measured +0.03 BLEU, and wrote
"E9's +16 % does not replicate" into a draft.

What caught it: the arm's `zdist` was 6.3, and the transform at t = 0.1 must give ≈ 8.5 (computed
independently in `src/diag_recolor.py`). The *effect size of the transform itself* is a checksum on
the parameter you think you set.

**Rule extracted:** when a transform has a scalar knob, print/verify the knob's effect (here:
z-distance) before interpreting any downstream metric. This is the same class of bug as the
project's `X @ W` vs `X @ W.T` incident (§9 of `CURRENT-PROGRESS.md`) — a naming/convention error
that flips a scientific conclusion. It cost ~40 minutes; the alternative was a wrong claim in the
report.

## 5. The control that changed the interpretation

Before running the CE arms I did not believe they would work, for a specific reason: E9's collapse
diagnosis says the mapped cloud is *shrunk*, and E9's recolor shows that partially undoing the
shrinkage is worth +16 %. So the CE objective might just be rediscovering scale.

Hence `ridge_scale122` — the *same* map multiplied by 1.22, which is exactly the norm ratio to
`ce_ridge` (9.3 / 7.6). It buys +0.42 BLEU. So:

- scale ≈ 15 % of the 2-epoch gain,
- covariance matching ≈ 21 %,
- **the rest is directional, and only a behavioural objective finds it.**

Without that control the headline would have been "CE fitting fixes the collapse", which is wrong
and would not survive review.

## 6. The negative result that matters most: `ce_scratch`

`ce_scratch` (CE from `W = I`) reaches only 2.25 BLEU and a *worse* val CE (3.364) than ridge
(3.307), with ‖H‖ 24.4 — it overshoots the genuine scale (22.9). So the CE landscape has bad basins,
and the least-squares solution is an excellent warm start.

Interpretation I am confident in: **MSE is not a model of the task; it is a good initialiser for a
model of the task.** That reframes the project's six rounds of "ridge beats MLP": ridge was never
winning because the relation is linear, it was winning because it lands in the right basin — and the
*objective* was never tested until now.

## 7. Engineering traps hit tonight

| trap | cost | rule |
|---|---|---|
| shell harness killed long foreground runs (once inside an SVD) | ~40 min + one lost run | long jobs go to background; heavy analysis steps get their own script |
| server restart mid-training (epoch 2 of 6) | ~50 min | persist **per arm**, never only at the end — the npz-per-arm design is why nothing else was lost |
| `o.labels` does not exist on `Seq2SeqLMOutput` | 10 min | keep the `labels` tensor you passed in |
| labels on CPU while the model is on CUDA | 5 min | `cc.labels()` returns `.cuda()` |
| SVD of a 1000×32768 gradient matrix inside the training script | killed the run after the arms were written | `randomized_svd(n_components=256)` in a separate `analyse_basis.py` |
| np.savez of a json string + `str()` round-trip | 5 min | fine, but `json.loads(str(z["hist"]))` — check it, don't assume |
| percent-encoded arm names | **the §4 mistake** | never encode a float as an integer without a printed checksum |
| COMET: `download_model("Unbabel/w08-2020-comet")` | 5 min | comet 2.2.7 raises `KeyError: Model ... not supported`; the cached checkpoint is `Unbabel/wmt22-comet-da` |
| COMET: `download_model(..., saving_directory=...)` | 3 min + 1.4 GB | it **ignores** the populated HF cache and re-downloads; call it without `saving_directory` |
| `model(encoder_outputs=..., attention_mask=...)` with no `labels`/`decoder_input_ids` | 5 min | T5 raises "You have to specify either…"; pass `labels=lab` to get teacher-forced logits |

## 8. What I would do differently

1. **Run `distill` before `ce_ridge_e6`.** KL to the oracle's distribution uses soft targets and is
   the theoretically better objective; if it dominates, the 6-epoch run was the wrong allocation.
2. **Fix `max_new=48`** for every arm from the start, to match E9 and E7/E8. Two tables were needed
   only because of this.
3. **Add COMET on the first arm**, not at the end. BLEU 2.5 → 7.0 is far outside BLEU noise, but a
   reviewer will ask whether longer/fluent-but-unfaithful output is being rewarded; COMET answers it
   in 2.5 min per arm.
4. **Assert the fold against the literal transform in the same script** that uses it. `diag_recolor`
   was written after the fact; it should be an assert at the top of `fit_behaviour`.

## 9. How to read the code

```
src/cc.py            channels + metrics + caching primitives (no project imports; `src` name clash)
src/cache_states.py  Phase 0: encode once into agent-work/cache/{test,val,fit}
src/validate.py      calibration vs E9 — run this first, always
src/coincidence.py   Q1/Q2/Q3: perturbation curves, anisotropy, subspace enrichment
src/fit_behaviour.py Phase 2/3: ridge_sub / recolor / scale / ce_* / distill arms, eval + bootstrap
src/analyse_basis.py behavioural overlap with the oracle + gradient spectrum (CPU-heavy SVD)
src/eval_maxnew48.py the comparable-to-E9 table
src/bootstrap_all.py paired CIs from persisted hypotheses (CPU only)
src/diag_recolor.py  fold == literal check + z-units vs ANOMALY.md
src/diag_maxnew.py   decoding-budget confound
```

`cc.py` deliberately re-implements `research-phase01-ver02/src/{config,tokutil,train_decoder}` rather
than importing them: both packages are named `src`, and `research-phase01-1.2/HANDOFF.md` documents
that importing across them fails on relative imports. The three functions copied verbatim
(`align_np`, the ridge convention, the recolor definition) carry a comment naming their source.
