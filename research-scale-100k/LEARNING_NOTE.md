# Learning note — the 100k linear mapping (read after REPORT.md)

Written for a reader who knows Python but has not done representation mapping before.
Numbers quoted here are the ones in `results/`; if they differ, the CSV wins.

## 1. What we are actually testing

The paper asks one narrow question: *can a learned map carry a sentence from one model's
representation space into another's, well enough that the second model generates the translation?*

Three things have to be true for that to work, and they fail independently:

| Layer | Failure looks like | How this folder measures it |
|---|---|---|
| Data | the pairs are not parallel, or one side is mis-detected | `data/pairs_manifest.json` + the sample in the notebook |
| Geometry | mapped vectors are far from gold vectors | `mse`, `cosine`, `retrieval@1` |
| Generation | vectors are close but the decoder cannot use them | BLEU of `mapped` vs `flan_encoder` |

Reporting only BLEU would hide layer 2; reporting only MSE would hide layer 3. `paperOverview.md`
§5.2 says exactly this, and it is the reason both tables exist.

## 2. Why a *linear* map, and why ridge

A linear map `W: R^768 -> R^768` (plus a bias) is the smallest hypothesis that can express "these two
spaces are related by a rotation, a scaling, and a shear". If it fails, a 2-layer MLP is the next
step; if it succeeds, there is nothing to explain.

Closed form, centred, with a ridge term:

```
W = (Xcᵀ Xc + λ I)⁻¹ Xcᵀ Yc ,   b = ȳ - x̄ᵀW
```

`Xc` is 97,743 × 768, so `Xcᵀ Xc` is a 768² matrix — the whole fit is a second of numpy, no GPU, no
epochs, no learning rate. λ is scaled by `n/1000` in the code so the sweep means the same thing for
the val split. **Trade-off**: `torch.linalg.lstsq` on the raw 97k × 768 design would be more stable
numerically but ~100× slower; the normal-equation path is fine here because the features are
principled (encoder states, not raw pixels) and λ > 0 removes the conditioning worry.

## 3. Why masked-mean pooling

A T5 encoder returns 768-dim vectors *per token*. The mapping needs one vector per sentence. Options:

1. **Masked mean** (used): stable, cheap, ignores padding. Dilutes word order.
2. **Last token**: carries more syntactic state for T5, but for a 60-token sentence it is one token's
   worth of information.
3. **Per-token mapping** (keep the whole (T, 768) matrix): preserves word alignment, lets the decoder
   copy order, and is the obvious next experiment — but the two models tokenise differently, so
   token-level correspondence has to be established first.

The single pooled vector is what makes the BLEU number in this folder *hard*: the decoder gets one
"token" of context. If a later experiment maps per-token states and BLEU jumps, that isolates
pooling as the bottleneck. That is a clean next step, not a hand-wave.

## 4. Reading `retrieval@1`

For each val prediction, take the cosine to all 97,743 train targets and ask whether the true target
is the nearest one. Chance level is 1/97,743 ≈ 0.001 %. Two controls are reported:

* `identity` — use the source vector unchanged. This is the honest baseline: if the two spaces already
  overlapped, the map would be pointless.
* `mean` — predict the target mean for every sentence. This is the floor: a map that is worse than
  this has learned nothing.

## 5. The BLEU control, and why the first version of this section was wrong

`flan_encoder` feeds flan's *own* encoder with the Vietnamese sentence and generates with the same
decoder. The intent was to measure the decoder's ability to produce English from a pooled 768-vector of
Vietnamese text. The measurement says something else: that decoder carries the **copy adapter**, so it
*reproduces its input*. 23.9 % of its output words are still Vietnamese, against 0.1 % in the English
references. Its 3.115 BLEU is mostly proper nouns and numbers that survive the copy.

So the three-way comparison is:

* `mapped` — 71.6 % degenerate loops, 0 % Vietnamese, BLEU 0.141. The vector is out of distribution.
* `flan_encoder` — a copy of Vietnamese, BLEU 3.115. **Not a ceiling, not a translation.**
* `zeroshot_vit5` — 80.9 % degenerate, prompt echoed, BLEU 0.240.

**The lesson, which generalises beyond this project**: a BLEU table cannot distinguish *bad translation*
from *echo* from *loop*. Compute cheap structural diagnostics on the hypotheses before interpreting any
score — share of source-language words, unique/total token ratio, empty rate
(`tools/diagnose_generation.py`, 12 lines). Doing that first would have saved the wrong paragraph that
this section originally contained.

Therefore, with this decoder, BLEU cannot attribute loss to the map at all. Fix the decoder (translation-
tuned, not copy-tuned) before re-running the comparison. What *is* attributable today is the geometry
(§4 of this note's sibling `REPORT.md`), because it does not involve generation.

## 6. Reproduce in four commands

```powershell
cd research-scale-100k
.\.venv\Scripts\python.exe -m src.run_all build     # ~30 s, CPU
.\.venv\Scripts\python.exe -m src.run_all encode    # ~5 min on the RTX 3050
.\.venv\Scripts\python.exe -m src.run_all map       # ~1 min, CPU
.\.venv\Scripts\python.exe -m src.run_all bleu      # ~15 min on the RTX 3050
```

Every stage skips finished work unless `--force` is passed, so a rerun after a crash is safe.

## 7. Pitfalls that cost time on this machine

1. **4 GB VRAM, one model at a time.** `vit5-base` and `flan-t5-base+LoRA` together do not fit; every
   stage loads one, uses it, and calls `torch.cuda.empty_cache()` before the next.
2. **`PeftModelForSeq2SeqLM` has no `.encoder`.** Attribute access silently reaches the wrapper, so
   `model.encoder` can raise. `encode.py` calls `merge_and_unload()` first: the adapter is folded into
   the weights, the module becomes a plain T5, and the forward pass is faster besides.
3. **`SET seed = 42` is not a DuckDB setting.** `USING SAMPLE ... (reservoir, 42)` takes the seed as an
   argument; a `SET` statement fails with `Catalog Error`.
4. **A quoted secret is not a secret.** `ExtractionKey="sk-or-..."` in `.env` returns 401 until the
   quotes are stripped. Both keys in this project are stored quoted.
5. **Yesterday's 4.9 sent/s was contention, not the model.** The same flan-t5-base LoRA run does
   26-38 sent/s when it owns the GPU. Before concluding "this model is slow", check what else is
   using the device.

## 8. What to try next, cheapest first

0. **Replace the copy-tuned decoder** with a translation-tuned one (or LoRA-tune flan-t5 on the translation
   task from the same 97k pairs). Until then BLEU says nothing about the map. This is the one change that
   makes the rest measurable.
1. Per-token mapping (align tokenisers with `offset_mapping`, map each vit5 state into flan's space,
   hand the whole sequence to the decoder). Highest expected gain on the representation side, ~1 h.
2. One 2-layer MLP + GELU between the same matrices. If cosine does not move, the relation really is
   close to linear.
3. Domain split: the corpus mixes legal text and literature. BLEU on food/culture sentences may differ
   sharply from BLEU on `Điều 80`-style legal text; report the split, not the average.
4. Run `tools/diagnose_generation.py` before writing any conclusion from a BLEU table. Twelve lines.
