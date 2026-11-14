# Learning note — phase 1 (mission 14Nov)

Written for a reader who wants to *understand the choices*, not just rerun them. Every number here
was measured on this laptop (RTX 3050 4 GB, 7.7 GB RAM, i5-12500H), and every claim that surprised me
says so.

---

## 1. The corpus is not 8M sentences, it is 8.3M rows

```
8,327,706 rows in data/data/*.parquet
7,228,344 survive the 20-300 character filter
3,618,468 distinct Vietnamese sentences          <-- 2.0x duplication
3,456,741 after removing every text a previous agent touched
```

Two consequences, both important:

1. **Dedup is not cosmetic, it is a leak control.** If sentence *s* appears three times and only one
   copy is sampled into the test split, the map is fitted on the other two and retrieval@1 is measured
   on a sentence the model has already seen. This is the most common silent inflation in this kind of
   experiment.
2. **"Trained on 8M sentences" would be false.** The honest sentence is "fitted on 3.46M unique
   pairs (all 8.33M rows of the corpus, deduplicated)". The report says exactly that.

Split rule: `ORDER BY md5(vi)`, first 2,000 = test, next 2,000 = val. No RNG, so a rerun cannot move a
sentence between splits, and the split is a pure function of the text.

Exclusions come from `config.EXCLUDE` — the 100k pair file, the 50k copy-LoRA training file, and the
4k copy-eval sources. 297,409 distinct texts were blocked this way.

## 2. Where the time goes, and why the first version was 2.5x too slow

The first working batch loop reached **140 seq/s** (6.8 h for the corpus). Profiling stage by stage:

| stage | ms / batch (B=96) | note |
|---|---|---|
| tokenise both sides | 22 | nothing to win |
| vit5 encoder | 28 -> really 248 at B=192 | the profiler lied until `torch.cuda.synchronize()` |
| flan encoder | 18 -> really 309 at B=192 | same |
| `.float()` on GPU | 229 (apparent) | it was the sync, not the cast: a 9.4M-element cast is 0.2 ms |
| `align` on CPU | 90 | moved to GPU |
| moments on CPU | 133 | moved to GPU, `float64` accumulator |

**Lesson that cost the most time:** asynchrony makes naive per-stage timing meaningless. The first
profile said "the encoders cost 46 ms", which was the *launch* cost. With `torch.cuda.synchronize()`
the truth was 557 ms for the pair, i.e. 345 seq/s, and the encoders were 100 % of it.

Three fixes took it to **347 seq/s on random-length text and ~600 seq/s on short sentences**:

1. **Pad to the batch maximum, not to `MAX_LEN`.** Vietnamese averages 29.7 subwords and English 27.6,
   so padding to 64 doubles the FLOPs for the median sentence. Rounded to multiples of 8 this is free
   accuracy and ~2x throughput.
2. **Do the cross-tokeniser pooling on the GPU** as one bf16 matmul with a 0/1 overlap matrix
   (`B x T_vi x T_en`), instead of a numpy einsum.
3. **Accumulate A = XᵀX, B = XᵀY, C = YᵀY on the GPU in float64** and copy 3 x 4.7 MB once per shard.
   A naive per-batch host copy is 2000 round trips per shard for no reason.

## 3. Why ridge needs only three matrices

The direct approach — encode 3.46M pairs, keep `(T,768)` activations — needs ~130 GB. The closed-form
ridge solution only needs second moments:

```
A = XᵀX, B = XᵀY, C = YᵀY, sx = Σx, sy = Σy, n
W = (A - n x̄x̄ᵀ + λI)⁻¹ (B - n x̄ȳᵀ),   b = ȳ - x̄W
MSE(λ) = [tr(WᵀA_cW) - 2tr(WᵀB_c) + tr(C_c)]/n + ‖b + x̄W - ȳ‖²
```

so a shard is a *reduction*, not a dataset. That gives three things for free: the map at any prefix of
the corpus, resumability (a shard is 200k pairs and a 14 MB npz), and λ selection on validation
without ever re-encoding anything. The whole fit is 18 shards x ~7 MB of statistics.

Precision: activations are bf16 (that is what inference uses anyway) and are cast to float32 before
the moments; the moments are float64 because `A` entries reach ~10⁸ and float32 has 7 digits.

## 4. Alignment across two tokenisers that disagree

The pipeline needs, for every Vietnamese subword, a target vector from the English encoder. The two
tokenisers cut differently, so the mapping is done on **character offsets**:

* `VietAI/vit5-base` ships a *slow* sentencepiece tokenizer; `return_offsets_mapping` raises `KeyError`.
  `convert_slow_tokenizer` + `tokenizers.Tokenizer.encode_batch` does return offsets, verified by
  reconstructing the source string exactly.
* Target token *j* is attached to source token *i* when their spans intersect.
* 13.7 % of source tokens (whitespace pieces, punctuation dropped by the other normaliser) intersect
  nothing. They are filled from the proportionally nearest target token rather than dropped, because a
  decoder reading a sequence with holes is a different experiment from one reading a full sequence.

This is the weakest part of the method and the report says so: attaching an English token to a
Vietnamese token that only shares whitespace is a guess, and 13.7 % of positions are that guess.

## 5. Why the generation experiment had to change

The 100k run's numbers (BLEU 0.141 mapped / 3.115 flan-encoder / 0.240 zero-shot) were not a
measurement of the map, because that decoder had been LoRA-tuned to **copy its input**. Its
"flan-encoder control" still emitted 23.9 % Vietnamese words: it was echoing, not translating.

The fix is a decoder that can read *either* channel, trained once, with the same weights and the same
pairs:

| channel | encoder | what it measures |
|---|---|---|
| native | flan reads the Vietnamese | a reference channel: the decoder's own attempt when the source arrives in its native tokenisation |
| mapped | vit5 reads the Vietnamese, W projects every token | the proposal |
| identity | vit5 reads the Vietnamese, no W | control: is W doing the work? |

The claim is the **gap** between channels, not the absolute BLEU: with a base-size decoder and one epoch,
the absolute number is limited by the decoder, not by the map. Measured outcome: mapped 6.28 BLEU beats
native 3.85 (paired bootstrap −2.44 [−3.03, −1.82]) and identity 0.01, while a 77M dedicated NMT model
reaches 22.27 — so the map is doing real work but is not competitive MT.

LoRA is attached **to decoder modules only**. If the encoder moved, the "native" states would drift
away from the space W was fitted into and the two channels would stop being comparable.

## 6. Decisions I would defend, and the ones I would revisit

**Defend**
* Dedup before splitting (leak control) and md5 split (no RNG).
* Streaming second moments instead of storing activations.
* Per-token rather than pooled: a pooled vector gives the decoder one token of context. That is the
  configuration that produced 71.6 % token loops at 100k.
* Reporting the identity control for both geometry and generation.

**Revisit**
* Character-overlap alignment is a proxy. Word-level alignment (fast_align) or an attention-based soft
  alignment would remove the 13.7 % guess, at a compute cost this laptop cannot pay today.
* The probe's labels are weak: recipes are formulaic, MTET prose is not, so a high AUC partly measures
  *style*. §3.3 uses it cross-domain, which is the honest use, but the label noise must be stated.
* One mixed decoder shares capacity between two channels. Two separate adapters would remove that
  confound and cost one more hour of GPU.
