# Learning note — ver02

Why the pipeline looks like this. Read `README.md` first for the commands; this file is the
reasoning, the measurements that forced each choice, and the traps that cost time.

## 0. The one-paragraph version

The research question is whether a 512×512 linear map can replace a translation model's encoder.
Two 60 M T5 models make that testable at a scale where generation is cheap (5 k sentences × 5
channels × 2 directions). The hard part is not fitting the map — with 9.3 M token pairs the ridge
solution is a single `numpy.linalg.solve` — it is proving that the *decoder*, not the map, is what
you think you are measuring. Hence the copy gate, the identity channel, the mean control, and the
paired bootstrap.

## 1. Five decisions, each with the measurement behind it

### 1.1 Per-token ridge as three moment matrices, not stored activations

9.3 M token pairs × 512 dims × 4 bytes = 19 GB for the source side alone, and the same again for
the target. The GPU has 4 GB and the laptop 7.7 GB of RAM, so the activations cannot exist.

Ridge only needs second moments. With `A = XᵀX`, `B = XᵀY`, `C = YᵀY` (plus sums for the intercept),
the normal equations are

    (A + λI) W = B,      b = ȳ − x̄ W

and `A, B, C` are additive over shards. Each batch is folded into three 512×512 float64 matrices
(6 MB total) and thrown away. Fitting is then `np.linalg.solve` on a 512×512 system — milliseconds.
**Trade-off:** you lose the ability to re-weight samples or fit a different loss without re-encoding.
For squared error that is a good trade; for a cosine loss it would not be.

### 1.2 The 512 "linear regressions" are one solve

The mission asks for 512 linear regressions, one per output dimension. Column *k* of `W` *is* the
least-squares coefficient vector of `y_k ~ x`. Solving the 512-right-hand-side system once is the
same estimator, not an approximation. This is worth stating in the paper because it makes the
comparison with LightGBM honest: LightGBM has no multi-output mode, so there the 512 fits are
literal and each one sees the same tokens.

### 1.3 LightGBM: bin once, swap the label

Measured on this laptop (108 k tokens × 512 features, 200 rounds, 15 leaves):

| setup | time per dimension |
|---|---|
| fresh `lgb.Dataset` each fit | 39.7 s |
| reuse one constructed `Dataset`, `set_label()` per dimension | 16.5 s |
| same, 100 k tokens, 150 rounds (what the pipeline uses) | ~14 s |

Binning dominates, and it does not depend on the label. So each of the 8 worker threads builds one
binned dataset and loops over its 64 dimensions. 512 × 14 s / 8 threads ≈ 15 min. **Trap:** the
first attempt created a Dataset per fit and was killed by a 30-minute timeout with no output,
because piping through `Select-Object` swallowed the progress lines. Never pipe a long run through
`Select-Object` if you want to watch it; redirect to a log file.

### 1.4 Coreset: the honest answer is "the shards are exchangeable"

The mission asks which of the 14 shards is the most representative. Protocol: 50 k rows per shard,
mean-pooled 512-d Vietnamese embeddings, centre, remove the first principal direction (raw
transformer states are strongly anisotropic and one direction hides everything), L2-normalise,
MiniBatchKMeans with K = 4 (MTet's four domains; also the number of dataset fields), then four
per-shard numbers.

Result: every raw metric differs in the **third decimal** (representativeness 0.051–0.052, cluster
entropy 0.995–0.997, outlier rate 0.048–0.052, coverage 1.000, JS divergence ≤ 0.0004), while a
100× within-shard bootstrap puts the SD of the composite score at ±0.46 on a 1.31 range. K = 4 picks
`train-00008`, K = 32 picks `train-00007`. **The shards are statistically indistinguishable**; the
ranking is sampling noise. We take the pre-registered K = 4 winner and report exchangeability as a
result, because "any shard is representative" is a stronger statement than an arbitrary pick.

**Trade-off:** the alternative was a full 6.67 M-sentence pass (44 min at the measured 2,540 sent/s).
Sampling 700 k rows costs 5 min and cannot change a difference that is already below the noise floor.

### 1.5 Two adapters, because they answer different questions

* **Copy adapter** (mission §2): `encoder(s) → decoder → s`, no prompt. It measures whether the
  decoder can render a vector that is *already correct*. If it cannot, no map can be blamed.
  Weakness: `T_out = T_in` invites a length-copy shortcut, so its BLEU is a diagnostic, not a
  translation score.
* **Translation adapter**: trained on `native` and `mapped` channels with real `T_out ≠ T_in`, so
  the BLEU gap between channels is attributable to the map.

Zero-shot copy on the 60 M English model: **9.6 % exact** (1,000 sentences). Phase 01 measured
2.9 % on flan-t5-base, so "small model" did not make the copy circuit reachable for free.

## 2. Traps paid for in this session

1. **`huggingface_hub` symlinks.** `snapshot_download` fails with `WinError 1314` (no Developer
   Mode) in the shared cache. Use `local_dir=`.
2. **`sentencepiece` vs `protobuf`.** The venv has protobuf 4.25.9; the installed sentencepiece
   wants `runtime_version` (protobuf ≥ 5.27). Every *slow* tokenizer crashes. Both new checkpoints
   ship `tokenizer.json`, so `T5TokenizerFast(tokenizer_file=...)` sidesteps it — and is the only
   way to get offsets from the Vietnamese checkpoint, whose config points at the slow class.
3. **Vocabulary overflow.** `t5-small-vietnamese` has `vocab_size=36096` but its tokenizer has
   36,100 ids (`<extra_id_96..99>`). They never occur in text; the encoder maps ids ≥ vocab to 0
   defensively.
4. **`file` column vs `LORA_FILE`.** The shard tag is `train-00000-of-00014` (no `.parquet`); the
   config constant originally had the extension, so the "dirty" shard was silently eligible for the
   coreset. It did not win, but the coreset was re-run after the fix.
5. **Loading 6.67 M strings.** `pq.read_table(...).to_pydict()` on the pairs file costs 2.4 GB of
   RAM and dropped free memory to 0.8 GB. Stream `iter_batches`, and materialise the chosen shard
   once (`data/chosen.parquet`, 30 MB) for every downstream stage.
6. **Throughput is not what the first smoke test says.** The 4 k-pair smoke test reported 563
   pairs/s because it includes model load and warm-up; the steady-state rate is ~1,700 pairs/s.
   Always read the *last* line of a progress log, not the first.

## 3. What I would do differently with more time

* Fit the map on *all* 13 eligible shards instead of one. With 9.3 M token pairs the ridge solution
  is already at the numerical-noise floor (the λ sweep is flat to 7e-9), so more data mainly helps
  the en→vi direction.
* Train the copy adapter with span dropout on the encoder input, so the length-copy shortcut is
  unavailable and its BLEU becomes interpretable.
* Replace the character-span alignment with word alignment; ~14 % of source tokens fall back to the
  nearest target token, which injects noise into exactly the tokens that matter (numbers, names).
