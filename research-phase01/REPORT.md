# Phase 1 report — per-token linear mapping vit5-base → flan-t5-base, vi → en

## 0. One line

A 768×768 ridge map fitted on **3,456,741 unique Vietnamese–English pairs** (all 8.33M rows of the
corpus, deduplicated, minus every row a previous agent had touched) transfers Vietnamese sentences from
vit5's representation space into flan's, and a flan decoder trained to read **either** channel is used to
ask whether the mapped representation carries enough to translate. Numbers in §4-§6; the corpus is 2×
duplicated and that is the first finding, not a truncation (§1).

## 1. What was asked / what was delivered

| Asked (`research-phase01/mission14Nov.md`) | Delivered | Evidence |
|---|---|---|
| §1 Vietnamese dish database (RecipeNLG schema) | **1,402 recipes** from two sources | `data-work/recipes_vn_all.csv` + `recipes_vn_all_manifest.json` |
| §1.2 atomic culture-fact database | **1,544 facts**, 7-relation vocabulary, page-level provenance | `data-work/structured/facts.db`, `culture_facts.jsonl` |
| §2 mapping over ~8M sentences, no leakage | **3,456,741 unique pairs** (8.33M rows: 2.0× duplicated) | `data/pairs_manifest.json` (sha256, split rule, exclusions) |
| §2 MT metrics (BLEU, chrF, COMET) | 3 channels × 2,000 test pairs, paired bootstrap | `results/metrics.json`, `results/gen_*.jsonl` |
| §3 COMET feasibility | smoke test passed before any reporting | §5.3 |
| §3.1 logistic probe on flan embeddings | 142,626 rows (100k recipes + 42,626 negatives) | `results/probe_metrics.json` |
| §3.2 decision model | morph/reflex `noul` via API, $0.000005/call | §6 |
| §3.3 probe vs decision model on MTET | agreement analysis, 2 classes, lexicon as third referee | `results/food_consistency.json` |

## 2. Corpus: "8M sentences" is 8.33M rows and 3.62M sentences

```
8,327,706 rows in data/data/*.parquet
7,228,344 after the 20–300 character filter
3,618,468 distinct Vietnamese sentences            (2.0× duplication)
3,456,741 after removing 297,409 texts used by earlier experiments
```

Split: `ORDER BY md5(vi)`, first 2,000 = test, next 2,000 = validation, rest = train. No RNG, so a rerun
cannot move a sentence between splits. Deduplication is mandatory, not cosmetic: with 2× duplication,
sampling a test sentence whose three copies live in the training set inflates retrieval@1 silently.

Excluded from training, validation and test: `research-scale-100k/data/pairs_100k.parquet`,
`finetune-and-lora/prototype/data/train_copy_50000.jsonl`, and `…/eval_sources.jsonl`.

## 3. Method

### 3.1 Representation

Per-token, not pooled. For a pair `(vi, en)`:

```
X = vit5-base encoder(vi)      (T_vi, 768)   frozen
Y = flan-t5-base encoder(en)   (T_en, 768)   frozen
```

`Y` is pooled onto `X`'s token positions with character-offset overlap (13.7 % of source positions have no
overlapping target token and take the proportionally nearest one). `X` and `Y` are the fitting pairs.

### 3.2 Fitting

Ridge in closed form from second moments, accumulated per 200k-pair shard:

```
A = XᵀX, B = XᵀY, C = YᵀY, Σx, Σy, n
W = (A − n x̄ x̄ᵀ + λI)⁻¹ (B − n x̄ ȳᵀ),  b = ȳ − x̄W
```

so nothing but 3 × 768² float64 crosses from GPU to host, and the fit is a reduction over shards
(18 shards × 200k pairs, resumable). `en→vi` costs nothing extra: swap `A↔C`, `B→Bᵀ`.

### 3.3 Generation, and why the decoder had to be retrained

The 100k run's BLEU table (0.141 mapped / 3.115 flan-encoder / 0.240 zero-shot) could not measure the
map: that decoder had been LoRA-tuned to **copy its input** and still emitted 23.9 % Vietnamese. Here one
adapter is trained on two channels with identical weights and identical targets:

| channel | what feeds the decoder's cross-attention |
|---|---|
| `native` | flan's own encoder reading the Vietnamese — a *reference* channel, not a ceiling (see §5) |
| `mapped` | vit5's encoder reading the Vietnamese, `xW + b` per token |
| `identity` | as `mapped` but without `W` (control, evaluated only) |
| `baseline` | not this decoder: `Helsinki-NLP/opus-mt-vi-en`, 77M params, a purpose-built vi→en model |

LoRA is attached to **decoder modules only** (`decoder.block.*.{SelfAttention,EncDecAttention,DenseReluDense}`
across 12 blocks). If the encoder moved, the native channel would drift out of the space `W` was fitted
into and the two channels would stop being comparable. Greedy decoding, `max_new_tokens=96`.

## 4. Mapping geometry

Fitted on **3,452,741 pairs / 85,089,885 aligned token pairs** (18 shards), evaluated on the 2,000
held-out test pairs. The λ sweep is flat: every λ in {0, 1e‑4 … 10} lands within 1e‑6 of the same
validation MSE (best gain 1.15e‑08), so λ = 0 (ordinary least squares) is used in both directions. With
85M token pairs the 768×768 normal equations are simply over-determined — the same conclusion the 100k
pooled run reached, now at 34× the data.

| direction | variant | token cos | token MSE | sentence cos | sentence MSE | retrieval@1 |
|---|---|---:|---:|---:|---:|---:|
| vi→en | **map** | **0.4515** | **7.485** | **0.7510** | **1.370** | **48.75 %** |
| vi→en | identity | 0.0038 | 687.92 | 0.0100 | 200.04 | 0.10 % |
| vi→en | mean predictor | 0.3234 | 8.464 | 0.5759 | 2.051 | 0.05 % |
| en→vi | **map** | **0.4512** | **530.45** | **0.8144** | **69.99** | **44.00 %** |
| en→vi | identity | 0.0038 | 687.92 | 0.0100 | 200.04 | 0.05 % |
| en→vi | mean predictor | 0.3563 | 587.66 | 0.6927 | 105.24 | 0.05 % |

Validation MSE: 7.630 (vi→en), 535.28 (en→vi).

How to read this table:

* **The identity control is the point.** Without a map the two spaces are orthogonal (cosine 0.0038) and
  retrieval is at chance. The 0.75 / 0.81 sentence cosine is learned.
* **The mean predictor is the trap.** Predicting the dataset mean earns sentence cosine 0.58 and 0.69 for
  free. Cosine alone would flatter a useless map; retrieval@1 (48.75 % / 44.0 % against 0.05 %) is what
  separates them.
* **MSE is not comparable across directions** (7.5 vs 530): the target spaces have different norms. Use
  cosine and retrieval@1.
* **Per-token fitting costs a little sentence-level accuracy.** The 100k run fitted *pooled* vectors and
  reached sentence cosine 0.853 with retrieval@1 51.4 % on a 97,743-item gallery; fitting per token and
  then pooling reaches 0.751 / 48.75 % on a 2,000-item gallery (an easier gallery). Fitting the pooled
  objective directly optimises the number it is scored on. Per-token is kept because generation is the
  downstream task, and generation needs a sequence, not a mean.

## 5. Translation quality

2,000 held-out pairs, greedy decoding, `max_new_tokens=80`, one adapter for `native` and `mapped`.
Intervals are paired bootstrap over sentences (1,000 resamples, same indices for every channel).
COMET is `Unbabel/wmt22-comet-da`.

| channel | BLEU | 95 % CI | chrF | COMET | degenerate | VI words in output | mean hyp words (ref 17.4) |
|---|---:|---|---:|---:|---:|---:|---:|
| **mapped** (proposal) | **6.28** | [5.86, 6.73] | **28.15** | **0.600** | 14.4 % | 0.00 % | 20.6 |
| native (flan enc. reads VI) | 3.85 | [3.43, 4.34] | 18.72 | 0.443 | 21.6 % | 0.10 % | 21.7 |
| identity (no map) | 0.01 | [0.01, 0.01] | 10.99 | 0.288 | 3.0 % | 3.62 % | 18.7 |
| *baseline: opus-mt-vi-en (77M)* | *22.27* | *[21.39, 23.25]* | *45.31* | *0.752* | *0.2 %* | *0.10 %* | — |

Paired bootstrap, difference in BLEU:

| comparison | mean | 95 % CI | P(other ≤ mapped) |
|---|---:|---|---:|
| native − mapped | −2.44 | [−3.03, −1.82] | 1.000 |
| identity − mapped | −6.27 | [−6.72, −5.85] | 1.000 |
| baseline − mapped | +15.98 | [+15.07, +17.00] | 0.000 |

### 5.1 What the numbers say

1. **The map does the work.** Removing it (`identity`) collapses BLEU to 0.01 and produces token salad
   (`"custom hole custom hole custom hole …"`, `"alt alt alt alt stump"`): the decoder never saw raw vit5
   states, so it has no idea what to do with them. Every other channel is significantly better
   (P = 1.000 over 1,000 paired resamples).
2. **The mapped channel beats flan's own encoder reading Vietnamese** by 2.44 BLEU and 0.157 COMET, also
   significant. This is the substantive finding: a Vietnamese-aware encoder (vit5) plus a 590k-parameter
   linear map is a *better* way to feed this English decoder than letting flan's English-centric encoder
   read Vietnamese, where it hallucinates fluent but unrelated English
   (`"I'm not sure what the hell I'm doing here."` for a sentence about adolescent cognition).
3. **It is not a translation system yet.** A 77M-parameter purpose-built NMT model reaches 22.27 BLEU and
   0.752 COMET on the same sentences — 16 BLEU above the whole pipeline. The honest claim is about
   representation transfer, not competitive MT. The gap is explained by the decoder budget (250M params,
   1 epoch, 100k pairs, decoder-only LoRA) rather than by the map, but this run cannot prove that.
4. The failure mode changed: 71.6 % of hypotheses were degenerate token loops in the 100k pooled run
   (BLEU 0.141); here 14.4 % are, and Vietnamese leakage is 0.00 % against 23.9 % for that run's
   flan-encoder control.

### 5.2 A caveat the examples forced

At least one of twelve inspected test pairs is **misaligned in the corpus itself** (source about
adolescent cognition, reference about a questionnaire). The mapped channel translated the *source*
correctly, so BLEU charged it for a reference error. Corpus noise therefore bounds every number in this
table from above, and a per-sentence manual audit of 100 references would be needed to quantify it. It was
not done here.

### 5.3 COMET feasibility (§3 gate)

`Unbabel/wmt22-comet-da` downloads 1.4 GB, loads on the 4 GB card and scores: a good pair 0.854, and
0.878 / 0.831 per sentence. Note the dependency trap in `README.md`: `unbabel-comet` pins `numpy<2`,
which breaks `pyarrow 26`; NumPy 2.5.4 was restored and COMET 2.2.7 runs fine against it.

## 6. Food probe and decision-model agreement (§3)

### 6.1 What the probe actually learned

142,756 rows: 100,000 RecipeNLG recipe fields (positive) and 42,756 MTET English sentences containing no
lexicon food term (negative). flan-t5-base encoder, masked-mean, 768 d, no LoRA.

| split | n | AUC | accuracy | F1 |
|---|---:|---:|---:|---:|
| validation | 21,404 | 0.9999994 | — | — |
| test | 21,404 | 0.99999997 | 0.99986 | 0.99990 |

**That number is not a food detector, it is a domain detector.** Recipe fields are comma-separated
ingredient lists; MTET rows are news and legal prose. A linear probe separates those two styles almost
perfectly, so §3.1 as specified cannot answer "does the encoder know a food name when it sees one". The
class imbalance is also real (100k vs 42.7k, because the lexicon rejects ~89 % of MTET sentences).

### 6.2 Probe vs decision model on MTET (§3.3)

2,000-sentence pool → 200 sentences, 100 deliberately containing a food lexeme and 100 without. Probe
score = max over 8-word sliding windows (stride 4); decision model = `morph/reflex` `noul` on the whole
sentence ("Does this sentence name or describe a food, a dish, or an ingredient?"), 200/200 answered, $0.001.

| quantity | value |
|---|---:|
| agreement (probe ≥ 0 vs decision > 0.5) | **98.5 %** |
| Cohen's κ | **0.00** |
| both say "food" | 0 |
| both say "not food" | 197 |
| decision-only "food" | 3 |
| mean decision probability | 0.0138 |
| probe vs lexicon (accuracy / κ) | 0.500 / 0.00 |
| decision vs lexicon (accuracy / κ) | 0.515 / 0.03 |

**The 98.5 % agreement carries no information.** κ = 0 because the two methods never disagree except in
three cases; they agree by both answering "no" to everything. Against the lexicon — which says half of
these sentences do contain a food name — both are at chance. So neither a domain-trained linear probe nor
a single sentence-level `noul` decision detects food mentions in general prose; the decision needs
span-level detection (or a rubric that distinguishes "mentions a food" from "is about food"), which is a
next-step item, not something this run can claim.

## 7. What this does NOT show

* **en → vi has geometry only.** Generating Vietnamese needs the vit5 decoder; the map for that
  direction exists (`data/map_en_vi.npz`) but no BLEU.
* **The decoder is base-size, one epoch, 100k pairs.** Absolute BLEU is capped by the decoder, not by the
  map; the claim is the *gap between channels*, which is a paired comparison on the same weights.
* **The probe's labels are weak and domain-confounded.** Recipes are formulaic, MTET prose is not, so part
  of the AUC is style. It is used cross-domain in §6, where a domain classifier is the honest reading.
* **Character-overlap alignment is a proxy.** 13.7 % of token positions are attached by nearest-neighbour.
* **`identity` is an out-of-distribution control** for the decoder (it never saw unmapped vit5 states), so
  its failure cannot be attributed solely to "W is doing the work".
* **US cultural facts are secondary sources.** The three "American" PDFs are two Bookey study guides and
  one publisher blurb, not the books (§8).

## 8. Data track (M1), with its caveats

| Source | Pages | Recipes | Facts |
|---|---|---|---|
| `mon_an_vn_truyen_thong.pdf` | 223 | 167 | 423 |
| `van-hoa-am-thuc-viet-nam…pdf` | 260 | — | 998 |
| `American Cuisine PDF.pdf` | 31 | — | 55 |
| `The Food of a Younger Land PDF.pdf` | 70 | — | 68 |
| `The Sioux Chef's…pdf` | 2 | — | 0 |
| `Kỹ thuật nấu ăn toàn tập.md` | — | 1,235 | — |

OCR: `qwen/qwen3.7-flash` vision, 180 DPI, $0.0667 for 586 pages. Structuring: same model, $0.0861
(recipes+facts) + $0.1245 (cookbook). Total DataWorker spend **$0.277 of $1.00**.

Caveats: the American files are study guides/blurbs, not the books; 57 % of cookbook titles were
reconstructed where the source text is encoding-damaged (grounded in ingredients/steps, raw title kept in
`cache/`); ingredient text keeps OCR damage, so quantities are unreliable.
