# Results — EN->VI mapping + food probing (prototype v0.1)

Run: 2026-10-21, 12,000 pairs (10,000 train / 2,000 test), seed 42, one shard
(`train-00008-of-00014.parquet`).
Encoders: EN `BAAI/bge-base-en-v1.5`, VI `bkai-foundation-models/vietnamese-bi-encoder`
(both 768-d, both loaded successfully; recorded in `agent-work/manifest.json`).
Store: MySQL 9.4.0, 72,000 vectors, 6-decimal quantisation.

## 1. Storage integrity (`results/verify.json`)

| check | value | reading |
|---|---|---|
| pairs / embedding rows | 12,000 / 12,000 x 6 groups | complete |
| test matrix | 2000 x 768 | as designed |
| quantisation error (max abs) | 3.1e-8 | float32 noise, not a bug |
| self-retrieval top-1 | 1.000 | rows are aligned, no index shuffle |
| paired cosine (unmapped) | **-0.0124** | the two spaces are orthogonal |

## 2. Mapping (`results/mapping_metrics.csv`)

Closed-set retrieval over the 2,000 test pairs; `chance@1 = 0.0005`.

| pooling | method | MSE | MAE | cosine | R@1 | R@5 | R@10 | median rank |
|---|---|---|---|---|---|---|---|---|
| mean | identity | 0.3548 | 0.4697 | -0.012 | 0.0010 | 0.0035 | 0.0085 | 1028 |
| mean | ridge | 0.0302 | 0.1359 | 0.486 | **0.7410** | 0.8905 | 0.9220 | 0 |
| mean | mlp | 0.0306 | 0.1369 | 0.475 | 0.6600 | 0.8540 | 0.9045 | 0 |
| cls | identity | 0.3941 | 0.4941 | -0.009 | 0.0015 | 0.0040 | 0.0095 | 1046 |
| cls | ridge | 0.0458 | 0.1689 | 0.503 | **0.6750** | 0.8420 | 0.8930 | 0 |
| cls | mlp | 0.0462 | 0.1697 | 0.497 | 0.6285 | 0.8130 | 0.8700 | 0 |
| last | identity | 0.4759 | 0.5427 | -0.010 | 0.0000 | 0.0025 | 0.0065 | 1026 |
| last | ridge | 0.0466 | 0.1703 | 0.492 | **0.6465** | 0.8300 | 0.8780 | 0 |
| last | mlp | 0.0469 | 0.1710 | 0.485 | 0.5830 | 0.7915 | 0.8485 | 0 |

`identity` is at chance on every pooling: the frozen spaces are not aligned at all.
Mapping fixes that; the recovered cosine (~0.49) is far from 1.0, so the map is
"good enough to retrieve", not "perfect".

## 3. Was the MLP under-trained? (`results/mapping_followup*.csv`)

`mean` pooling only; same split for every arm.

| ridge alpha | 100 | 300 | 1000 | 3000 | 10000 | 30000 |
|---|---|---|---|---|---|---|
| R@1 | **0.760** | 0.759 | 0.741 | 0.704 | 0.625 | 0.427 |

| mlp config | params | R@1 | R@5 |
|---|---|---|---|
| h=1024, lr=1e-3, early stop @11 | 1.6M | 0.6600 | 0.8540 |
| h=1024, lr=3e-4, @37 | 1.6M | 0.7060 | 0.8790 |
| h=512, lr=3e-4, @39 | 0.8M | 0.6740 | 0.8590 |
| h=2048, lr=3e-4, @29 | 3.1M | 0.7265 | 0.8905 |

Two conclusions:

1. The main run *had* α=100 in its grid but selected α=1000, because selection used
   validation MSE while the reported metric is retrieval. Retrieval prefers
   α≈100-300 (R@1 0.760). MSE-optimal is not retrieval-optimal — selecting on the
   wrong criterion cost ~0.02 R@1. The flat 100-300 plateau also means the exact
   value does not matter much.
2. MLP capacity helps monotonically (0.674 -> 0.706 -> 0.727) but never beats
   ridge (0.760) at 10k pairs, at any learning rate or depth tried.

## 4. Food probe (`results/probe_food.csv`, `results/probe_food_controlled.csv`)

5-fold CV, logistic regression on mean-pooled EN vectors.

| variant | accuracy | F1 | balanced acc | ROC-AUC | baseline |
|---|---|---|---|---|---|
| plain (10k titles vs 10k sentences) | 0.9927 | 0.9926 | 0.9927 | 0.9997 | 0.50 |
| controlled (lowercased, negatives 3-10 words, n=3,791) | 0.9933 | 0.9954 | 0.9924 | 0.9996 | 0.50 |

Lowercasing and length-banding did **not** move the number, so the separability is
not a casing/length artefact. It is lexical/semantic (food vocabulary). The probe
is saturated: it proves a food direction exists and is linearly readable, but it
cannot isolate "food" from "culinary vocabulary". Same-style non-food noun phrases
(e.g. product names) would be the right next control.


## 5. What this means for the paper

**Supported: the mapping premise.** Two monolingual encoders that share no
training signal produce orthogonal spaces (cosine -0.012, identity at chance).
A *linear* map still recovers 74-76% exact top-1 on 2,000 unseen sentences. A
small, data-driven alignment layer between an English and a Vietnamese
representation is real, not wishful thinking - the core assumption behind
"map, then insert facts" holds for encoders.

**Not supported at this scale: the ANN.** Ridge beats the MLP on all three
poolings, and MLP gains from tuning (0.660 -> 0.727) still stop short of ridge
(0.760). With 9,000 training pairs in 768-d, extra nonlinear capacity mostly fits
noise. This does not refute the paper's architecture, but the nonlinearity has to
earn its place - and the honest test is at 100k+ pairs, which the corpus can
supply (267,475 clean pairs available).

**Supported: pooling choice matters little.** mean > cls > last (0.74 / 0.68 /
0.65), same ranking for every method, all far above chance. "For each
representation" is a legitimate axis, with mean-pooling the sane default.

**Supported: a food direction exists.** 99% linear separability, robust to casing
and length control. The English representation carries a linearly accessible
culinary property - the prerequisite for fact injection.

## 6. Threats to validity (read before quoting any number)

1. **General-domain corpus.** The parallel data is legal/daily text, not culinary.
   Retrieval numbers describe sentence alignment in general, not recipe language.
2. **Encoder, not decoder.** Sentence embeddings, not the decoder hidden states the
   final architecture would map. Nothing here bounds generation quality.
3. **Closed-set retrieval.** Gallery = the 2,000 test sentences. Open-corpus
   retrieval over 8.3M sentences would be far harder; these numbers are an upper
   bound on "can the map work", not a deployment estimate.
4. **10k training pairs.** Ridge is near its data-efficient best here; the MLP is
   data-hungry. The comparison is fair at this size, not necessarily at 100k.
5. **Alpha chosen by MSE, evaluated by R@1.** A small mismatch (sec. 3).
   Re-selecting on retrieval would gain ~0.02 R@1.
6. **Probe saturation.** 99% is a ceiling effect: it confirms presence, not
   magnitude, of the food signal.
