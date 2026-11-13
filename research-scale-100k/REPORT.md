# Linear mapping vit5-base ↔ flan-t5-base at 100k — report

## 0. One line

The linear map is **geometrically strong but nothing downstream uses it**: mapped vectors land at cosine
**0.854** with **51.4 % retrieval@1** against a 97,743-sentence gallery (identity baseline: 0.008 / 0.1 %),
yet all three generation runs fail in different ways — mapped output is **71.6 % degenerate token loops**
(BLEU 0.141), the flan-encoder control **echoes the Vietnamese** because its decoder is the copy adapter
(BLEU 3.115, 23.9 % Vietnamese words left in the output), and zero-shot vit5 **repeats the prompt**
(BLEU 0.240, 80.9 % degenerate). None of the three is a translation. The geometry result is real; the
generation result is a negative one, and the report says which component is responsible.

## 1. What was asked / what was delivered

| Asked (`data/mission.md` §3) | Delivered | Path | Verified by |
|---|---|---|---|
| 100k sentences | **99,743** pairs (97,743 / 1,000 / 1,000), seed 42 | `data/pairs_100k.parquet` + `data/pairs_manifest.json` | `rows` + `sha256 4f1051e1…` in the manifest |
| mapping vit5-base → flan-t5-base (LoRA'd) | closed-form ridge, both directions | `data/map_vi_en.npz`, `data/map_en_vi.npz` | `results/ridge_sweep.csv`, `results/mapping_metrics.csv` |
| new folder, `.venv` copied | `research-scale-100k/`, venv copied (6.24 GB) | `research-scale-100k/.venv` | `setup.ps1` is the from-zero path |
| notebook preferred | executed in place, real outputs | `01_mapping_100k.ipynb` | `tools/execute_notebooks.py` |
| "Check BLEU" | 3 generation runs, n = 1,000 | `results/bleu_vi_en.json`, `results/gen_*.jsonl` | §3.2 |
| M2 prerequisite (flan-t5-base LoRA) | **DONE** 2.9 % → 99.8 % exact copy | `../finetune-and-lora/adapters/flan-t5-base/` | `results/before_after.csv` of that folder |

## 2. Method, with the cost of each stage

| Stage | What | Wall clock | Hardware |
|---|---|---|---|
| `build` | dedupe by Vietnamese side, keep 20–300 chars, reservoir sample 100k (seed 42), detect the vi side by diacritic/tone count, split by index | 25 s | CPU + DuckDB |
| `encode` | masked-mean of the last encoder state, 768-d, MAX_LEN 64, batch 96 | 118.8 s (839 seq/s) vit5 · 124.8 s (799 seq/s) flan+LoRA · 1.9 s control | GPU |
| `map` | ridge closed form on 97,743 × 768, λ swept on val, 2 directions | 216 s (dominated by the 97k-gallery retrieval, not the fit) | CPU |
| `bleu` | 3 generation runs × 1,000 sentences, greedy, max 96 new tokens | 101 s mapped · 100 s flan-encoder · 73 s zero-shot | GPU |

Representation choice: **one pooled vector per sentence** (as the "final hidden layer" starting point in
`paperOverview.md` §3.3). The decoder therefore receives a single "token" of context; that is a property
of this experiment, not of the corpus.

`flan-t5-base` is used **with its LoRA copy adapter attached** (`merge_and_unload()`), because the mission
specifies the LoRA'd model. The map is fitted into *that* space.

## 3. Results

### 3.1 Geometry (test split, 1,000 queries against the 97,743 train targets)

| direction | λ | mse | cosine | retrieval@1 |
|---|---|---|---|---|
| vi → en | 0.0 | 0.00088 | **0.8535** | **0.514** |
| en → vi | 0.01 | 0.07018 | **0.8577** | **0.506** |
| vi → en, *identity control* | — | 0.26078 | 0.0083 | 0.001 |
| vi → en, *mean control* | — | 0.00163 | 0.7028 | 0.000 |
| en → vi, *identity control* | — | 0.26078 | 0.0083 | 0.000 |
| en → vi, *mean control* | — | 0.12161 | 0.7324 | 0.000 |

Read this carefully:

* **The identity control is the point.** Without a map, the two spaces are almost orthogonal (cosine
  0.008) and retrieval is at chance. So the 0.85 cosine is learned, not inherited.
* **The mean control is the trap.** Predicting the dataset mean gives cosine 0.70 for free — cosine alone
  would flatter a useless map. Retrieval@1 (0.514 vs 0.000) and MSE are what separate them.
* **λ = 0 wins in one direction.** No regularisation helps, i.e. with 97,743 rows the 768×768 normal
  equations are well-conditioned. That is evidence the relation really is close to linear.
* **MSE is not comparable across directions** (0.0009 vs 0.07): the target spaces differ in norm. Use
  cosine and retrieval@1 for cross-direction comparison.

### 3.2 Generation (vi → en, n = 1,000, greedy, sacrebleu 13a)

| run | BLEU | chrF | empty outputs | mean hyp tokens (ref 17.5) |
|---|---|---|---|---|
| **mapped** (vit5 encodes vi → map → flan+LoRA decoder) | **0.141** | 4.947 | 14 | 22.0 |
| **flan_encoder control** (flan+LoRA encoder reads the Vietnamese, same decoder) | **3.115** | 10.424 | 0 | 15.1 |
| **zero-shot vit5-base** (prompt "Dịch sang tiếng Anh: …") | 0.240 | 6.245 | 0 | 60.2 |

### 3.3 What the three rows mean

A BLEU number alone cannot distinguish "bad translation" from "echo" from "loop", so
`tools/diagnose_generation.py` measures three properties directly on the same 1,000 hypotheses:

| run | VI words in output | degenerate (unique/total ≤ 0.5) | empty | copy-of-input |
|---|---|---|---|---|
| mapped | 0.0 % | **71.6 %** | 1.4 % | 0.0 % |
| flan_encoder | **23.9 %** | 0.4 % | 0.0 % | 0.0 % |
| zero-shot vit5 | **54.6 %** | **80.9 %** | 0.0 % | 0.0 % |

(References: 0.1 % Vietnamese words. Inputs: 80.2 %.)

1. **Mapped output is not bad English, it is a loop.** 71.6 % of the hypotheses are token repetitions
   (`"the a n the a , a ..."`, `"a a a a ..."`). The decoder treats the mapped vector as an
   out-of-distribution state and collapses — the failure mode named in `paperOverview.md` §5.1.
2. **The `flan_encoder` control is contaminated and cannot serve as the ceiling.** The decoder carries the
   copy adapter, so when its encoder reads Vietnamese it *copies Vietnamese*: 23.9 % of output words still
   carry Vietnamese diacritics, versus 0.1 % in the references. Its 3.115 BLEU comes mostly from proper
   nouns and numbers shared with the reference (place names, `12.000`, `Ersatz Kaiserin Augusta`), not from
   translation. My first draft called this "the decoder can translate when it reads the right encoder" —
   the diagnostic disproves that claim, and the draft was wrong.
3. **Zero-shot vit5 loops too** (80.9 %), repeating the prompt `"Dịch sang tiếng Anh"` in 54.6 %
   Vietnamese-word output; its 0.240 BLEU is noise, not signal.
4. **Therefore the generation result is negative for the whole pipeline, not just for the map.** With this
   decoder (copy-tuned, base-size) a single pooled vector — correct or mapped — does not produce a
   translation. The map cannot be blamed or exonerated by BLEU alone at this configuration.
5. **The geometry result survives intact.** Cosine 0.854 and retrieval@1 51.4 % against an identity control
   of 0.008 / 0.1 % mean the two 768-d spaces are related almost linearly in 97,743 paired examples. That
   claim is unaffected by the decoder's collapse, and it is the reusable part of today's work.

## 4. What this does NOT show

* **It is not a claim about the method in general.** One representation (pooled last encoder state), one
  decoder (base-size, copy-tuned, not translation-tuned), one corpus (mixed legal/literary, not culinary).
* **en → vi has no BLEU.** Only its geometry is measured. Generating Vietnamese from mapped vectors needs
  the vit5 decoder, which was out of the time box.
* **The decoder is a copy machine, by construction.** The mission specifies the LoRA'd flan-t5, and that
  adapter was trained to reproduce its input. Any cell in §3.2 where the decoder reads its own encoder's
  output therefore measures copying, not translating. A **translation-tuned** decoder is required before
  BLEU can judge the mapping at all — that is the single biggest design flaw of this run.
* **BLEU on 1,000 sentences cannot separate 0.141 from 0.240.** The diagnostic in §3.3 is what makes the
  claim, not the BLEU ordering. Report a bootstrap interval before quoting any of these numbers as a
  difference.
* **`flan_encoder` is not an upper bound.** An earlier reading of it as "what the decoder can do given the
  right vector" was wrong, because that run's output is 23.9 % Vietnamese (§3.3 item 2). The honest
  statement is: no run in this folder is a translation system, so no run establishes a ceiling.
* **Catastrophic forgetting is not measured.** The flan adapter was never evaluated on MTET; its
  instruction/translation behaviour after copy-tuning is unknown.
* **Vocabulary effect not isolated.** vit5 and flan tokenise Vietnamese differently; part of any loss may
  be tokenisation mismatch rather than the map.

## 5. Reproduce

```powershell
cd research-scale-100k
powershell -ExecutionPolicy Bypass -File setup.ps1
.\.venv\Scripts\python.exe -m src.run_all all      # build -> encode -> map -> bleu, ~25 min
.\.venv\Scripts\python.exe tools\execute_notebooks.py
```

Traceability note: the first "full" `bleu` call (10:05:59) wasted 5 min because the writer skipped files
that already existed, so it re-scored the 40-row smoke-test output; `bleu.py` was fixed to overwrite what
it regenerated, and the real run is the one that started at 10:14:31. The committed `gen_*.jsonl` are the
1,000-row ones (checked: 1,000 lines each).

## 6. Not done, in the order I would do it

0. **Fix the decoder first — otherwise every BLEU is uninterpretable.** The adapter used here was trained to
   copy. Either fine-tune flan-t5 (LoRA) on the *translation* task from the same 97k pairs, or use a
   translation-trained seq2seq. Only then does "mapped vs native encoder" mean anything.
1. **Per-token mapping** (`(T,768)` instead of a pooled vector, tokenisers aligned via `offset_mapping`).
   A loop like `"the a n the a"` is what a decoder does when it has one vector and no alignment.
2. **MLP map** (768 → 1024 → 768, GELU) to test whether the residual error is nonlinear. Cheap, and the
   ridge λ = 0 result suggests there is little left to gain — verify rather than assume.
3. **en → vi BLEU** through the vit5 decoder with mapped flan vectors.
4. **Domain split** of BLEU (legal vs literary vs food), because averages over this corpus hide a lot.
5. **Bootstrap CI** on every BLEU here (n = 1,000); 0.141 vs 0.240 is inside the noise band.
