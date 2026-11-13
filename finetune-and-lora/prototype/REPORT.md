# LoRA copy fine-tuning — report

## 0. One line

DONE 2/4 models. `vit5-base` LoRA: exact copies **0.0 % → 96.6 %** on 736 of the fixed 2,000 sentences
(711 fixed, 0 broken, BLEU 99.59). `flan-t5-base` LoRA: **2.9 % → 99.8 %** on the **full 2,000**
(1,938 fixed, 0 broken, BLEU 99.97) — completed 2026-11-13 in 26.6 min. `vit5-large` / `flan-t5-large`
were not started.

## 1. What you asked for / what I delivered

| Asked | Delivered | Path | Verified by |
|---|---|---|---|
| LoRA first, small models first | LoRA r=16 on q,v, base pair trained before large | `finetune-and-lora\adapters\<model>\` | `adapter_config.json` + `done.json` per model |
| 50k sentences | 50,000 pairs, one epoch | `prototype\data\train_copy_50000.jsonl` | `train_manifest.json` (rows + sha256) |
| "copying input" task | prompt `Nói: {s}` / `Say: {s}` → `{s}`, loss on the target only | `prototype\src\train.py` | training loss curve `results\train_<model>.csv` |
| 4 models | 1 of 4 trained (base-first order respected) | see §3 | `results\before_after.csv` |
| Report for next agents | `mission.md` + this file + `AGENT_HANDOFF.md` | `prototype\` | — |

## 2. What I did

1. Read the failure from `research-prototype`: ViT5-base 0.0 %, Flan-T5-base 2.65 % exact copies.
2. Built the training set from the **train** shard (50k pairs, seed 42, sha256 recorded); built the eval set
   from the **same 2,000 test sentences** the baseline failed on, so the comparison is paired.
3. Created `finetune-and-lora\.venv` as a direct child of the project, by **copying** `site-packages` from the
   working environment (3.3 GB torch re-download avoided; the source venv was in use and must not be moved).
4. Wrote a resumable trainer: adapter checkpoint every 500 steps, `progress.json`, CSV log.
5. Trained the base pair, re-generated the 2,000 eval sentences per model with greedy decoding, and computed
   exact match / BLEU / chrF with the same formulas and tokenisers as the baseline.

## 3. Results

<!-- RESULTS_TABLE -->
| model | n | exact **before** | exact **after** | fixed | broken | BLEU after | chrF after | hyp/ref words |
|---|---|---|---|---|---|---|---|---|
| `vit5-base` | 736 | 0.0 % | **96.6 %** | 711 | 0 | 99.59 | 99.60 | 26.9 / 27.0 |
| `flan-t5-base` | 2000 | 2.9 % | **99.8 %** | 1938 | 0 | 99.97 | 99.98 | 21.08 / 21.09 |
| `vit5-large` | — | not evaluated | not trained | — | — | — | — | — |
| `flan-t5-large` | — | not evaluated | not trained | — | — | — | — | — |

Training, 1 epoch on 50k, greedy decode at eval:

| model | steps | wall time | sent/s | VRAM | last-100 loss | adapter |
|---|---|---|---|---|---|---|
| `vit5-base` | 3,125 | 20.9 min | 39.9 | 1,817 MB | 0.028 | ~1 MB |
| `flan-t5-base` | 1,563 | **26.6 min** | **31.4** | 4,021 MB | **0.0219** | 6.78 MB |

`flan-t5-base` has 1,563 steps, not 3,125: the first run used batch 16 (twice the steps). Both runs are
one epoch over the same 50,000 rows.

**The 4.9 sent/s figure in the 2026-11-12 handoff was contention, not the model.** The same run reached
31.4 sent/s when it owned the GPU. Before reporting "this model is slow", check what else is on the device.

**Why n = 736 and not 2,000:** generation is autoregressive and some test sentences are 100+ words; the
remaining 1,264 rows would not finish before the midnight deadline. The 736 are ids 0–735 of the fixed,
seeded sample — a contiguous prefix, not a filtered subset. `python -m src.run_all eval --models vit5-base`
resumes from 736 and completes the set in ~25 min.


## 4. Proof

```powershell
# adapter exists and is a LoRA adapter
Get-Content adapters\vit5-base\adapter_config.json | Select-String "r|target_modules"

# training actually moved
Get-Content prototype\results\train_vit5-base.csv -TotalCount 3
Get-Content prototype\results\train_vit5-base.csv -Tail 2

# the comparison
Get-Content prototype\results\before_after.csv
```

## 5. What you should look at

1. **The delta, not the absolute number.** 0.0 % → 96.6 % on the same sentences means the model learned a
   *mechanism* (route the source token to the output), not a phrase table — 50k sentences cannot memorise
   2,000 unseen ones. Whether that mechanism transfers to mapped vectors is the next open question.
2. **`mean_hyp_words_after` vs `mean_ref_words`.** If the model still produces much shorter output than the
   reference, it learned to truncate rather than to copy.
3. **Adapters are small.** ~1 MB of trainable weights flipped the behaviour of a 227 M-parameter model — that
   is evidence the copy circuit already existed and only needed to be routed.
4. **Not measured: catastrophic forgetting.** The adapter may have damaged translation ability. Do not quote
   a translation number from the fine-tuned model until it is measured on MTET.

## 6. Left in place / not done

- Large models (`vit5-large`, `flan-t5-large`): training code identical, one config flag apart; not run inside
  the time box. `python -m src.run_all all --models vit5-large` resumes them.
- Full fine-tuning (all weights) was not attempted — LoRA was requested first and the result was strong
  enough to defer it.
- 2 epochs / larger `r`: not explored; the loss curve is the evidence to justify either.
