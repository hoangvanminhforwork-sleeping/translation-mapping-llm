# Handoff — research-phase01-ver02 — 2026-11-15

## 0. Read first, in this order

1. `paper.md` — problem definition, method, results, discussion.
2. `results/metrics_vi_en.json` — the generation table with bootstrap intervals.
3. `results/geometry_vi_en.csv` — representation geometry with identity/mean controls.
4. `results/lgbm_metrics.json` + `results/geometry_vi_en.csv` — the LightGBM reversal (in-sample win,
   held-out loss). This is the most interesting result in the folder.
5. `LEARNING_NOTE.md` — why the pipeline looks like this, and the six traps.
6. `01_results.ipynb` — the same numbers as tables, executed.

## 1. State — every stage finished

| # | Deliverable | Evidence |
|---|---|---|
| 1 | corpus, per-shard dedup, prior rows excluded | `data/pairs_manifest.json` (6,668,910 rows, sha `6f175d40…`) |
| 2 | coreset + shard choice | `results/coreset.{csv,json}`, `data/chosen_shard.json` |
| 3 | ridge, both directions | `data/map_vi_en.npz`, `data/map_en_vi.npz`, `results/ridge_sweep.csv` |
| 4 | 512 LightGBM boosters | `data/lgbm/vi_en.pkl.gz` (42 MB), `results/lgbm_metrics.json` |
| 5 | copy gate, both models | `results/copy_gate_{en,vi}*.json`, `adapters/copy-t5-small{,-vi}/` |
| 6 | translation adapters | `adapters/trans-t5-small-vi-en/`, `adapters/trans-t5-small-en-vi/` |
| 7 | generation, 5 channels | `results/gen_vi_en_{mapped,lgbm,native,identity,copy}.jsonl` |
| 8 | generation, 3 channels | `results/gen_en_vi_{mapped,native,identity}.jsonl` |
| 9 | BLEU/chrF/COMET + bootstrap | `results/metrics_vi_en.json`, `results/metrics_en_vi.json` |
| 10 | geometry | `results/geometry_{vi_en,en_vi}.csv` |
| 11 | neural mining | `results/mining.json`, `results/mining.md` |
| 12 | paper + mining report | `paper.md`, `neuralMining.md` |

Headline (vi→en, 3,000 sentences): mapped **2.84 BLEU / 19.96 chrF / NLL 3.20** vs identity
0.03 / 6.09 / 4.62 vs native 4.17 / 15.35 / 3.37 vs LightGBM 2.32 / 16.11 / 3.38.
Copy-readout channel: 0.37 BLEU, 95 % degenerate (the naive vector-injection hypothesis fails).

## 2. Environment as measured

Shared venv `../research-scale-100k/.venv` (not copied): torch 2.7.1+cu128, transformers 4.57.6,
peft 0.21.2, **lightgbm 4.6.0 (added this session)**, numpy 2.5.4, sacrebleu 2.5.1.
Models live in `../research-prototype/models/hf/local/{t5-small,t5-small-vietnamese}` — downloaded
with `local_dir` because this `huggingface_hub` cannot create symlinks without Developer Mode.

Measured throughput on the RTX 3050 4 GB / 7.7 GB RAM laptop:
* aligned token moments: **~1,700 pairs/s** steady state (563 pairs/s in the first 4 k-pair smoke test
  — that number includes warm-up and is misleading)
* pooled sentence embeddings: **~2,540 sent/s**
* translation LoRA training: **~117 pairs/s** (60 k pairs in 8.5 min)
* copy LoRA training: **~29 sent/s** (slower: shorter sequences, more Python overhead per step)
* greedy generation, batch 64, max 48 new tokens: **~17 sent/s**; LightGBM channel **~4.2 sent/s**

## 3. Frozen — do not touch

* `data/pairs.parquet` (370 MB, gitignored) — ids are `md5(vi)`-ordered; changing them invalidates the
  split and every number.
* `data/stats/00008_{train,val}.npz` — the only way to refit the map without 8 min of GPU.
* `data/chosen.parquet` — the materialised shard (479,394 rows) every stage reads.
* `results/gen_*.jsonl` — the primary generation evidence.
* `data/map_*.npz` — the fitted artefact (512×512).

## 4. Three results that contradict their own framing (do not "fix" by re-running)

1. **LightGBM wins in-sample and loses out-of-sample.** +0.024 R² on tokens from the same sentences,
   −0.004 token cosine and −0.033 retrieval@1 on 5,000 held-out sentences. The nonlinearity is
   sentence-specific. This is reported as the answer to "why linear wins", not as a bug.
2. **`native` beats `mapped` on BLEU but loses on chrF and reference NLL** (4.17 vs 2.84 BLEU, 15.35
   vs 19.96 chrF, 3.37 vs 3.20 NLL). `native` degenerates into token loops 31.6 % of the time
   (`the sex of the sex of …`). BLEU is precision-oriented and rewards it; the other metrics do not.
3. **The copy gate passes for English and fails for Vietnamese.** 85.9 % exact from the English
   model's own encoder after 200 LoRA steps; 13.5 % from the Vietnamese model after 15 k examples
   (copy loss 0.39 vs 0.02). The Vietnamese decoder learns *when to stop* (22.7 vs 22.4 words) but
   not *what to emit*. That is why en→vi has no `copy` channel.
4. **The copy readout fails even where the gate passes.** 85.9 % exact copy from the model's own
   encoder, 0.37 BLEU / 95 % degenerate when fed mapped vectors. The readout is
   distribution-specific; the naive "predict the vector then copy it" story is falsified.

## 5. Highest-value next steps, in order

1. **Train the translation adapter longer.** The loss is still falling at step 1,876 (1 epoch / 60 k
   pairs). `python -m src.train_decoder --direction vi_en --pairs 200000` resumes the adapter.
   This is the binding constraint on absolute BLEU, not the map: `native` (no transfer needed) is
   only 4.17 BLEU.
2. **Word-level alignment** instead of character-span overlap. ~14 % of source tokens fall back to
   the nearest target token, injecting noise into numbers and names.
3. **Two adapters instead of one mixed adapter**, to remove the shared-capacity confound between
   `native` and `mapped`.
4. **Fit the map on all 13 eligible shards** (the coreset says they are exchangeable, so this is
   safe) — 9.28 M → ~65 M token pairs.
5. Add a **small end-to-end MT baseline** trained on the same 60 k pairs, which is the missing
   "is this architecture worth it?" comparison.

## 6. Resume commands

```powershell
cd research-phase01-ver02
$py = "..\research-scale-100k\.venv\Scripts\python.exe"

& $py -m src.build_pairs                                  # idempotent, checks sha256
& $py -m src.embed sentence_sample --per-file 50000        # idempotent
& $py -m src.coreset
& $py -m src.embed moments --file <chosen shard>
& $py -m src.embed subset --file <chosen shard>
& $py -m src.fit_ridge
& $py -m src.fit_lgbm --tokens 100000 --threads 8
& $py -m src.copy_gate train --side en --pairs 50000
& $py -m src.train_decoder --direction vi_en --pairs 60000
& $py -m src.generate --direction vi_en --n 3000 --batch 64 --max-new 48
& $py -m src.score --direction vi_en --comet
& $py -m src.eval_geometry --direction vi_en --n 5000
& $py -m src.mine; & $py -m src.figures
& $py tools\make_notebook.py; & $py tools\execute_notebook.py
```

Every stage checks its own output and skips; `--force` redoes that stage only.

## 7. Artefacts and storage

Committed to git: `data/map_*.npz` (~1 MB each), `results/*` (metrics, geometry, mining, figures),
`adapters/` (~4 MB each), notebooks, `paper.md`, `neuralMining.md`.
Pushed as a GitHub release asset and then deleted locally: `data/sent_sample_fp16.npy` (716 MB),
`data/lgbm/vi_en.pkl.gz` (42 MB). See `ARTIFACTS.md` for the exact names and how to re-create them.
