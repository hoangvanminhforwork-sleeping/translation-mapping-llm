# research-phase01-ver02 — cross-model linear mapping with 60 M models

The phase-01 claim, re-tested on a pair of *small* models where generation is cheap enough to
measure on 5,000 sentences instead of 2,000:

> A single 512×512 linear map can carry a Vietnamese sentence from `t5-small-vietnamese`'s
> representation space into `t5-small`'s, well enough that the decoder renders the English.

New in this version, relative to phase 01:

| | phase 01 | ver02 |
|---|---|---|
| models | vit5-base 220 M / flan-t5-base 220 M (768-d) | t5-small-vietnamese / t5-small (60 M, 512-d) |
| corpus | global dedup, 3.46 M pairs | per-shard dedup, 6.67 M pairs, tagged by shard |
| shard choice | all data | coreset-selected single shard (mission §3.3) |
| split | 2000 / 2000 / rest | 80 / 10 / 10 |
| second mapper | — | 512 LightGBM regressions (nonlinearity probe) |
| readout | translation adapter only | copy adapter (vector readout) **and** translation adapter |
| generation test | 2,000 | 5,000 × 5 channels × 2 directions |

## Read in this order

1. `paper.md` — problem definition, method, results, discussion.
2. `neuralMining.md` — why a linear map keeps winning (M1–M4).
3. `results/metrics_vi_en.json` — the generation table with bootstrap intervals.
4. `results/geometry_vi_en.csv` — representation geometry with identity/mean controls.
5. `01_results.ipynb` — the same numbers as tables, executed.
6. `AGENT_HANDOFF.md` — state for the next agent.
7. `LEARNING_NOTE.md` — why the pipeline looks like this.

## Layout

```
src/
  config.py          paths, models, split rule, hyper-parameters
  build_pairs.py     6.67M pairs, per-shard dedup, md5-ordered ids, prior rows excluded
  tokutil.py         T5 fast tokenizers from tokenizer.json + cross-tokeniser alignment
  tokenbatch.py      GPU alignment + length-cropped batches + tokeniser prefetch
  embed.py           streaming moments (A,B,C) + sentence embeddings for the coreset
  coreset.py         KMeans K=4 (and K=32), per-shard representativeness/diversity/outliers
  fit_ridge.py       512 linear regressions as one 512x512 solve + per-dim R^2
  fit_lgbm.py        512 LightGBM boosters, threaded, compared with ridge on identical tokens
  copy_gate.py       identity copy check + decoder-only copy LoRA
  train_decoder.py   translation LoRA (native + mapped channels), both directions
  generate.py        5 channels x 5,000 test pairs, with per-sentence decoder NLL
  score.py           BLEU, chrF, COMET, paired bootstrap, structural diagnostics
  eval_geometry.py   token/sentence cosine, retrieval@1, identity + mean controls
  mine.py            M1-M4 neural-mining measurements
  figures.py         every figure in the paper
data/                pairs.parquet, stats/, map_*.npz, lgbm/, sent_sample_*
results/             every number, committed
adapters/            copy + translation LoRA adapters
```

## Reproduce

```powershell
cd research-phase01-ver02
$py = "..\research-scale-100k\.venv\Scripts\python.exe"   # shared venv, not copied

& $py -m src.build_pairs                              # ~5 min, CPU+DuckDB
& $py -m src.embed sentence_sample --per-file 50000    # ~5 min, GPU (coreset input)
& $py -m src.coreset                                  # ~4 min, CPU -> data/chosen_shard.json
& $py -m src.embed moments --file <chosen shard>       # ~8 min, GPU
& $py -m src.embed subset  --file <chosen shard>       # ~1 min, writes data/chosen.parquet
& $py -m src.fit_ridge                                # seconds, CPU
& $py -m src.fit_lgbm --tokens 100000 --threads 8     # ~38 min, CPU
& $py -m src.copy_gate eval  --side en                # zero-shot copy gate
& $py -m src.copy_gate train --side en --pairs 50000  # only if the gate fails
& $py -m src.copy_gate eval  --side en --adapter
& $py -m src.train_decoder --direction vi_en --pairs 60000   # ~9 min, GPU
& $py -m src.generate --direction vi_en --n 3000 --batch 64 --max-new 48
& $py -m src.score --direction vi_en
& $py -m src.eval_geometry --direction vi_en --n 5000
& $py -m src.mine; & $py -m src.figures
& $py tools\make_notebook.py; & $py tools\execute_notebook.py
```

Every stage checks its own output and skips; `--force` redoes that stage only.

## Prerequisites

| What | Where | Notes |
|---|---|---|
| EN↔VI corpus, 8.33 M rows | `../data/data/train-*.parquet` | 4.4 GB, already present |
| t5-small, t5-small-vietnamese | `../research-prototype/models/hf/local/` | downloaded with `local_dir` (no symlink privilege) |
| Python env | `../research-scale-100k/.venv` | torch 2.7.1+cu128, transformers 4.57.6, peft 0.21.2, lightgbm |
| COMET | HF cache | `Unbabel/wmt22-comet-da`, already present |

**Two environment traps, both hit and solved here:**
1. `huggingface_hub` cannot create symlinks without Developer Mode (`WinError 1314`), so
   `snapshot_download` fails in the shared cache. Use `local_dir=`.
2. The venv's `sentencepiece` wants `protobuf>=5.27` but has 4.25.9, so any *slow* tokenizer
   crashes. Both checkpoints ship `tokenizer.json`, so `T5TokenizerFast(tokenizer_file=...)` avoids
   sentencepiece entirely — and it is the only way to get character offsets from the Vietnamese
   checkpoint anyway.

## Hardware reality

RTX 3050 4 GB / 7.7 GB RAM. Both 60 M encoders together use 250 MB of VRAM and run at 563 pairs/s
(aligned) or 2,540 sentences/s (pooled). One heavy job at a time; LightGBM is CPU-only and can run
while a GPU job is idle but not while a GPU job is also holding RAM.
