# autonomous-research-experiment — one night, two frozen models, one question

> *Is **latent** agreement a usable reference-free quality signal for MT — and does it beat the
> **string** version of the same idea?*

Everything here was produced on 2026-11-19 22:20 → 2026-11-20 06:30 on a laptop (i5-12500H, 7.7 GB
RAM, RTX 3050 4 GB, Windows) with **two frozen 77 M models** and **zero fine-tuning as a claim**.

Read in this order: `HYPOTHESES.md` (pre-registered) → `REPORT.md` (results) → `PROGRESS.md` (run log,
including my own mistakes) → `HANDOFF.md` (what to do next) → `LEARNING_NOTE.md` (why, for a human).

## Layout

```
src/
  config.py        paths, the two model ids, split sizes, seed              (single source of truth)
  build_split.py   3.46M pairs -> 80/10/10 by md5 rank (test 10k, val 10k)
  gen.py           candidate pools from the frozen translator (greedy | beam-K | sample-K | mixed)
  features.py      reference-free features: back-translation, string sims, latent agreement
  arms.py          pool -> one selection per arm (beam top-1, MBR-chrF, rt_str, rt_lat, len-ratio)
  comet_labels.py  per-candidate COMET (teacher for the oracle + the tiny ranker)
  ranker.py        tiny MLP rankers (string-only vs latent), QE diagnostics, oracle arm
  score.py         BLEU / chrF / COMET + paired bootstrap over sentences
  tables.py        markdown tables straight from the JSON (no hand-typed numbers)
  lora_control.py  LABELLED CONTROL: 4 M-param LoRA (not part of the claim)
data/              split.parquet (built, deterministic), lora_train.parquet (control only)
results/           pools (.jsonl), features (.parquet), COMET labels, selections, metrics
logs/              one log per job
models/hf/         project-local HF cache (keeps C: untouched)
queue*.ps1         the exact job chains that were run, in order
```

## Reproduce from nothing

```powershell
cd autonomous-research-experiment
$py = "..\research-scale-100k\.venv\Scripts\python.exe"   # torch 2.7.1+cu128, transformers 4.57.6

# 1. split (idempotent; --force to rebuild)
& $py -m src.build_split

# 2. candidate pools (chain 1)
& $py -m src.gen --split test  --n 10000 --mode greedy --batch 24 --out pool_test_greedy.jsonl
& $py -m src.gen --split test  --n 10000 --mode beam --k 8 --batch 8 --out pool_test_beam8.jsonl
& $py -m src.gen --split train --n  5000 --mode beam --k 8 --batch 8 --out pool_train_beam8.jsonl

# 3. reference-free features + COMET teacher labels (chain 2)
./queue2.ps1

# 4. selections + tiny rankers + oracle
& $py -m src.arms --pool pool_test_beam8.jsonl --feat feat_test_beam8.parquet
& $py -m src.ranker

# 5. scores (BLEU/chrF on 10k, COMET on the 2k core subset)
& $py -m src.score --hyp results/sel_A0_greedy.jsonl results/sel_A1_beam.jsonl ... --out metrics_10k.json

# 6. control (optional, 1 h GPU)
./queue3.ps1
```

Every script skips work whose output already exists (delete the file, or pass `--force`, to redo) and
writes to `.tmp` then renames, so an interrupted run can never look finished.

## How to verify a claim in 5 minutes

1. `python -m src.tables` — regenerates every table in `REPORT.md` from `results/*.json`.
2. `results/sel_*.jsonl` — one line per test sentence per arm: `vi`, `ref`, the chosen `hyp`, and
   which pool rank it came from. Any table number can be recomputed from these three files with
   sacrebleu/comet in a few lines.
3. `results/rank_diag.json` — Spearman / top-1 accuracy / seed spread for every utility.
4. `PROGRESS.md` — every number that changed, every mistake, with timestamps.

## Environment notes (why the code looks like this)

- `dtype=` (transformers 4.57.6), not `torch_dtype=`; `compute_transition_scores` OOMs on 4 GB →
  `sequences_scores`/`getattr`; greedy `generate` has no `beam_indices`; `pyarrow` `filters=` on a
  single file fails in this version → filter in Python.
- One GPU job at a time (two jobs on a 4 GB card do not OOM, they *stall*, which is worse).
- RAM 7.7 GB is the binding constraint: streams/row groups everywhere, encoder states freed per chunk.
