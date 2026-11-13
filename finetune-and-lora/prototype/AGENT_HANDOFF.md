# Handoff — finetune-and-lora — 2026-11-12

## 0. Read first, in this order
1. `mission.md` — objective, data provenance, method, fences.
2. `REPORT.md` — what was delivered and what the numbers mean.
3. `results/before_after.csv` — the one table that matters.

## 1. Status per deliverable

| # | Deliverable | Status | Evidence | Notes |
|---|---|---|---|---|
| 1 | 50k train set | DONE | `data/train_copy_50000.jsonl` + `train_manifest.json` | sha256 recorded; file itself gitignored |
| 2 | Fixed eval set (same 2,000 as baseline) | DONE | `data/eval_sources.jsonl` | 2,000 vi + 2,000 en = 4,000 rows |
| 3 | LoRA harness | DONE | `src/train.py` | resumable, 500-step adapter checkpoints |
| 4 | `vit5-base` adapter | **DONE** | `../adapters/vit5-base/` | 0.0 % → 96.6 % exact on n=736 |
| 5 | `flan-t5-base` adapter | **DONE (2026-11-13)** | `../adapters/flan-t5-base/` + `results/before_after.csv` | 2.9 % → **99.8 %** exact on the **full 2,000** (1,938 fixed, 0 broken, BLEU 99.97), 26.6 min, `done.json` written |
| 6 | `vit5-large` adapter | NOT RUN | — | ~45 min |
| 7 | `flan-t5-large` adapter | NOT RUN | — | ~45 min |

Also done on 2026-11-13: `src/config.py` `SAVE_EVERY` 500 → 250 (a time-boxed run keeps a usable
adapter); the `vit5-base` eval is still 736/2,000 and can be completed with the same command.

## 2. Frozen — do not touch
- `research-prototype/results/raw/copy_*.jsonl` — the eval set is copied from these; deleting them breaks
  `src/data.py build_eval` until they exist again.
- `research-prototype/models/hf/` — shared checkpoint cache (7.6 GB). `HF_HOME` points here on purpose.
- `data/eval_sources.jsonl` — regenerate only with `--force`, and only if you accept that "before/after"
  changes meaning.

## 3. Environment as measured
- `finetune-and-lora/.venv` — Python 3.12.10, torch 2.7.1+cu128, transformers 4.57.6, peft 0.21.2, CUDA true.
- The venv was seeded by **copying** `site-packages` from `research-prototype/.venv` (the source was in use;
  moving would have broken it). `requirements.txt` is the from-zero path.
- Training: LoRA r=16 q,v; 1.77 M trainable / 227.7 M (0.78 %); ~1,800 MB VRAM; ~37–40 sent/s at batch 16.

## 4. Open questions for the user
- Large models now, or move on to the mapping experiment? Default: **large models**, because the paper's
  claim is about a pair, and the base pair already supports it.
- 1 epoch or 3? Default: **1**, the loss curve flattens early (see `results/train_*.csv`).

## 5. Continue here

```powershell
cd finetune-and-lora\prototype
..\.venv\Scripts\python.exe -m src.run_all all --models vit5-large flan-t5-large
```

Then, before any translation claim:
```powershell
# catastrophic forgetting is NOT measured yet: evaluate the adapter on MTET, not on the copy task
```

If a run is interrupted: do **not** delete `adapters/<model>/progress.json` unless you want to restart from
step 0. Training resumes from that step exactly.
