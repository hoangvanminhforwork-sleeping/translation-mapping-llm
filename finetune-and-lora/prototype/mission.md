# Mission — copy fine-tuning (LoRA) for ViT5 / Flan-T5

**Objective.** Produce LoRA adapters at `finetune-and-lora\adapters\<model>\` such that each model, given
`Nói: {s}` (Vietnamese) or `Say: {s}` (English), reproduces `{s}` exactly, so that the vector-mapping
methodology has a read-out stage that actually obeys its prompt.

## Why this mission exists

`research-prototype` measured the gate and it failed:

| model | exact copies / 2000 | copy BLEU |
|---|---|---|
| `vit5-base` | **0.0 %** | 13.25 |
| `flan-t5-base` | **2.65 %** | 8.40 |

ViT5 emits sentinel-like garbage (`*+__Z…`, `nói: nói: nói:`); Flan-T5 paraphrases or answers instead of
repeating. A mapped vector cannot be read out by a model that will not even repeat its own input, so
fine-tuning on the copy task is a precondition, not an optimisation.

## Deliverables

| # | Artifact | Path | Format | Count |
|---|---|---|---|---|
| 1 | LoRA adapters | `finetune-and-lora\adapters\<model>\adapter_model.safetensors` | PEFT adapter | 4 (base pair first) |
| 2 | Training logs | `finetune-and-lora\prototype\results\train_<model>.csv` | CSV, one row / 50 steps | 4 |
| 3 | Before/after table | `finetune-and-lora\prototype\results\before_after.csv` | CSV | 1 |
| 4 | Generated hypotheses | `finetune-and-lora\prototype\results\raw\ft_<model>.jsonl` | JSONL, 2,000 rows / model | 4 |
| 5 | Report | `finetune-and-lora\prototype\REPORT.md` | markdown | 1 |

**Not a deliverable:** full fine-tuning of the base weights, a new tokenizer, translation quality,
or any change to `research-prototype\` (read-only here).

## Data — exact provenance

| | Path | Rows | Notes |
|---|---|---|---|
| Train | `data\train_copy_50000.jsonl` | **50,000 pairs** | sampled seed 42 from `data\data\train-00000-of-00014.parquet`; sha recorded in `data\train_manifest.json` |
| Eval | `data\eval_sources.jsonl` | **4,000 rows** (2,000 vi + 2,000 en) | copied from `research-prototype\results\raw\copy_*.jsonl`: the **same sentences the baseline failed on**, so before/after is paired |

Train and eval are disjoint by construction (train shard vs test split).

Each pair carries `vi` and `en`. The Vietnamese model trains on the `vi` side, the English model on the
`en` side, of the **same** 50,000 pairs. Prompt = `prefix + sentence`; target = sentence.

## Method and the reason for each choice

| Choice | Why |
|---|---|
| **LoRA first** (not full fine-tune) | ~1 % of parameters, ~1 MB on disk; if the copy circuit is reachable at all, LoRA reaches it. A negative LoRA result is strong evidence. |
| `r=16, alpha=32, dropout=0.05`, targets `q`,`v` | copying is an attention-routing behaviour; q,v are the cheapest sufficient pair |
| `lr=1e-3`, cosine decay, 100-step warmup | LoRA tolerates a high lr; cosine prevents late oscillation |
| `1 epoch` of 50k | the time box; more epochs are the documented next step, not a silent default |
| `max_len=64`, dynamic padding | 95 % of sentences fit; padding to a fixed 128 doubles the step time for nothing |
| `bfloat16` base + adapter | fp16 is numerically unstable for T5 heads |
| greedy decode at eval | matches the baseline exactly (the comparison must differ only by the adapter) |
| batch 16 (base) / 4 (large) | fits 4 GB VRAM with room for activations |

## Fences

- **May read:** `research-prototype\results\raw\` (baseline), `data\data\*.parquet`, `paperOverview.md`.
- **May write:** `finetune-and-lora\` only. `research-prototype\` is frozen except for the resumed stages
  listed below.
- **Must not touch:** `data\data\` (input corpus), `research-prototype\results\` (baseline evidence),
  `.env`, any previous adapter.
- **Never** re-download the checkpoints: `HF_HOME` points at `research-prototype\models\hf` (7.6 GB already
  present). Override with an env var only if that cache disappears.

## How to run

```powershell
cd finetune-and-lora\prototype
..\.venv\Scripts\python.exe -m src.run_all data
..\.venv\Scripts\python.exe -m src.run_all train --models vit5-base
..\.venv\Scripts\python.exe -m src.run_all eval  --models vit5-base
..\.venv\Scripts\python.exe -m src.run_all compare
```

Resumable: adapters save every 500 steps with `progress.json`; re-running continues from the saved step.
`--force` restarts a model from scratch.

## Status (updated at the end of each session)

See `AGENT_HANDOFF.md` in this folder for the live state. As of the first session (2026-11-12, midnight
deadline):

- [x] 50k train set built, sha recorded (`data/train_manifest.json`)
- [x] fixed 4,000-row eval set copied from the baseline
- [x] LoRA training harness (resumable, logs, adapter checkpoints)
- [x] `vit5-base` adapter — trained 1 epoch / 50k, **6.78 MB, 20.9 min, final loss 0.028**
- [x] `vit5-base` before/after eval — see `results/before_after.csv`
- [ ] `flan-t5-base` adapter — training started, **killed by the deadline** at step ~150 (loss 0.038);
      no adapter was checkpointed (saves happen every 500 steps). Re-run with
      `python -m src.run_all all --models flan-t5-base`
- [ ] `vit5-large` adapter — not run
- [ ] `flan-t5-large` adapter — not run

## Known traps (learned the hard way in `research-prototype`)

1. A PowerShell text pipeline destroys Vietnamese diacritics. Use Python for text.
2. `Set-Content -Encoding UTF8` writes a BOM; `json.loads` rejects it.
3. T5 shifts labels internally — do not shift twice when computing per-token loss.
4. The MTET test split has **six** prompt templates, not four; matching four swaps 17 % of rows.
5. `python -m jupyter` needs console scripts; prefer `python -m nbconvert`.
6. Moving `Lib\site-packages` between venvs gives imports but not entry points.
