# research-scale-100k — linear mapping between vit5-base and flan-t5-base

100k EN–VI pairs, two 768-d representation spaces, one linear map per direction, and a BLEU gate
that asks whether the flan-t5 decoder can generate from a *mapped* vector.

**Read in this order**: `REPORT.md` (what was delivered, what the numbers mean, what failed) →
`LEARNING_NOTE.md` (why each choice was made) → `01_mapping_100k.ipynb` (executed, real outputs).

## Layout

```
src/
  build_pairs.py   100k pairs, deduped, vi/en side detected, sha256 in data/pairs_manifest.json
  encode.py        two 100k x 768 matrices (vit5 encoder on vi, flan+LoRA encoder on en)
  mapping.py       ridge sweep on val, metrics + controls, saves data/map_*.npz
  bleu.py          3 generation runs on the 1,000-sentence test split
  run_all.py       build -> encode -> map -> bleu
tools/
  make_notebook.py      regenerates the notebook (never touches results)
  execute_notebooks.py  runs it in place with the research-scale-100k kernel
data/       pairs_100k.parquet (gitignored) + pairs_manifest.json (committed) + enc_*.npy + map_*.npz
results/    ridge_sweep.csv, mapping_metrics.csv, mapping_summary.json, bleu_vi_en.json, gen_*.jsonl
```

## Reproduce from zero

```powershell
cd research-scale-100k
powershell -ExecutionPolicy Bypass -File setup.ps1     # venv + kernel + HF_HOME check
.\.venv\Scripts\python.exe -m src.run_all all          # ~25 min on an RTX 3050 4 GB
.\.venv\Scripts\python.exe tools\execute_notebooks.py  # executed notebook in place
```

The `.venv` in this folder was **copied** from `../research-prototype/.venv` (6.24 GB, same pins, plus
`peft`, `pypdf`, `sacrebleu`) to avoid a 20-minute reinstall; `setup.ps1` is the from-zero path.

Prerequisites that must already exist:

| What | Where | How it was made |
|---|---|---|
| `data/data/*.parquet` (EN–VI corpus, 4.4 GB) | `../../data/data/` | downloaded earlier |
| vit5-base / flan-t5-base checkpoints | `../research-prototype/models/hf/` | `HF_HOME` points here |
| flan-t5-base LoRA adapter | `../finetune-and-lora/adapters/flan-t5-base/` | `finetune-and-lora/prototype`: `python -m src.run_all train --models flan-t5-base` |

If the adapter is missing, `encode.py` and `bleu.py` stop with a message rather than silently using
the base model — a mapping onto the *base* flan space would be a different experiment.

## Idempotency

Every stage checks its own output and skips. `--force` rewrites that stage only. Deleting
`data/pairs_100k.parquet` is safe (rebuilt in ~30 s with the same seed and the same sha256);
deleting `results/gen_*.jsonl` costs one GPU generation run.
