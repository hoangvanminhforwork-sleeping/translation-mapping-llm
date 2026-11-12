# Handoff — research-prototype — 2026-11-12

## 0. Read first, in this order
1. `REPORT.md` — what was asked, what was delivered, what failed.
2. `results/copy_summary.csv` — the gate numbers.
3. `LEARNING_NOTE.md` — only if you will change the code.

## 1. Status per deliverable

| # | Deliverable | Status | Evidence | Notes |
|---|---|---|---|---|
| 1 | 4 checkpoints | DONE | `models\hf\hub\*`, 7.6 GB | TF/Flax duplicates intentionally not fetched |
| 2 | Model exploratory (1 nb / model) | DONE | `model-exploratory\0[1-4]_*.ipynb` | executed in place |
| 3 | Copy test, 2,000 / model | 2 of 4 DONE; `vit5-large` 1,960/2,000; `flan-t5-large` 0 | `results\raw\copy_*.jsonl` | GPU was reassigned to `finetune-and-lora` at the user's instruction |
| 4 | Perplexity (dish vs rest) | **not executed** (code smoke-tested) | `src\score.py`, `agent-work\smoke_ppl.py` | `python -m src.run_all ppl` |
| 5 | Zero-shot BLEU | **not executed** (code smoke-tested) | `src\score.py` | `python -m src.run_all translate` |
| 6 | Notebooks for §3 and §5 | DONE | `model-instruction-following\`, `test-token-generation\` | read the CSVs |

## 2. Frozen — do not touch
- `data\data\*.parquet` (input corpus, 4.4 GB)
- `results\raw\*.jsonl` (primary evidence; regenerating costs GPU hours)
- `models\hf\` (shared with `finetune-and-lora`, deleting forces a 7.6 GB download)

## 3. Environment as measured (2026-11-12, 21:00)
- venv `.venv` = Python 3.12.10, torch 2.7.1+cu128, transformers 4.57.6, CUDA true, RTX 3050 4 GB
- kernelspec `research-prototype` registered **inside** the venv
- HF cache `models\hf`; RAM 7.7 GB total, 1–2 GB free under load; disk D: ~50 GB free

## 4. Open questions for the user
- Is `flan-t5-large` still wanted? It costs ~20 min of GPU for one row in the table.
  Default if no answer: **skip**, because the fine-tune mission needed the GPU.

## 5. Continue here
```powershell
.\.venv\Scripts\python.exe -m src.run_all copy --models flan-t5-large   # resumes/appends
.\.venv\Scripts\python.exe -m src.run_all ppl    --models flan-t5-large
.\.venv\Scripts\python.exe -m src.run_all translate --models flan-t5-large
.\.venv\Scripts\python.exe ".\tools\execute_notebooks.py"              # after results exist
```

If you change any metric code, re-read `LEARNING_NOTE.md §4.4` first: the T5 shift bug silently inflates
every perplexity by one token of context, and it is invisible unless you print one sentence.
