# Handoff — research-scale-100k — 2026-11-13 (10:30 VN)

## 0. Read first, in this order

1. `REPORT.md` — delivered, numbers, what failed, what it does *not* show.
2. `results/bleu_vi_en.json` and `results/mapping_metrics.csv` — the two tables that matter.
3. `LEARNING_NOTE.md` §2–§5 — only if you will change the mapping or the pooling.

## 1. Status

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| 1 | 100k pairs | DONE (99,743) | `data/pairs_manifest.json` (sha256 `4f1051e1…`) |
| 2 | two 100k×768 matrices | DONE | `data/enc_vit5_vi.npy`, `data/enc_flan_en.npy` (+ `enc_flan_vi_test.npy` control) |
| 3 | linear map, both directions | DONE | `data/map_vi_en.npz`, `data/map_en_vi.npz`; cos 0.853/0.858, R@1 0.514/0.506 |
| 4 | BLEU (vi→en, n=1,000) | DONE | `results/bleu_vi_en.json`: mapped 0.141 / flan_encoder 3.115 / zero-shot 0.240 |
| 5 | executed notebook | DONE (or see §5 if missing) | `01_mapping_100k.ipynb` |
| 6 | en→vi BLEU | NOT DONE | geometry only; needs the vit5 decoder |

## 2. Environment as measured

* `.venv` copied from `research-prototype/.venv` (6.24 GB), plus `peft==0.21.2`, `pypdf`, `sacrebleu`.
  Kernel `research-scale-100k` is registered (user-level kernelspec).
* `HF_HOME` = `../research-prototype/models/hf` (shared 7.6 GB cache).
* GPU: RTX 3050 4 GB. Encoding ran at ~800 seq/s; generation at ~10 sent/s.
* RAM 7.7 GB total, 1.1 GB free at session start — one heavy process at a time.

## 3. Frozen — do not touch

* `data/pairs_100k.parquet` — 10.9 MB, gitignored, rebuilt in 25 s from `../../data/data/*.parquet`
  with seed 42; the sha256 in the manifest must match or every number changes.
* `data/enc_*.npy` — 292 MB each, gitignored, 2 min to rebuild.
* `results/gen_*.jsonl` — 1,000 rows each; the primary generation evidence.
* `../finetune-and-lora/adapters/flan-t5-base/` — the map is fitted into *that* space.

## 4. Two traps found the hard way

1. `PeftModelForSeq2SeqLM` has no `.encoder`; `encode.py` calls `merge_and_unload()` and uses the plain
   T5 encoder. Without this the script fails or silently passes the wrapper.
2. `bleu.py` used to skip writing a `gen_*.jsonl` that already existed, so a `--force` run re-scored the
   old (smoke-test) file. Fixed: it overwrites when it regenerated. If you see BLEU on 40 rows again,
   that guard regressed.

## 5. Continue here

```powershell
cd research-scale-100k
# if the notebook was never executed (check for "execution_count": null)
.\.venv\Scripts\python.exe tools\execute_notebooks.py

# the highest-value next experiment: per-token mapping
#   keep (T, 768) from both encoders, align tokenisers via offset_mapping, fit W on all token pairs,
#   pass the mapped sequence (not a pooled vector) to flan's decoder via encoder_outputs.
```

Before claiming anything about translation quality: run en→vi BLEU and a bootstrap interval on the
1,000-sentence split (see `REPORT.md` §4, bullet 3).
