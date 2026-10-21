# Reproduce the mapping + probing prototype

Everything runs from this folder (`small-experiment/`). Nothing is written to C:.

## 0. Prerequisites

* Docker Desktop running (MySQL 9.4 image pulls to `artifacts/mysql`, ~600 MB).
* ~4 GB free on D: for torch + wheels, ~2.5 GB RAM free at encoding time.
* `data/data/train-00008-of-00014.parquet` and `data/RecipeNLG.csv` present.

## 1. Environment (idempotent: it deletes and rebuilds `.venv`)

```powershell
cd small-experiment
powershell -ExecutionPolicy Bypass -File .\install_env.ps1
```

The script removes `.venv`, installs torch from the CUDA 12.8 wheel index
(`torch==2.7.1+cu128`, RTX 3050 = sm_86), installs `requirements.txt`, then
freezes exact versions to `requirements.lock.txt`. To restore a known-good
environment later:

```powershell
Remove-Item -Recurse -Force .\.venv
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
```

## 2. Database

```powershell
docker compose up -d          # create/start csi-mysql on 127.0.0.1:3307
docker compose ps             # wait for "healthy"
```

`docker compose down` stops it; the data stays in `artifacts/mysql` because the
schema is created with `IF NOT EXISTS` and all writes are upserts. To start over:
`docker compose down` then delete `artifacts/mysql`.

## 3. Pipeline (each stage is safe to re-run)

```powershell
.\.venv\Scripts\python.exe -m src.pipeline sample     # 12k pairs -> artifacts/sample_pairs.parquet
.\.venv\Scripts\python.exe -m src.pipeline encode     # 72k vectors -> MySQL
.\.venv\Scripts\python.exe -m src.pipeline verify     # row counts + MySQL vs numpy k-NN
.\.venv\Scripts\python.exe -m src.pipeline mapping    # results/mapping_metrics.csv + png
.\.venv\Scripts\python.exe -m src.pipeline probe      # results/probe_food.json + csv
```

`--force` on `sample`/`encode` redraws the 12k pairs (only do this if you also
re-run everything downstream). Override the seed in `src/config.py`.

Notebooks in `notebooks/` are thin wrappers around the same functions:
`01_sample` -> `02_encode_mysql` -> `03_mapping` -> `04_probing`.

## 4. What to check

| file | meaning |
|---|---|
| `agent-work/manifest.json` | per-stage: seed, SQL, package versions, git commit, data + result SHA-256 |
| `results/mapping_metrics.csv` | MSE / MAE / cosine / R@k for identity, ridge, MLP x 3 poolings |
| `results/probe_food.json` | 5-fold CV accuracy / F1 / balanced acc / ROC-AUC vs majority baseline |
| `results/retrieval_at1.png` | R@1 per pooling and method |

Design decisions and their justifications: `agent-work/ASSUMPTIONS.md`.
