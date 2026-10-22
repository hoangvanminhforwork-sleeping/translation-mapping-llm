# Reproduce small-experiment-02

Everything below runs from `small-experiment-02/`. Total cost on the reference
machine (i5-12500H, 8 GB RAM, RTX 3050 4 GB): **~10 minutes**, GPU-light.

## 0. Prerequisites (checked, not assumed)

* `../small-experiment/.venv` exists with `torch==2.7.1+cu128`, `transformers==4.57.6`.
  This experiment **reuses** it instead of creating a second environment (a
  duplicate would add ~3 GB on D: for identical pins). Recreate it deterministically
  with `../small-experiment/install_env.ps1`.
* `../small-experiment/models/hf` holds both checkpoints (936 MB). Nothing is downloaded.
* `../small-experiment/artifacts/sample_pairs.parquet` holds the frozen 12k draw.

Added pins for this experiment, already installed in that venv:

```
sacrebleu==2.5.1
rouge-score==0.1.2
nltk==3.9.2
```

## 1. Clean slate (optional, and destructive)

Reproducibility means "delete, then rebuild", so remove the derived products
first. Source inputs are never touched.

```powershell
cd small-experiment-02
Remove-Item artifacts\vec_*.npy, artifacts\pairs.parquet, maps\*, results\hyp_*.parquet -Force -ErrorAction SilentlyContinue
```

`results/*.csv` and `results/summary.md` are evidence, not derived caches; they
are rewritten wholesale by the last stage.

## 2. Run

```powershell
$env:PYTHONIOENCODING = "utf-8"          # Windows console is cp1252; Vietnamese text crashes without this
$env:HF_HOME = "$PWD\..\small-experiment\models\hf"
$py = "..\small-experiment\.venv\Scripts\python.exe"

& $py -m src.pipeline cache     #  ~2 min   -> artifacts/vec_{en,vi}_{mean,cls,last}.npy + results/cache_summary.json
& $py -m src.pipeline maps      #  ~30 s    -> maps/*.npz, maps/*.pt, results/map_sanity.csv
& $py -m src.pipeline nn        #  ~8 min   -> results/nn_metrics.csv   (the main result)
$env:NLTK_DATA = "$PWD\agent-work\nltk_data"
& $py -m src.pipeline metrics   #  ~1 min   -> results/token_metrics.csv (only after `invert`)
```

`cache` and `maps` skip work whose output already exists; add `--force` to refit
maps. `nn` and `metrics` always rewrite their CSV.

Anything can be run twice: `.npy` files are keyed by (language, pooling), the
pair file is a one-time copy, the DDL-free stages just overwrite.

## 3. The inversion attempt (slow, documented negative result)

```powershell
& $py -m src.pipeline invert --n=64 --steps=100    # ~27 min; writes results/hyp_*.parquet
```

Do not run it on 2,000 items: measured 4.2 s per item per (direction, pooling)
means ~18.7 h. See `results/summary.md` §3 for why it was abandoned.

The small-scale demonstration that the method itself works:

```powershell
& $py agent-work\r1_pilot_short_sentences.py        # 8 sentences, ~40 s
```

## 4. Verify you got the same thing

| file | check |
|---|---|
| `results/map_sanity.csv` | en2vi/mean ridge raw cosine 0.485, norm ratio 0.53 |
| `results/nn_metrics.csv` | en2vi/mean ridge BLEU 3.27, roundtrip 5.10, shuffled 0.24 |
| `results/cache_summary.json` | `pairs_sha256 = 8ac9561b...60bc`, six arrays of shape [12000, 768] |

Small deviations (<1%) are expected: fp16 encoding on a different GPU is not
bit-identical.

## 5. Traps this experiment already hit

1. `torch.load(..., weights_only=True)` rejects numpy arrays inside the payload.
   Scalers are stored as tensors in `maps.py` for this reason.
2. Raw cosine is not comparable across spaces (English is anisotropic). Any
   cosine claim must say whether the mean direction was removed.
3. `Path.write_text` fails if the directory does not exist; `src/config.py`
   creates `results/`, `maps/`, `artifacts/`, `agent-work/` on import.
4. The NLTK corpora must go to `agent-work/nltk_data` (already set via
   `NLTK_DATA` in `src/config.py`), never the C: drive.
