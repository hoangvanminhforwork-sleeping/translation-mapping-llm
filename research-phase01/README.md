# research-phase01 — the whole phase-1 research, vi → en

One pipeline, one decoder, three channels. The claim under test is narrow and falsifiable:

> A single 768×768 linear map can carry a Vietnamese sentence from vit5-base's representation space
> into flan-t5-base's, well enough that flan's decoder produces the same English it would produce from
> its own encoder.

## Read in this order

1. `REPORT.md` — what was measured, what the numbers mean, what they do *not* show.
2. `LEARNING_NOTE.md` — why each choice was made, including the two that cost hours.
3. `01_results.ipynb` — the same numbers as tables, executed.
4. `AGENT_HANDOFF.md` — state for the next agent (or for you, tomorrow).

## Layout

```
src/
  config.py          paths, models, hyper-params, the no-leakage exclusion list
  build_pairs.py     3.46M unique pairs, md5-ordered split (no RNG), prior rows excluded
  tokutil.py         vit5 slow->fast spiece conversion (char offsets), cross-tokeniser alignment
  tokenbatch.py      GPU alignment + length-cropped batches + tokeniser prefetch
  fit_tok.py         streaming per-token ridge: A,B,C moments per 200k-pair shard
  eval_geometry.py   token/sentence cosine, retrieval@1, identity + mean controls
  train_decoder.py   mixed-channel LoRA (native + mapped), decoder-only, encoder frozen
  generate.py        3 channels x 2,000 test pairs
  score.py           BLEU, chrF, COMET + loop/echo diagnostics
  probe.py           §3.1 logistic probe on frozen flan embeddings
  decision_check.py  §3.3 probe vs morph/reflex agreement on MTET
data/                pairs.parquet, stats/*.npz, map_vi_en.npz, probe_* (gitignored except manifests)
results/             every number, committed
data-work/           M1: 586 OCR'd PDF pages, markdown, structured recipes + atomic facts
```

## Reproduce

```powershell
cd research-phase01
$py = "..\research-scale-100k\.venv\Scripts\python.exe"   # reused venv, not copied

& $py -m src.build_pairs                      # ~4 min, CPU+DuckDB
& $py -m src.fit_tok shards                   # 18 shards, resumable, ~2 h GPU
& $py -m src.fit_tok fit                      # seconds, CPU
& $py -m src.eval_geometry                    # ~1 min GPU
& $py -m src.train_decoder --pairs 100000     # ~30-60 min GPU
& $py -m src.generate                         # ~7 min GPU
& $py -m src.score --comet                    # ~5 min GPU (+2.3 GB COMET download)
& $py -m src.probe                            # ~25 min (200k embeddings)
& $py -m src.decision_check --n 200           # ~2 min (API, $0.001)
& $py tools\make_notebook.py                  # rebuilds 01_results.ipynb
```

Every stage checks its own output and skips; `--force` redoes that stage only. `data/stats/*.npz` is the
resume unit, so a run interrupted at shard 12 keeps shards 0-11 and a later run adds to them.

## Prerequisites

| What | Where | Notes |
|---|---|---|
| EN↔VI corpus, 8.33M rows | `../data/data/*.parquet` | 4.4 GB, already present |
| vit5-base, flan-t5-base | `../research-prototype/models/hf/` | `HF_HOME` points here |
| Python env | `../research-scale-100k/.venv` | torch 2.7.1+cu128, transformers 4.57.6, peft 0.21.2, unbabel-comet 2.2.7 |

**Dependency note:** `pip install unbabel-comet` pins `numpy<2`, which breaks `pyarrow 26` and
`contourpy 1.4` in this venv. NumPy was restored to 2.5.4 and COMET 2.2.7 imports and runs fine
against it; `setuptools<81` is required for `pkg_resources` (torchmetrics). Do not upgrade setuptools.

## Hardware reality

RTX 3050 4 GB / 7.7 GB RAM. Encoding runs at ~600 seq/s while the GPU is shared and ~350 seq/s when two
encoders share the card alone; one heavy job at a time.
