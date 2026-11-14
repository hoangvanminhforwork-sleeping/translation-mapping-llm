# Handoff — research-phase01 — 2026-11-14, ended 19:40 VN

## 0. Read first, in this order

1. `REPORT.md` — every number, what it means, what it does not show (`§5.2` has the corpus-noise caveat).
2. `results/metrics.json` — the generation table with bootstrap intervals.
3. `results/mapping_metrics.csv` — geometry, both directions, with identity/mean controls.
4. `LEARNING_NOTE.md` — why the pipeline looks like this; the two traps that cost hours.

## 1. State — all stages finished, nothing half-written

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| 1 | corpus, no leakage | DONE 3,456,741 pairs | `data/pairs_manifest.json` (sha256 `d99030e3…`) |
| 2 | per-token ridge, both directions | DONE, λ=0 | `data/map_vi_en.npz`, `data/map_en_vi.npz` |
| 3 | geometry + controls | DONE | `results/mapping_metrics.{csv,json}` |
| 4 | mixed-channel decoder | DONE 3,126 steps / 100k pairs / 43.3 min | `adapters/flan-decoder-vi-en/` + `done.json` |
| 5 | 4 channels × 2,000 generations | DONE | `results/gen_{mapped,native,identity,baseline}.jsonl` |
| 6 | BLEU/chrF/COMET + bootstrap | DONE | `results/metrics.json` |
| 7 | probe (§3.1) | DONE but **fails its purpose** — see §4 | `results/probe_metrics.json` |
| 8 | probe vs decision model (§3.3) | DONE, κ = 0.00 | `results/food_consistency.json` |
| 9 | M1 data | DONE 1,402 recipes / 1,544 facts | `data-work/`, `results/data_audit.json` |
| 10 | executed notebook | DONE | `01_results.ipynb` |

Headline: mapped **6.28 BLEU / 28.15 chrF / 0.600 COMET** vs native 3.85 / 18.72 / 0.443 vs identity
0.01 / 10.99 / 0.288 vs opus-mt baseline 22.27 / 45.31 / 0.752.

## 2. Environment as measured

`research-scale-100k/.venv` reused (not copied). torch 2.7.1+cu128, transformers 4.57.6, peft 0.21.2,
unbabel-comet 2.2.7, numpy 2.5.4, setuptools 80.10.2 (must stay <81 for `pkg_resources`).
HF cache: `../research-prototype/models/hf` (COMET checkpoint included, 1.4 GB).
GPU RTX 3050 4 GB: encoding 600 seq/s at 79-81 °C; decoder training 38 pairs/s at 3,955/4,096 MiB VRAM.

## 3. Frozen — do not touch

* `data/pairs.parquet` (306 MB, gitignored) — the split is pinned by `ORDER BY md5(vi)`; changing it
  invalidates every number.
* `data/stats/*.npz` — 19 files, the only way to refit the map without 2 h 37 min of GPU.
* `results/gen_*.jsonl` — 2,000 rows each; the primary generation evidence.
* `results/metrics.json` — includes `bleu_ci95` and the paired comparisons.
* `adapters/flan-decoder-vi-en/` — the map is fitted into *that* decoder's cross-attention behaviour.

## 4. Two results that contradict their own framing (do not "fix" them by re-running)

1. **The probe is a domain classifier, not a food detector.** Test AUC 0.99999997 because positives are
   recipe ingredient lists and negatives are news/legal prose. A within-domain probe (windows of recipe
   text with vs without a NER term) is the experiment §3.1 actually needs.
2. **§3.3 agreement is 98.5 % with κ = 0.00.** Probe and decision model agree only by both answering
   "no"; both are at chance against the lexicon. Sentence-level `noul` is the wrong instrument — the
   decision needs span-level detection.

## 5. Highest-value next steps, in order

1. **Train the decoder longer.** 1 epoch / 100k pairs is the binding constraint, not the map: the same
   adapter resumes (`python -m src.train_decoder --pairs 300000`), and the whole encoder→decoder gap
   (6.28 vs 22.27) is the thing a reader will ask about.
2. **en→vi BLEU.** The reverse map already exists (`data/map_en_vi.npz`, geometry 0.814 sentence cosine,
   44.0 % retrieval@1). Needs a vit5 decoder LoRA trained the same two-channel way.
3. **Word-level alignment** instead of the 13.7 %-guess character overlap.
4. **A per-sentence audit of 100 references** to quantify corpus misalignment (one case seen in twelve).
5. Two adapters instead of one mixed adapter, to remove the shared-capacity confound.

## 6. Resume commands

```powershell
cd research-phase01
$py = "..\research-scale-100k\.venv\Scripts\python.exe"
& $py -m src.fit_tok fit                 # instant, needs data/stats/*
& $py -m src.eval_geometry
& $py -m src.train_decoder --pairs 100000    # resumes the adapter if one exists
& $py -m src.generate --force
& $py -m src.score --comet
& $py -m src.audit_data
& $py tools\make_notebook.py; & $py tools\execute_notebook.py
```
