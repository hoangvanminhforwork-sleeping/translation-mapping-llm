# Handoff — research-phase01-1.2 — 18 Nov 2026

State: **missions 1, 2, 3, 5 and 6 done. Mission 4 specified but not run.**

Budget spent: `paperExtraction` $0.34 of $1.00 · `SAD-score` $0.178 of $0.50 (separate
project) · `coding` $0.00 of $3.00.

## Read first, in this order

1. `results/ANOMALY.md` — the one substantive new finding. Read this before planning
   any further representation work.
2. `translation-intro.md` and `neural-mining-intro.md` — the two introductions.
3. `agent-work/research/text/index.json` — the 87-paper corpus, already extracted.
4. `draft/translation/PRISMA.xlsx`, `draft/neural-mining/PRISMA.xlsx` — completed.

## Mission status

| # | Mission | State | Where |
|---|---|---|---|
| 1 | Rename PDFs to titles | **done** | 87 files; `agent-work/research/rename_log.json` is the inverse |
| 2 | PRISMA | **done** | both workbooks filled; 67 identified → 65 dedup → 63 with full text → 36 included |
| 3 | Read papers / extract text | **done** | `agent-work/research/text/`, 5.12 M chars, page-tagged, $0 |
| 4 | Implementation (reduce ambiguity) | **specified, not run** | see "Mission 4" below |
| 5 | One-class anomaly detection | **done** | `src/anomaly.py` → `results/ANOMALY.md` |
| 6 | Two introductions | **done** | `translation-intro.md`, `neural-mining-intro.md` + 3 figures |

## The corpus

87 PDFs, all with a text layer, no OCR needed.

```
agent-work/research/
  pdf_index.json       title/year/venue/abstract/figure-pages per PDF (recon pass)
  rename_log.json      old name -> new name, one entry per file
  text/index.json      char counts, page counts, figure-candidate pages, sha
  text/<group>/*.md    front matter + abstract + [[p.N]] page-tagged body
  prisma_records.json  the 65 records, each matched to a corpus file
  prisma_screening.json C1/C2/C3 verdict + reason per record
```

Groups: `translation` (37), `neural-mining` (30), `other-references` (15),
`References` (5). **25 corpus PDFs are never cited by any recorded PRISMA search** —
they are background reading, deliberately kept out of the screening flow.

**5 duplicate groups** exist (the same paper in both the `translation` and
`neural-mining` folders, plus one exact duplicate). By instruction they were
renamed and left in place, never deleted. **They are one record each for PRISMA
purposes** — do not count them twice when quoting search yields.

## PRISMA, as built

Criteria (approved 18 Nov), positions kept from the template:

- `C1` cross-lingual / representation-level intervention (template column said
  "Intelligent tutoring or chatbot")
- `C2` NLP / MT evaluation (template column said "IT or CS subjects")
- `C3` survey or review
- **included iff C1 ∧ C2 ∧ ¬C3**

| | translation | neural-mining |
|---|---|---|
| records identified | 37 | 30 |
| no full text | 0 | 2 |
| screened | 37 | 28 |
| included | 17 | 19 |

`Full text URL` is a **direct arXiv / ACL Anthology link** where the pre-rename
filename carried an id (8 of 17 in the translation workbook), otherwise a
**Google Scholar search link** — the header cell says so. Do not cite those
Scholar links as verified locations of the PDF.

## Mission 5 — the finding to carry forward

Mapped vectors are **not anomalous — they are collapsed toward the target mean**.

| Detector | held-out normal | mapped | identity |
|---|---|---|---|
| Isolation Forest | 0.099 | **0.000** | 0.846 |
| GMM (8 comp) | 0.103 | **0.000** | 1.000 |
| Autoencoder | 0.102 | **0.000** | 1.000 |

Median z-distance from the target mean: **6.31** for mapped vs **22.77** for
genuine vectors (p05/p95: 0.99/9.85 vs 16.65/27.46). A constant vector already
reproduces **80 %** of the reported token cosine. Mission 5's hypothesis was
falsified in its literal form: the decoder receives a vector from the *most*
familiar part of its space, not an unfamiliar one — which predicts exactly the
31.6 % token loops ver02 measured.

Full argument and caveats in `results/ANOMALY.md`.

> **Correction.** The first run used `X @ W.T + b`; ver02 uses `X @ W + b`
> (`eval_geometry.py:75`). The transposed run gave the opposite conclusion. The
> tell was a constant vector beating the map on cosine. Always sanity-check an
> unfamiliar matrix convention against the original caller.

## Mission 4 — done (diagnostics + repairs), generation not run

`results/MAP_REPAIRS.md` and `results/recolor_sweep.json`. Headline: **the metric
ver02 reports is gameable** — a constant vector scores `cos_abs` 0.3541 against
Procrustes' 0.2279, even though Procrustes has 22 % better retrieval. Report
`cos_centre` and `const_cos` beside every cosine from now on.

Best two configurations to test on generation:
`procrustes + recolor(t=0.3)` → ret@1 **0.0955**; `ridge + recolor(t=0.1)` →
ret@1 **0.0915** with `cos_centre` 0.2132. Both far above the 0.00025 chance floor.

Correction to the earlier plan: **matching the marginal distance distribution is
not the fix.** The sweep that lands `zdist_med` on the genuine 22.8 (t≈0.3) has
worse retrieval than the uncorrected map. The collapse is a symptom, not the
cause.


## Frozen — do not regenerate casually

- `agent-work/research/text/` — 87 extracted files; rebuild takes ~1 min and no
  money (`extract_text.py`), but downstream PRISMA IDs would shift.
- `agent-work/research/rename_log.json` — the only record of the old filenames.
  Every arXiv/ACL id used for the PRISMA URLs was recovered from it.
- `research-phase01-ver02/data/` — untouched by this session. `map_vi_en.npz` is
  read, never written.
- `data/anomaly_vectors.npz` (~600 MB) — recomputable in 15 s on the GPU via
  `python -m src.anomaly` (delete the file to force it).

## Environment as measured

Shared venv `../research-scale-100k/.venv`: torch 2.7.1+cu128, transformers,
peft, lightgbm 4.6, **sklearn 1.9.1**, matplotlib 3.11.2, pymupdf 1.28.6.
RTX 3050 4 GB, 7.7 GB RAM (1.85 GB free after closing Zalo / msedge / ChatGPT).

Throughput measured this session: **~10,000 aligned token pairs/s** for the frozen
pair encoders (145 k pairs in 15 s), i.e. ~400 sentences/s, faster than the 1,700
pairs/s recorded in the ver02 handoff because that figure included shuffle-order
access to `pairs.parquet` rather than `chosen.parquet`.

**Import trap:** `research-phase01-ver02/src` and `research-phase01-1.2/src` are
both called `src`. `src/anomaly.py` loads the old package under the alias `v2` via
`importlib`, because its modules use relative imports (`from . import config`).
Reusing `sys.path` alone fails with `ModuleNotFoundError: No module named 'src.config'`.

**API trap (hit twice):** `qwen/qwen3.6-35b-a3b` returns `content: ''` when
`max_tokens` is small, because reasoning tokens are drawn from the same budget.
Use `max_tokens >= 3000` for structured JSON output, and always surface the
provider's error text — a bare "FAILED" cost 30 minutes of wall clock here.

## Known limitations

- `src/anomaly.py` autoencoder: CPU, 40 epochs, one seed, fixed architecture.
- Vietnamese-specific literature claims: **none**. No corpus paper studies
  vi→en representation mapping; all Vietnamese numbers are this project's own.
- `pdf_index.json` `first_author` is wrong on ~12 of 87 rows (author block
  merged with a figure or a journal banner). Titles and years are reliable;
  authors must be read off the PDF headers before citing.
- The intros were drafted by sub-agents against this corpus; every claim carries
  an inline citation, but a human pass over the citations before submission is
  still required.
