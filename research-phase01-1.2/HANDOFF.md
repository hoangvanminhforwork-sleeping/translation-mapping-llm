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

Mapped vectors are **not outliers**; they are **over-dispersed and directionless**.

| Detector | held-out normal | mapped |
|---|---|---|
| Isolation Forest | 0.099 | **0.000** |
| GMM (8 comp) | 0.103 | **0.366** |
| Autoencoder | 0.102 | **0.737** |

`cosine(M − μ, Y − μ) = 0.020` on 145,298 held-out token pairs. Variance along the
target's principal axes is 0.296× target while total variance is 1.276× target.

Consequence: the decoder does not see "a weird vector", it sees a vector whose
displacement is orthogonal to the correct one — which predicts fluent, content-free
degeneration, i.e. exactly the 95 % degenerate copy-readout already observed in
ver02. Full argument and caveats in `results/ANOMALY.md`.

## Mission 4 — what to do next, concretely

The insight above converts "reduce the ambiguity" from a vague goal into one
measurable defect: **the map's anisotropy**.

Suggested first experiment, cheap because the embeddings are cached
(`data/anomaly_vectors.npz`, 145 k held-out token pairs, no GPU needed):

1. Compare the current ridge map against a **whitening-constrained** map:
   fit `M` so that `cov(M) ≈ cov(Y)` on the target's principal axes, e.g.
   Procrustes / CCA / a ridge fit followed by `Σ_Y^{1/2} Σ_M^{-1/2}`.
2. Report, on the same held-out pairs: variance ratio on target axes (target:
   1.000), `cosine(M−μ, Y−μ)` (baseline: 0.020), token-level retrieval@1
   (baseline from ver02: 0.0884), and the three detectors' flag rates.
3. Only if direction improves should anyone spend GPU time on generation. The
   ver02 lesson is that a better scalar metric does not imply better text.

Do not re-run the anomaly experiment to "confirm" 0.737 — the number is not the
contribution; the **disagreement between the three detectors** is.

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
