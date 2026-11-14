# Data work — PDF pages → images → vision OCR → markdown

Model: `qwen/qwen3.7-flash` (OpenRouter chat completions, image input).
Render: PyMuPDF at **180 DPI**, RGB, no alpha. Prompt: `common.PROMPT` (`transcribe-v1`),
re-run variant `transcribe-v2` used only for flagged pages.

Date: 2026-11-14. Wall clock ≈ 15 min (render 120 s, OCR 572 s with 6 workers).

## Deliverables

| path | content |
|---|---|
| `pages/<slug>/page-NNNN.png` | 586 rendered page images (kept on purpose, ~100 MiB) |
| `ocr/<slug>/page-NNNN.md` | 586 per-page transcriptions (the resumable unit) |
| `markdown/<slug>.md` | merged book text with `<!-- page N -->` separators |
| `ocr_log.jsonl` | one JSON line per API call (book, page, status, tokens, usd, latency) |
| `cost.json`, `quality.json`, `accuracy.json`, `inkcheck.json` | measured cost + QA output |
| `scripts/` | the pipeline (see below) |

## Results

| book | pages | transcribed | API fails | mean chars/page | USD |
|---|---|---|---|---|---|
| vn_mon_an_truyen_thong | 223 | 223 | 0 | 1010 | 0.0289 |
| vn_van_hoa_am_thuc | 260 | 260 | 0 | 1323 | 0.0269 |
| us_american_cuisine | 31 | 31 | 0 | 966 | 0.0033 |
| us_sioux_chef | 2 | 2 | 0 | 887 | 0.0002 |
| us_food_younger_land | 70 | 70 | 0 | 832 | 0.0074 |
| **total** | **586** | **586** | **0** | – | **0.0667** |

Cost guard ($0.50) never approached: final spend **$0.066722**, i.e. ~$0.000114/page.
Mean usage per page: 2291 prompt tokens, 326 completion tokens.

## Pipeline / how to reproduce

```powershell
$py = 'research-scale-100k\.venv\Scripts\python.exe'
& $py research-phase01\data-work\scripts\render.py        # 180 DPI, skips existing PNGs
& $py research-phase01\data-work\scripts\ocr.py --workers 6 # resumable, $0 on rerun
& $py research-phase01\data-work\scripts\merge.py
& $py research-phase01\data-work\scripts\accuracy.py      # English books only
& $py research-phase01\data-work\scripts\summarize.py
```

Every stage is skip-if-exists; `--force` redoes. `ocr.py` writes `agent-work/dataworker/STOPPED.md`
and stops submitting work the moment cumulative spend in `ocr_log.jsonl` passes $0.50.

Two decisions worth knowing:

1. **Reasoning disabled** (`"reasoning": {"enabled": false}`). With thinking on, page 1 cost
   3127 completion tokens ($0.00049) instead of 330 ($0.00013) for the *same* transcription;
   the thinking tokens are billed as output and eat the `max_tokens` budget, which risks
   truncated pages. Turning it off cut cost 4× and removed truncation risk (0 truncated of 586).
2. **180 DPI is enough.** Smoke test on `vn_mon_an` p1 returned perfect diacritics
   (`MÓN ĂN VIỆT NAM TRUYỀN THỐNG`, `CÁ ĐUỐI NẤU CÀ RI`, `XÔI VÒ`, `MẮM TÔM CHUA`), no `?`
   mangling, so no DPI escalation was needed.

## Spot-check (PNG vs transcription, read side by side)

| page | verdict |
|---|---|
| `vn_mon_an_truyen_thong` p30 (recipe body) | Matches line for line. One real error: image `được`, OCR `dược` (đ lost at line start). Diacritics otherwise exact. |
| `vn_van_hoa_am_thuc` p20 (dense academic page) | Full page present, all diacritics correct (`Văn hóa ẩm thực`, `Đào Duy Anh`, `Ngô Đức Thịnh`). Deviation: the 30 hard-wrapped source lines were re-flowed into 5 paragraphs; italics kept as `*...*`. No missing sentence. |
| `us_american_cuisine` p12 | Byte-identical to the page, line breaks preserved, footer captured. |
| `us_food_younger_land` p40 | Complete, correct reading order, footer UI text captured. Bold headings came back as plain text, not `##`. |
| `vn_van_hoa_am_thuc` p1 / p109 / p195 (covers + part-title pages) | Correct, including `VĂN HÓA ẨM THỰC VIỆT NAM`, `TS.NGUYỄN THỊ BẢY`. |

Quantitative check on the three English PDFs (they carry a clean Unicode text layer, so the
transcription can be diffed against ground truth): mean character similarity 0.987 / 0.983 /
0.947 and mean word similarity 0.989 / 0.994 / 0.939. The "worst" pages (0.78–0.85) are only
markdown emphasis added around already-correct text (`**stuffed eggs**`), not lost content.
The two Vietnamese PDFs cannot be scored automatically: `vn_mon_an`'s text layer is legacy
VNI-mojibake (`MOÁN ÙN` for `MÓN ĂN`) and `vn_van_hoa` is a pure scan with no text layer.

## Failure modes found (all handled, none silent)

* **Sparse pages lose body text.** `vn_mon_an` p64 (header + one sentence + footer) came back as
  just `Nhiều tác giả / 64`; a 240 DPI retry and a v2 prompt both failed the same way. Root cause:
  the whole-page image gets downscaled and the single line disappears. Fixed with
  `redo_short.py` (detects pages whose word count is < 60 % of the PDF text layer's) +
  `redo_blocks.py`, which crops each *text block bbox* at 400 DPI and OCRs it separately →
  correct text recovered. This flagged exactly **1 of 586 pages**.
* **Band crops hallucinate.** A first attempt that sliced the page into horizontal bands made
  the model invent an advert headline (`THẾ HỆ MỚI / VỚI CÔNG NGHỆ`) on an empty strip. That
  approach is abandoned; `redo_blocks.py` only ever shows the model pixels that provably
  contain text. The hallucinated text was removed from the output.
* **`[BLANK]` / `[IMAGE-ONLY]` never fired.** No page was skipped: the model always transcribed
  at least the running header, so ~1 % of pages are just `Nhiều tác giả <page>` where the page
  is physically near-empty. Nothing was invented to fill them.
* 0 HTTP failures, 0 retries beyond the first try, 0 truncated responses across 586 pages.

## Caveats for downstream use

* `us_american_cuisine` and `us_food_younger_land` are **Bookey study-guide PDFs**, not the books
  (they are Q&A summaries with `More Free Books on Bookey` footers). `us_sioux_chef` is a 2-page
  publisher blurb, not the cookbook. Only the two Vietnamese PDFs are full books.
* The Vietnamese scan has line-break reflow (paragraphs joined); the English PDFs keep original
  line breaks. If paragraph boundaries matter, re-split on the source PDF.
* Rare `đ → d` substitutions (measured: 1 on the inspected page 30). Everything else inspected
  was exact. For print-critical quotes, check `đ` in the `markdown/` output against the PNG.
