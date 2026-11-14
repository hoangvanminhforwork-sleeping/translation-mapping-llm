# `vn_dishes_cookbook.csv` — extraction report

Source: `data/Vietnamese/Kỹ thuật nấu ăn toàn tập.md` (2,508,334 bytes, 90,050 lines, UTF-8).
Output: `research-phase01/data-work/structured-cookbook/vn_dishes_cookbook.csv` (1,235 rows).
Schema identical to `research-phase01/data-work/structured/vn_dishes.csv`
(`id, title, ingredients, directions, link, source, NER`).

## 1. Three numbers

| | count |
|---|---|
| TOC entries (front matter, `.NNN` page pattern) | **87** (printed pages 7–1144, last TOC line = 357) |
| Body recipe blocks found by the segmenter | **1,293** usable of 1,418 candidates |
| Rows written to the CSV | **1,235** |

**The TOC cannot estimate the dish count.** It is a chapter/section table only: 31 dish
chapters (`Các món luộc`, `Các món canh`, … `Các món bánh - xôi - chè - rau câu.807`) plus
the theory chapters. Its 87 entries are chapters, not dishes, so 87 vs 1,293 is not a
mismatch — it is a different granularity. The body has no per-dish index.

## 2. What the file actually is

- Only 4 markdown `#` headings (lines 28164, 35057, 38352, 81048) — recipe boundaries are
  plain text, not headings.
- Lines 0–357: multi-page TOC. Lines ~360–8301: **theory** (nutrition, cooking methods,
  flower arranging / `cắm hoa`, food preservation). Those sections use `Vật liệu` +
  `Cách thực hiện` (flower arranging) and have no `Nguyên liệu` header at all.
- Line **8302** is the first real recipe (`Luộc Rau`); the recipe chapters run to line 90050.
- The text is OCR output of a scanned book and is visibly damaged
  (`Nguyên liệu` → `Nguyền liệu`, `I.` → `r.`/`h`/`T`/`L`/`!`, dish titles often pure
  gibberish such as `* ^íín' ^tấm/`).

## 3. Markers used (exact)

Normalisation before matching: NFD → strip combining marks → `đ→d` → lowercase → keep
`[a-z0-9 ]` → collapse spaces. A line qualifies only if it is ≤60 chars, does **not** start
with a bullet (`-•*■+♦·`) and is either ≥70 % uppercase or starts with a numbering prefix.

```
ingredient header : ^(?:[a-dl]|i{1,3}|iv|v|[rht!]|\d{1,2})?\s*nguyen\s*lie[ue]?u?$
                    e.g. "I. NGUYÊN LIỆU", "NGUYÊN LIỆU", "1. Nguyên liệu", "a, Nguyên liệu",
                         "r. NGUYÊN LIỆU:", "L NGUYÊN LIỆU;"
directions header : ^(?:...)?\s*(?:cach\s*(?:lam|nau|thuc\s*hien|che\s*bien)|thuc\s*hien|
                    che\s*bien|lam\s*banh|lam)$
                    e.g. "II. CÁCH THỰC HIỆN", "THỰC HIỆN", "Cách làm", "b. Cách thực hiện"
title             : the last non-blank line above the ingredient header (≤70 chars,
                    not ending in . , ; :) — i.e. the printed dish name
block end         : the line before the next recipe's title
```

Counts: 1,206 ingredient headers, 1,705 directions headers.
- 1,075 blocks have both header types,
- 131 have an ingredient header but no recoverable directions header (OCR loss) — kept,
  the whole block is sent to the model,
- 212 are *header-less*: two chapters (drinks ~line 86,000, sorbet/ice-cream ~87,200)
  print only `N. <dish>`, ingredient lines, then steps. Those blocks are cut on short
  numbered dish-title lines (`^\d{1,2}[.)]`) inside the long header-less spans.

## 4. Extraction

`qwen/qwen3.7-flash`, temperature 0, reasoning disabled, `response_format=json_object`,
one call per block (whole section text, ≤14,000 chars), concurrency 5, 5 tries with
exponential backoff. The prompt asks for
`{is_recipe, title, ingredients[], directions[], NER[]}`, one array element per printed
ingredient line / per printed step, Vietnamese only, no invention.

Validation: `json.loads` + type check on all three arrays; one repair call per invalid
response; a final "structural override" call when the model answered `is_recipe=false`
for a block that demonstrably has an ingredient header + ≥3 ingredient lines + ≥2
direction lines (this recovered ~30 dessert/regional recipes that the model had rejected
because their printed title was unreadable).
Result: **0 failures after repair**, 58 blocks classified `is_recipe=false` (see §6).

Caching: `cache/<sha1(prompt_ver|model|section_text)>.json` — 1,293 files. Re-running
`extract.py` with the caches present makes **0 API calls** and reproduces the CSV
byte-for-byte (verified: same SHA-256, `cache hits: 1293, to call: 0`). `--force` re-calls.

## 5. Cost

| | |
|---|---|
| this task, all calls (incl. smoke test, repairs, overrides) | **$0.1245** |
| budget cap | $0.50 |
| calls | 1,642 (includes an early smoke test and re-runs of blocks whose boundaries moved) |
| ~1.63 M input chars (~0.83 M tokens) at $0.03/M, ~0.36 M output tokens at $0.13/M | |

`cost.json` in this folder tracks it. The OCR pipeline's `research-phase01/data-work/cost.json`
was **not** touched.

## 6. Failures and skipped blocks (58)

`failures.json` → `failures: []` (nothing lost after the repair attempt).
`skipped`: 58 blocks the model judged not to be recipes — inspected, these are correct
except for a handful:

| reason | n | what they are |
|---|---|---|
| tips/household section | ~38 | `1. Cách giữ rau …`, `9. Bảo quản tủ lạnh` — advice, not recipes |
| theory / preservation | 4 | `1. Tồn trữ thực phẩm bằng acid`, `2. LÊN MEN`, … |
| index-like lists | 3 | `3. Xốt cà`, `1. Trà`, `17. Kem rượu táo` — bare dish names only |
| dessert block still rejected | ~5 | real kem/sorbet recipes with no readable header (e.g. line 88020, a 127-line merged block) |
| fragments | ~8 | a leftover final step line with no ingredient list |

## 7. Two example rows

```
id            : cookbook-0002
title         : Trứng luộc
ingredients   : ["1 quả trứng", "1/2 bát (chén) nước"]
directions    : ["Trứng rửa sạch.", "Thả nhẹ trứng vào nước lạnh đun sôi.", ...6 steps]
link          : ...Kỹ thuật nấu ăn toàn tập.md#L8331
source        : ky_thuat_nau_an
NER           : ["trứng", "nước", "chanh", "tỏi"]
```
```
id            : cookbook-0401
title         : Sườn nướng
ingredients   : ["1 kg Sườn cốt lết (côteletles)", "2 muỗng xúp sả bào mỏng",
                 "5 tép hành lá lấy phần cù trắng", "1 trái dưa chuột, xà lách", ...]
directions    : ["Chuẩn bị: Sườn chặt ra từng miếng dày độ 8 ly đến 1cm.", "Sả bằm nhừ.", ...]
link          : ...Kỹ thuật nấu ăn toàn tập.md#L34903
NER           : ["Sườn", "Sườn nướng", "Sườn cốt lết"]
```

`link` line numbers are **1-based** (`#L8331` = the dish-title line in the .md file).

Dataset totals: 12,213 ingredient lines and 10,648 step lines; mean 9.9 ingredients and
8.6 steps per row. 1,176 distinct titles out of 1,235 rows (max repeat: `Chè đậu xanh` ×4 —
the book really does print several variants of the same dish).

## 8. Caveats — read before using this data

1. **57 % of titles are reconstructions.** For **706 / 1,235 rows** the printed title line
   was OCR gibberish (`* ^íín' ^tấm/`, `^ ^^ấnẲ`). The prompt asks the model to name the
   dish from its ingredient list and steps in that case; the result is plausible and
   grounded (`^ ^^ấnẲ` + bánh canh + thịt nạc cua → `Bánh canh cua`) but it is *inferred*,
   not read. The raw printed line is preserved in `cache/*.json` (`raw_title`) if you need
   to audit or discard those rows.
2. **Ingredient and step text keeps OCR damage** (`lũũg hành tím`, `cù cải Irẳng`). The
   model was told to fix only obvious typos; quantity digits are frequently misread
   (`300g` vs `30ũg`). Do not treat quantities as reliable without a cleaning pass.
3. **NER is title-only and imperfect.** It is derived from the title, but the model
   occasionally adds a step-text ingredient (`Trứng luộc` → `chanh`, `tỏi`).
4. **Ambiguous sections**: (a) the 131 ingredient-header-only blocks — the directions
   header was lost by OCR, so the model split ingredients from steps on its own;
   (b) the two header-less chapters (§3) where the block boundary is a numbered title
   line rather than a header — one 127-line sorbet block still merges two recipes;
   (c) `III/2. CÁC MÓN LUỘC` prints two "recipes" that are really technique notes
   (`Luộc Rau` has a single line "Chọn rau tươi, không sâu, úng." as its ingredient list).
5. **Not extracted on purpose**: nutrition theory, 30+ pages of flower arranging, food
   preservation, and the fruit-carving chapter (`* Nguyên liệu: Đu đủ xanh…` one-liners,
   no steps). They contain no ingredient-list + steps structure.
6. The file is an OCR of a 1,144-page book, so 1,235 rows is a plausible yield for
   ~31 recipe chapters (≈40 dishes per chapter), not a truncation artefact.

## 9. Files

| file | purpose |
|---|---|
| `vn_dishes_cookbook.csv` | the dataset (1,235 rows) |
| `REPORT.md` | this file |
| `cost.json` | spend for this task |
| `failures.json` | `failures: []`, `skipped: [58]` with reason + raw title |
| `manifest.json` | prompt version, model, block count, digest of block starts |
| `cache/<sha1>.json` | one record per block: raw printed title, parsed JSON, token usage |
| `../../scripts/cookbook/segment.py` | segmenter (the markers of §3) |
| `../../scripts/cookbook/extract.py` | API loop, cache, validation, CSV writer |
| `../../scripts/cookbook/discover.py` | recomputes the three numbers + CSV validation, 0 API calls |
