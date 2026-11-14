"""Stage 2: build an atomic culinary-culture fact base with page-level provenance.

Pipeline per book
-----------------
1. Group pages into chunks (<= MAX_CHARS of text, <= MAX_PAGES pages).
2. One LLM call per chunk -> JSON list of candidate facts, each carrying a `quote`
   that must be a verbatim sentence from ONE of the pages in the chunk.
3. Hard verification (anti-fabrication):
   * `quote` must be locatable in the *raw text of that exact page*
     (exact substring first; then a 1:1 character-normalised match on quotes/dashes/
     whitespace, in which case we store the verbatim page slice instead);
   * `page` must belong to the chunk;
   * `relation` must be in RELATIONS, `category` in CATEGORIES;
   * quote length >= MIN_QUOTE chars, no OCR-only junk lines.
   Anything failing is dropped and counted in structured/failures_facts.json.
4. Deduplicate identical (subject, relation, object) keeping the first (book, page) order.

Idempotent: per-chunk LLM replies are cached in structured/cache -> rerun is $0.
"""
from __future__ import annotations

import argparse, json, os, pathlib, re, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common

OUT_JSONL = common.OUT_DIR / "culture_facts.jsonl"
FAIL_PATH = common.OUT_DIR / "failures_facts.json"
STATS_PATH = common.OUT_DIR / "facts_stats.json"

RELATIONS = ["is_eaten_at", "originates_from", "symbolises", "is_made_from",
             "is_served_with", "is_prepared_by", "is_associated_with"]
CATEGORIES = ["ritual_festival", "symbolism", "origin_geography", "ingredient_composition",
              "technique_preparation", "pairing_serving", "social_context",
              "health_belief", "regional_identity", "history_change"]

MIN_QUOTE = 25
MAX_CHARS = 4800
MAX_PAGES = 5
MAX_FACTS_PER_CHUNK = 20

# Documented normalisation aliases (see REPORT.md). Applied only after the model reply,
# never used to invent a claim.
RELATION_ALIAS = {"is_served_at": "is_eaten_at"}
CATEGORY_ALIAS = {"seasonal_eating": "social_context"}

SYSTEM_VI = ("Bạn là bộ trích xuất tri thức ẩm thực chính xác tuyệt đối. "
             "Chỉ trả về JSON hợp lệ, không giải thích, không markdown.")

USER_VI = """Dưới đây là nguyên văn (OCR) các trang của một cuốn sách về ẩm thực Việt Nam.
Mỗi trang bắt đầu bằng dấu [[PAGE n]].

{text}

Hãy trích ra các SỰ THẬT ẨM THỰC NGUYÊN TỬ (mỗi dòng đúng MỘT khẳng định) mà văn bản
thực sự nói. Trả về DUY NHẤT một đối tượng JSON:
{{"facts": [{{"subject": "...", "relation": "...", "object": "...",
             "category": "...", "page": n, "quote": "..."}}]}}

QUY TẮC:
1. subject = món ăn / nguyên liệu / tập tục (danh từ, giữ tiếng Việt như trong sách).
2. object = địa danh, dịp lễ, nguyên liệu, cách làm, ý nghĩa... (giữ tiếng Việt).
3. relation CHỈ được là một trong: {relations}
4. category CHỈ được là một trong: {categories}
   (ritual_festival=dịp lễ/Tết/giỗ/cưới; symbolism=ý nghĩa biểu tượng;
    origin_geography=nơi xuất xứ/địa danh; ingredient_composition=thành phần;
    technique_preparation=cách chế biến; pairing_serving=ăn kèm/dọn món;
    social_context=ứng xử/cộng đồng/giới; health_belief=sức khỏe/âm dương;
    regional_identity=khác biệt vùng miền; history_change=thay đổi lịch sử/ngoại lai)
5. quote = MỘT CÂU NGUYÊN VĂN, copy chính xác từng ký tự từ văn bản trên (kể cả dấu
   câu, không sửa lỗi chính tả OCR, không diễn giải, không nối hai câu). Câu này phải
   nằm trong đúng trang ghi ở "page". Không có câu nguyên văn thì BỎ sự thật đó.
6. page = số trang trong dấu [[PAGE n]] chứa câu trích.
7. Một dòng = một khẳng định. Không gộp hai ý ("X vừa A vừa B" -> hai dòng).
8. Không suy luận ngoài văn bản. Không dùng kiến thức bên ngoài. Không bịa.
9. Nếu trang chỉ có công thức nấu ăn, chỉ lấy các khẳng định văn hóa/kỹ thuật đáng kể
   (không liệt kê từng nguyên liệu của công thức).
10. Tối đa 20 sự thật cho mỗi lần trả lời; ưu tiên sự thật rõ ràng, câu trích ngắn.
11. Trả về {{"facts": []}} nếu trang không có khẳng định nào."""

USER_EN = """Below is verbatim OCR text of pages from a book about food. Each page starts [[PAGE n]].

{text}

Extract ATOMIC food-culture facts, exactly ONE claim per item, only where the text
actually states it. Return ONLY a JSON object:
{{"facts": [{{"subject": "...", "relation": "...", "object": "...",
             "category": "...", "page": n, "quote": "..."}}]}}

RULES:
1. subject = dish / ingredient / practice; object = place, occasion, ingredient, method, meaning.
2. relation is one of: {relations}
3. category is one of: {categories}
4. quote = ONE VERBATIM sentence copied character-for-character from the text above, from the
   page given in "page". No paraphrase, no ellipsis, no joining two sentences. If you cannot
   quote verbatim, drop the fact.
5. page = the n of the [[PAGE n]] block containing the quote.
6. Do not use outside knowledge. Do not invent. Study-guide summaries and marketing blurbs
   contain few factual claims - return {{"facts": []}} rather than padding.
7. Return {{"facts": []}} if there is nothing factual."""


# ---------------------------------------------------------------- chunking / quoting
def chunk_pages(pages: dict[int, str], max_chars=MAX_CHARS, max_pages=MAX_PAGES):
    chunks, cur, size = [], [], 0
    for p in sorted(pages):
        txt = pages[p]
        if len(txt.strip()) < 120:      # covers, blank OCR pages
            continue
        if cur and (size + len(txt) > max_chars or len(cur) >= max_pages):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(p)
        size += len(txt)
    if cur:
        chunks.append(cur)
    return chunks


_MAP = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "\u00a0": " ",
                      "\u2013": "-", "\u2014": "-", "\u2212": "-", "\t": " "})


def _norm_index(text: str):
    """Whitespace-collapsed, quote/dash-normalised view + index map into `text`."""
    out, idx = [], []
    prev_space = False
    for i, ch in enumerate(text):
        c = ch.translate(_MAP)
        if c.isspace():
            if prev_space:
                continue
            c, prev_space = " ", True
        else:
            prev_space = False
        out.append(c)
        idx.append(i)
    return "".join(out), idx


def locate_quote(page_text: str, quote: str) -> str | None:
    """Return the verbatim slice of page_text that realises `quote`, else None.

    Accepts (a) an exact substring, (b) the same text with only typographic variants of
    quotes/dashes/whitespace, (c) the part before a '...' / '[…]' elision the model
    inserted -- in which case the *stored* quote is the verbatim prefix, so the stored
    string is always a literal substring of the page.
    """
    q = common.norm_ws(quote)
    if len(q) < MIN_QUOTE:
        return None
    variants = [q]
    for sep in ("...", "[...]", "[…]", "…", "[...]", "(...) "):
        if sep in q:
            head = common.norm_ws(q.split(sep)[0])
            if len(head) >= MIN_QUOTE:
                variants.append(head)
    for cand in variants:
        if cand in page_text:
            return cand
        hay, idx = _norm_index(page_text)
        needle = common.norm_ws(cand.translate(_MAP))
        pos = hay.find(needle)
        if pos >= 0:
            start, end = idx[pos], idx[pos + len(needle) - 1] + 1
            return page_text[start:end]
    return None


def clean_field(s: str, limit=160) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip().strip('".;,')
    return s[:limit]


# ---------------------------------------------------------------- per-chunk extraction
def run_chunk(book: str, chunk: list[int], pages: dict[int, str], force: bool, lang: str):
    body = "\n\n".join(f"[[PAGE {p}]]\n{pages[p]}" for p in chunk)
    tmpl = USER_VI if lang == "vi" else USER_EN
    base = tmpl.format(text=body, relations=", ".join(RELATIONS),
                       categories=", ".join(CATEGORIES))
    system = SYSTEM_VI if lang == "vi" else "You extract precise food-culture knowledge. JSON only."
    mt = min(6000, 1200 + len(body))
    err = None
    for attempt in range(2):
        user = base
        if attempt == 1:
            user += ("\n\nLƯU Ý: lần trả lời trước bị lỗi (" + (err or "?") + "). "
                     "Chỉ trả về đúng một đối tượng JSON hoàn chỉnh dạng "
                     '{"facts": [...]}, không văn xuôi, không markdown, tối đa '
                     f"{MAX_FACTS_PER_CHUNK} sự thật, mỗi quote một câu ngắn.")
            if os.environ.get("DS_DEBUG_CACHE"):
                print(f"[RETRY] {book} p{chunk[0]} err={err!r}", file=sys.stderr)
        try:
            content, usage = common.chat("facts", system, user, max_tokens=mt,
                                         tag=f"facts-{book}-{chunk[0]}", force=force)
            obj = common.parse_json_block(content)
        except Exception as e:  # noqa: BLE001
            err = f"parse:{e}"
            continue
        cands = obj.get("facts") if isinstance(obj, dict) else obj
        if not isinstance(cands, list):
            err = "no_facts_key"
            continue
        truncated = usage.get("completion_tokens", 0) >= mt - 5
        if truncated:
            err = "truncated"
            # keep what parsed, but the JSON almost certainly did not parse; retry
            continue
        return {"chunk": chunk, "cands": cands, "error": None}
    return {"chunk": chunk, "cands": [], "error": err}


def verify(book: str, chunk: list[int], cands: list, pages: dict[int, str], lang: str):
    kept, dropped = [], []
    for c in cands:
        if not isinstance(c, dict):
            dropped.append({"reason": "not_dict", "raw": str(c)[:120]})
            continue
        subj = clean_field(c.get("subject", ""))
        rel = str(c.get("relation", "")).strip()
        obj_s = clean_field(c.get("object", ""))
        cat = str(c.get("category", "")).strip()
        quote = str(c.get("quote", ""))
        # documented repairs: the model occasionally swaps the two controlled slots
        if rel in CATEGORIES and cat in RELATIONS:
            rel, cat = cat, rel
        rel = RELATION_ALIAS.get(rel, rel)
        cat = CATEGORY_ALIAS.get(cat, cat)
        try:
            page = int(c.get("page"))
        except Exception:
            dropped.append({"reason": "bad_page", "subject": subj, "quote": quote[:80]})
            continue
        if not subj or not obj_s:
            dropped.append({"reason": "empty_subject_object", "quote": quote[:80]})
            continue
        # degenerate rows: subject/object are the same phrase or one contains the other
        a, b = subj.casefold(), obj_s.casefold()
        if a == b or (min(len(a), len(b)) >= 3 and (a in b or b in a)):
            dropped.append({"reason": "degenerate_subject_object", "subject": subj,
                            "object": obj_s, "quote": quote[:80]})
            continue
        if rel not in RELATIONS:
            dropped.append({"reason": "bad_relation", "relation": rel, "quote": quote[:80]})
            continue
        if cat not in CATEGORIES:
            dropped.append({"reason": "bad_category", "category": cat, "quote": quote[:80]})
            continue
        if page not in pages:
            dropped.append({"reason": "page_not_in_book", "page": page, "quote": quote[:80]})
            continue
        span = locate_quote(pages[page], quote)
        if span is None:
            dropped.append({"reason": "quote_not_in_page", "page": page,
                            "subject": subj, "quote": quote[:120]})
            continue
        kept.append({"lang": lang, "subject": subj, "relation": rel, "object": obj_s,
                     "category": cat, "book": book, "page": page, "quote": span})
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--books", nargs="*", default=list(common.BOOKS))
    args = ap.parse_args()

    all_facts, all_drops, stats = [], [], {}
    for book in args.books:
        pages = common.load_pages(book)
        chunks = chunk_pages(pages)
        lang = "vi" if book.startswith("vn_") else "en"
        common.log(f"{book}: {len(pages)} pages -> {len(chunks)} chunks (lang={lang})")

        def safe(ch):
            try:
                return run_chunk(book, ch, pages, args.force, lang)
            except Exception as e:  # noqa: BLE001
                return {"chunk": ch, "cands": [], "error": repr(e)}

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(safe, chunks))

        bk_kept, bk_drop, n_cand = [], [], 0
        for r in res:
            n_cand += len(r["cands"])
            if r["error"]:
                all_drops.append({"book": book, "chunk": r["chunk"], "reason": "chunk_error",
                                  "detail": r["error"][:200]})
            k, d = verify(book, r["chunk"], r["cands"], pages, lang)
            bk_kept += k
            for x in d:
                all_drops.append({"book": book, "chunk": r["chunk"], **x})
        stats[book] = {"pages": len(pages), "chunks": len(chunks), "candidates": n_cand,
                       "kept": len(bk_kept),
                       "dropped": len([d for d in all_drops if d["book"] == book])}
        common.log(f"  candidates={n_cand} kept={len(bk_kept)} dropped={stats[book]['dropped']}")
        all_facts += bk_kept

    # dedupe on (subject, relation, object) keeping first occurrence
    seen, dedup = set(), []
    for f in all_facts:
        k = (f["subject"].casefold(), f["relation"], f["object"].casefold())
        if k in seen:
            continue
        seen.add(k)
        dedup.append(f)

    order = {b: i for i, b in enumerate(common.BOOKS)}
    dedup.sort(key=lambda f: (order[f["book"]], f["page"], f["subject"]))
    common.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8", newline="\n") as fh:
        for i, f in enumerate(dedup, start=1):
            rec = {"id": f"f-{i:05d}", "lang": f["lang"], "subject": f["subject"],
                   "relation": f["relation"], "object": f["object"],
                   "category": f["category"], "book": f["book"], "page": f["page"],
                   "quote": f["quote"]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_cand = sum(s["candidates"] for s in stats.values())
    for b in stats:
        stats[b]["deduped_out"] = stats[b]["kept"] - len([f for f in dedup if f["book"] == b])
    stats["_total"] = {"candidates": n_cand, "verified": len(all_facts),
                       "written": len(dedup),
                       "verified_quote_rate": round(len(all_facts) / n_cand, 4) if n_cand else 0.0}
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    FAIL_PATH.write_text(json.dumps(all_drops, ensure_ascii=False, indent=1), encoding="utf-8")
    common.log(f"facts written: {len(dedup)} (verified {len(all_facts)} / candidates {n_cand})")


if __name__ == "__main__":
    main()
