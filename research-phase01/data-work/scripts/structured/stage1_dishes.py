"""Stage 1: extract one Vietnamese recipe per TOC section -> structured/vn_dishes.csv

Method
------
1. Segments come from segments.py logic (TOC page numbers delimit the sections).
2. One LLM call per segment with a strict JSON-only prompt (temperature 0, no thinking).
3. Every row is validated: the three array columns must `json.loads` into non-empty
   lists of non-empty strings; `title` must be a verbatim substring of the section text.
4. Failures are recorded in structured/failures_dishes.json (not silently dropped).

Idempotent: responses are cached per (model, prompt) in structured/cache, so a rerun
is $0 and rebuilds the CSV from cache. `--force` re-calls the API.
"""
from __future__ import annotations

import argparse, csv, io, json, pathlib, re, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common

SLUG = "vn_mon_an_truyen_thong"
OUT_CSV = common.OUT_DIR / "vn_dishes.csv"
FAIL_PATH = common.OUT_DIR / "failures_dishes.json"

SYSTEM = (
    "Bạn là bộ trích xuất dữ liệu ẩm thực chính xác tuyệt đối. "
    "Chỉ trả về JSON hợp lệ, không giải thích, không markdown."
)

USER_TMPL = """Dưới đây là nguyên văn (OCR) một công thức món ăn Việt Nam trong sách.

TÊN MÓN (in hoa như trong sách): {title}
SỐ TRANG: {pages}

VĂN BẢN:
<<<
{text}
>>>

Hãy trả về DUY NHẤT một đối tượng JSON:
{{
  "ingredients": ["<một dòng nguyên liệu như in trong sách, gồm cả định lượng>"],
  "directions": ["<một bước làm, theo đúng thứ tự in trong sách>"],
  "NER": ["<thực phẩm xuất hiện trong tên món>"]
}}

QUY TẮC BẮT BUỘC:
1. KHÔNG dịch. Giữ nguyên tiếng Việt, giữ nguyên dấu, giữ nguyên chữ như trong văn bản.
2. ingredients: mỗi phần tử là MỘT nguyên liệu kèm định lượng như in. Nếu sách in thành
   danh sách gạch đầu dòng, giữ đúng từng dòng. Nếu nguyên liệu chỉ nằm trong một câu văn
   (ví dụ "Tẩm ướp với: 3 muỗng nước mắm + 1 muỗng đường"), hãy tách thành từng nguyên liệu
   riêng, giữ nguyên cách gọi và định lượng trong câu. TUYỆT ĐỐI không thêm nguyên liệu
   không có trong văn bản; không thêm nguyên liệu mà bạn suy đoán.
3. directions: mỗi phần tử là một bước nấu, theo thứ tự trong sách. Bỏ các đoạn bình luận,
   quảng cáo thương hiệu, dẫn giải lịch sử; giữ các lưu ý ảnh hưởng đến cách nấu.
   Mỗi bước phải là câu ĐẦY ĐỦ như in, không cắt cụt: nếu câu vừa nêu nguyên liệu vừa nêu
   thao tác thì giữ cả câu. Không lặp lại nhãn bước (A., B., 1., 2.) hai lần trong cùng câu.
4. NER: chỉ các thực phẩm (món, nguyên liệu chính) có trong TÊN MÓN, viết thường, ví dụ
   ["cá đuối", "cà ri"].
5. Không bịa. Nếu văn bản không có nguyên liệu rõ ràng, trả mảng rỗng cho trường đó.
6. Chỉ JSON, không thêm chữ nào ngoài JSON."""


def build_segments():
    pages = common.load_pages(SLUG)
    hdr = re.compile(r"^(M[ÓO]N ĂN VIỆT NAM TRUY[NỀN]N TH[ỐÔ]NG|Nhi[ềe]u tác gi[ảa])\s+\d+\s*$")
    foot = re.compile(r"^(https?://\S+|_+\s*)$")

    def clean(txt):
        keep = []
        for line in txt.splitlines():
            s = line.strip()
            if hdr.match(s) or foot.match(s):
                continue
            if len(s) < 70 and re.search(r"\d{1,3}$", s) and "VIỆT NAM TRUY" in s.upper():
                continue
            keep.append(s)
        return "\n".join(keep).strip()

    toc = []
    for pg in range(1, 8):
        for line in pages[pg].splitlines():
            m = re.match(r"^(.+?)\.{3,}\s*(\d{1,3})\s*$", line.strip())
            if m:
                toc.append((m.group(1).strip(), int(m.group(2))))
    last = max(pages)
    segs = []
    for i, (t, p) in enumerate(toc):
        end = toc[i + 1][1] - 1 if i + 1 < len(toc) else last
        body = "\n\n".join(clean(pages[x]) for x in range(p, end + 1) if x in pages)
        segs.append({"idx": i, "toc_title": t, "page": p, "pages": list(range(p, end + 1)), "text": body})
    return segs


LABELS = {"NGUYÊN LIỆU", "VẬT LIỆU", "CÁCH LÀM", "THỰC HÀNH", "GHI CHÚ", "THƯỞNG THỨC",
          "DỌN ĂN", "VẬT LIỆU - THỰC HÀNH", "VẬT LIỆU – THỰC HÀNH", "MỤC LỤC", "LỜI NÓI ĐẦU"}


def guess_title(text: str) -> str | None:
    for line in text.splitlines()[:8]:
        s = line.strip().lstrip("#").strip()
        if len(s) < 3 or len(s) > 90:
            continue
        if re.match(r"^(\d+[./)]|[-*+•]|[A-Z]\.)\s", s) or s.endswith(":"):
            continue
        if s in LABELS or s.rstrip(":").strip() in LABELS:
            continue
        letters = [c for c in s if c.isalpha()]
        if len(letters) < 4 or sum(c.isupper() for c in letters) / len(letters) < 0.6:
            continue
        return s
    return None


def valid_str_list(v) -> bool:
    return isinstance(v, list) and len(v) > 0 and all(isinstance(x, str) and x.strip() for x in v)


def run_one(seg, force: bool):
    title = guess_title(seg["text"]) or seg["toc_title"]
    base_user = USER_TMPL.format(title=title, pages=f"{seg['pages'][0]}-{seg['pages'][-1]}",
                                 text=seg["text"])
    last = None
    mt = 1800 if len(seg["text"]) <= 1800 else min(4000, 900 + len(seg["text"]) // 2)
    for attempt in range(2):
        user = base_user
        if attempt == 1:
            user += ("\n\nLƯU Ý: câu trả lời trước KHÔNG hợp lệ (" + (last or "?") +
                     "). Hãy trả lại đúng một đối tượng JSON với 3 khóa ingredients, "
                     "directions, NER, mỗi khóa là mảng chuỗi KHÔNG RỖNG. Trả lời ngắn gọn.")
            mt = max(mt, 3000)
        try:
            content, usage = common.chat("dishes", SYSTEM, user, max_tokens=mt,
                                         tag=f"dish-{seg['idx']}", force=force)
        except Exception as e:  # noqa: BLE001
            last = f"api:{e}"
            continue

        try:
            obj = common.parse_json_block(content)
        except Exception:
            last = "unparseable_json"
            continue

        # tolerate a wrapping list / alternate casing of the keys
        if isinstance(obj, list):
            obj = next((o for o in obj if isinstance(o, dict) and "directions" in
                        [str(k).lower() for k in o]), None)
        if not isinstance(obj, dict):
            last = "not_a_json_object"
            continue
        low = {str(k).lower(): v for k, v in obj.items()}
        ing = low.get("ingredients") or low.get("nguyên liệu")
        dirs = low.get("directions") or low.get("steps")
        ner = low.get("ner")
        bad = [k for k, v in (("ingredients", ing), ("directions", dirs), ("NER", ner))
               if not valid_str_list(v)]
        if bad:
            last = "invalid:" + ",".join(bad)
            continue
        if common.norm_ws(title) not in common.norm_ws(seg["text"]):
            last = "title_not_in_text"
            continue
        return {"idx": seg["idx"], "page": seg["page"], "title": title, "ok": True,
                "ingredients": ing, "directions": dirs, "NER": ner,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0)}

    return {"idx": seg["idx"], "page": seg["page"], "title": title, "ok": False,
            "reason": last or "unknown"}


def write_csv(rows):
    common.OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(["id", "title", "ingredients", "directions", "link", "source", "NER"])
        for r in rows:
            w.writerow([
                r["id"], r["title"],
                json.dumps(r["ingredients"], ensure_ascii=False),
                json.dumps(r["directions"], ensure_ascii=False),
                r["link"], r["source"],
                json.dumps(r["NER"], ensure_ascii=False),
            ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore cache, re-call the API")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    if OUT_CSV.exists() and not args.force and not any(common.CACHE_DIR.glob("*.json")):
        print("nothing to do; csv exists and cache empty")
        return

    segs = build_segments()
    common.log(f"segments: {len(segs)}")

    def safe(s):
        try:
            return run_one(s, args.force)
        except Exception as e:  # noqa: BLE001
            return {"idx": s["idx"], "page": s["page"], "title": "?", "ok": False,
                    "reason": f"exception:{e}"}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(safe, segs))

    ok = [r for r in results if r["ok"]]
    fails = [r for r in results if not r["ok"]]
    rows = []
    for i, r in enumerate(ok, start=1):
        rows.append({
            "id": f"vn-{i:04d}",
            "title": r["title"],
            "ingredients": r["ingredients"],
            "directions": r["directions"],
            "link": f"research-phase01/data-work/markdown/{SLUG}.md#p{r['page']}",
            "source": SLUG,
            "NER": r["NER"],
        })
    write_csv(rows)
    FAIL_PATH.write_text(json.dumps(fails, ensure_ascii=False, indent=1), encoding="utf-8")
    common.log(f"ok={len(ok)} fail={len(fails)} -> {OUT_CSV.name}")
    for f in fails[:10]:
        common.log(f"  FAIL idx={f['idx']} p={f['page']} {f['title'][:40]!r} {f['reason']}")


if __name__ == "__main__":
    main()
