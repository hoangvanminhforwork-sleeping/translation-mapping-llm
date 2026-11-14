"""Stage 0: segment vn_mon_an_truyen_thong.md into per-recipe sections using the TOC.

Writes agent-work/structurer/segments.json (scratch) with {idx, toc_title, page, pages[], title,
text}. Title heuristic = first ALL-CAPS line on the recipe's first page (running header/footer
removed); validated to be a substring of the page text.
"""
import sys, json, re, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common

SLUG = "vn_mon_an_truyen_thong"
pages = common.load_pages(SLUG)

HDR = re.compile(r"^(M[ÓO]N ĂN VIỆT NAM TRUY[NỀN]N TH[ỐÔ]NG|Nhi[ềe]u tác gi[ảa])\s+\d+\s*$")
FOOT = re.compile(r"^(https?://\S+|_+\s*)$")


def is_running_header(s: str) -> bool:
    """OCR-safe running-header test (accent variants tolerated)."""
    if HDR.match(s):
        return True
    if len(s) < 70 and re.search(r"\d{1,3}$", s):
        up = s.upper()
        if "VIỆT NAM TRUY" in up and up.startswith("M"):
            return True
        if up.startswith("NHIỀU TÁC GIẢ"):
            return True
    return False


def clean(txt: str) -> str:
    keep = []
    for line in txt.splitlines():
        s = line.strip()
        if is_running_header(s) or FOOT.match(s):
            continue
        keep.append(s)
    return "\n".join(keep).strip()


# TOC (pages 1-7): "<garbled title>....<page>"
toc = []
for pg in range(1, 8):
    for line in pages[pg].splitlines():
        m = re.match(r"^(.+?)\.{3,}\s*(\d{1,3})\s*$", line.strip())
        if m:
            toc.append({"toc_title": m.group(1).strip(), "page": int(m.group(2))})
assert all(toc[i]["page"] <= toc[i + 1]["page"] for i in range(len(toc) - 1)), "TOC not monotonic"

last_page = max(pages)
segs = []
for i, e in enumerate(toc):
    end = (toc[i + 1]["page"] - 1) if i + 1 < len(toc) else last_page
    seg_pages = list(range(e["page"], end + 1))
    body = "\n\n".join(clean(pages[p]) for p in seg_pages if p in pages)
    segs.append({"idx": i, "toc_title": e["toc_title"], "page": e["page"], "pages": seg_pages, "text": body})

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
        if len(letters) < 4:
            continue
        if sum(c.isupper() for c in letters) / len(letters) < 0.6:
            continue
        return s
    return None


missing = []
for s in segs:
    t = guess_title(s["text"])
    s["title"] = t
    if t is None:
        missing.append(s["idx"])

common.SCRATCH.mkdir(parents=True, exist_ok=True)
(common.SCRATCH / "segments.json").write_text(json.dumps(segs, ensure_ascii=False), encoding="utf-8")

print(f"TOC recipe entries: {len(toc)}")
print(f"segments: {len(segs)}  with title: {len(segs) - len(missing)}  missing: {len(missing)} -> {missing}")
print("page span distribution:",
      sorted({len(s['pages']) for s in segs}))
print("max segment chars:", max(len(s['text']) for s in segs),
      "median:", sorted(len(s['text']) for s in segs)[len(segs) // 2])
print("total chars:", sum(len(s['text']) for s in segs))
