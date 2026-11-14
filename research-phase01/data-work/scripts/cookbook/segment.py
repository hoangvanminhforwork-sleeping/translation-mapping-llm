"""Segment the OCR'd cookbook into recipe blocks.

Discovered structure (numbers in structured-cookbook/REPORT.md):
  * lines 0..~260   : multi-page chapter-level TOC, 87 entries, max printed page 1144
  * lines ~260..8300: theory (nutrition, cooking methods, flower arranging) - no recipes
  * line 8303       : first recipe header ("a, Nguyên liệu" of LUỘC RAU)
  * recipe layout   : [dish title] / [ingredient header] / ingredient lines /
                      [directions header] / steps
      ingredient header variants seen (normalised, diacritic-insensitive):
        "I. NGUYÊN LIỆU", "NGUYÊN LIỆU", "1. Nguyên liệu", "a, Nguyên liệu",
        OCR manglings of "I.": "L", "r", "h", "T", "!"  (e.g. "r NGUYÊN LIỆU:")
      directions header variants:
        "II. CÁCH THỰC HIỆN", "THỰC HIỆN", "Cách làm", "b. Cách thực hiện",
        "Cách chế biến", "Cách nấu", "Làm bánh"
  * two chapters (drinks ~86k, sorbet/ice-cream ~87.2k) print no headers at all:
    a numbered dish title, ingredient lines, then steps.  Those regions are split
    on short numbered dish-title lines (desert split).
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "data" / "Vietnamese" / "Kỹ thuật nấu ăn toàn tập.md"
OUT = ROOT / "agent-work" / "cookbook"

# OCR of the roman numeral "I" in headers: I L 1 l r h T !
PREFIX = r"(?:[a-dl]|i{1,3}|iv|v|[rht!]|\d{1,2})?"
ING_RE = re.compile(rf"^{PREFIX}\s*nguyen\s*lie[ue]?u?\b")
DIR_RE = re.compile(
    rf"^{PREFIX}\s*(?:cach\s*(?:lam|nau|thuc\s*hien|che\s*bien)"
    r"|thuc\s*hien|che\s*bien|lam\s*banh|lam)$")
NUMISH = re.compile(rf"^\s*{PREFIX}\s*[.,:;)\-*]")
PAGENO = re.compile(r"^[\d\s.,\-]{1,6}$")
# short numbered dish-title line, used only inside anchor deserts
TITLE_LINE = re.compile(r"^\s*(?:\d{1,2}|[A-Za-zÀ-ỹ])[.)]\s*\S.{0,38}$")
STEPWORD = re.compile(r"^(cach|thuc hien|chuan bi|so che|nau|chien|hap|luoc|xa|uop|tron|"
                      r"trinh bay|lam|pha|dun|nuong|sot|xot|ghi chu)")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("đ", "d").replace("Đ", "D").lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def upper_frac(s: str) -> float:
    a = [c for c in s if c.isalpha()]
    return sum(1 for c in a if c.isupper()) / len(a) if a else 0.0


def classify(lines: list[str]) -> tuple[list[int], list[int]]:
    ing, dr = [], []
    for i, l in enumerate(lines):
        s = l.strip()
        if not s or len(s) > 60:
            continue
        nz = norm(s)
        if s[0] in "-•*■+♦▪·":
            continue  # prose bullet ("- Nguyên liệu cần thiết..."), not a header
        head = upper_frac(s) >= 0.7 or bool(NUMISH.match(s))
        if not head:
            continue
        if ING_RE.match(nz):
            # allow a short trailing note such as "NGUYÊN LIỆU (dùng cho 4 người)"
            if re.match(r"^nguyen lieu\b", nz) and len(nz) > 40:
                continue
            ing.append(i)
        elif DIR_RE.match(nz):
            dr.append(i)
    return ing, dr


def build(lines: list[str]) -> list[dict]:
    ing, dr = classify(lines)
    first = ing[0]
    dr = [d for d in dr if d > first]

    blocks: list[dict] = []
    for k, gi in enumerate(ing):
        end = ing[k + 1] - 1 if k + 1 < len(ing) else len(lines) - 1
        # the dish title is printed above the ingredient header (1-3 lines up, page
        # numbers / blanks skipped); block starts there so the link points at it
        stop = blocks[-1]["ing"] if blocks else 0
        start = gi
        for j in range(gi - 1, max(stop, gi - 7), -1):
            s = lines[j].strip()
            # a printed dish title is short and does not end like a sentence; a
            # preceding step line ("Múc xúp ra dĩa ...") must not become the start
            if not s or PAGENO.match(s) or j in ing or j in dr:
                continue
            if len(s) > 70 or s.endswith((".", ",", ";", ":")):
                break
            start = j
            break
        blocks.append({"ing": gi, "start": start, "dir": None, "end": end})
    for b in blocks:
        for j in range(b["ing"] + 1, b["end"] + 1):
            if j in set(dr):
                b["dir"] = j
                break

    # --- desert pass: header-less chapters split on numbered dish titles ----
    extra: list[dict] = []
    for b in blocks:
        if b["end"] - b["start"] < 400:
            continue
        lo = b["dir"] + 1 if b["dir"] else b["start"] + 1
        cuts = []
        for j in range(lo, b["end"]):
            s = lines[j].strip()
            if not (4 <= len(s) <= 45) or not TITLE_LINE.match(s):
                continue
            if STEPWORD.match(norm(s)):
                continue
            numbered = bool(re.match(r"^\d{1,2}\s*[.)]\s*\S", s))
            nxt = next((lines[q].strip() for q in range(j + 1, min(j + 4, len(lines))) if lines[q].strip()), "")
            follows_list = nxt.startswith(("-", "■", "•", "*", "+", "♦", "5", "1", "2", "3"))
            if numbered or follows_list:
                cuts.append(j)
        if len(cuts) < 3:
            continue
        cuts = sorted(set(cuts))
        prev = b["end"]
        for c in reversed(cuts):
            extra.append({"ing": None, "start": c, "dir": None, "end": prev})
            prev = c - 1
        b["end"] = prev
    blocks.extend(extra)
    blocks.sort(key=lambda x: x["start"])

    def body(a: int, b: int) -> list[str]:
        return [l.strip() for l in lines[a:b + 1] if l.strip()]

    for b in blocks:
        if b["ing"] is not None:
            i_end = (b["dir"] - 1) if b["dir"] else b["end"]
            b["n_ing_lines"] = len(body(b["ing"] + 1, i_end))
        else:
            b["n_ing_lines"] = len(body(b["start"] + 1, (b["dir"] or b["end"])))
        b["n_dir_lines"] = len(body(b["dir"] + 1, b["end"])) if b["dir"] else 0
        b["n_lines"] = len(body(b["start"], b["end"]))
        b["title"] = lines[b["start"]].strip()
        b["text"] = "\n".join(body(b["start"], b["end"]))
    return blocks


if __name__ == "__main__":
    lines = SRC.read_text(encoding="utf-8").splitlines()
    blocks = build(lines)
    usable = [b for b in blocks if b["n_lines"] >= 3 and b["title"]]
    print("blocks:", len(blocks), "usable:", len(usable))
    print("ing+dir:", sum(1 for b in blocks if b["ing"] is not None and b["dir"] is not None),
          "ing only:", sum(1 for b in blocks if b["ing"] is not None and b["dir"] is None),
          "desert:", sum(1 for b in blocks if b["ing"] is None))
    chars = sum(len(b["text"]) for b in usable)
    print("chars:", chars, "~tokens", chars // 2)
    lens = sorted(b["n_lines"] for b in usable)
    print("non-blank lines/block: min", lens[0], "median", lens[len(lens) // 2],
          "p95", lens[int(len(lens) * .95)], "max", lens[-1])
    print("blocks >400 lines:", sum(1 for b in usable if b["n_lines"] > 400))
    (OUT / "blocks3.json").write_text(json.dumps(usable, ensure_ascii=False), encoding="utf-8")

