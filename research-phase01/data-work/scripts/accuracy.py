"""Stage 6: measure OCR fidelity against the PDF's own text layer where one exists.

Only the three English PDFs carry a clean Unicode text layer. The two Vietnamese PDFs cannot be
scored this way: vn_mon_an's layer is legacy VNI-mojibake ("MOÁN ÙN" for "MÓN ĂN") and
vn_van_hoa has no text layer at all (scanned). For those, quality is judged by eye only.

Metric: after collapsing whitespace and stripping page-number/footer noise, we compute
character-level similarity (difflib ratio) and word-level F1 on the whole page text.

Usage: python accuracy.py  -> writes accuracy.json, prints a table
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

import pymupdf  # noqa: E402

SCORABLE = ["us_american_cuisine", "us_sioux_chef", "us_food_younger_land"]
FOOTER = re.compile(r"(More Free Books on Bookey|Scan to Download|Listen summary|"
                    r"More Book Summary|View on Bookey Website)", re.I)


def norm(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = FOOTER.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def words(s: str) -> list[str]:
    return re.findall(r"[a-z']+", s)


def main() -> int:
    out = {}
    for slug in SCORABLE:
        doc = pymupdf.open(common.book_path(slug))
        ratios, f1s, n_txt = [], [], 0
        worst = []
        for n in range(1, doc.page_count + 1):
            layer = doc[n - 1].get_text()
            md = common.page_md(slug, n)
            if not (md.exists() and md.stat().st_size > 0):
                continue
            if len(norm(layer)) < 60:      # image-only page in the PDF: nothing to score
                continue
            got = norm(md.read_text(encoding="utf-8"))
            ref = norm(layer)
            r = difflib.SequenceMatcher(None, ref, got).ratio()
            rw = difflib.SequenceMatcher(None, words(ref), words(got)).ratio()
            ratios.append(r)
            f1s.append(rw)
            n_txt += 1
            worst.append((round(r, 3), n))
        doc.close()
        worst.sort()
        out[slug] = {
            "pages_scored": n_txt,
            "mean_char_similarity": round(sum(ratios) / len(ratios), 4) if ratios else None,
            "mean_word_similarity": round(sum(f1s) / len(f1s), 4) if f1s else None,
            "min_char_similarity": worst[0][0] if worst else None,
            "worst_pages": worst[:5],
        }
        print(f"{slug}: scored={n_txt} char_sim={out[slug]['mean_char_similarity']} "
              f"word_sim={out[slug]['mean_word_similarity']} "
              f"min={out[slug]['min_char_similarity']} worst={worst[:3]}", flush=True)
    (common.DATA_WORK / "accuracy.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
