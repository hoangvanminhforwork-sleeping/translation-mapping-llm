"""Stage 5: quality summary per book, for REPORT.md.

Responsibilities:
  * count pages with an ocr/*.md file, failures in ocr_log.jsonl, [BLANK]/[IMAGE-ONLY];
  * flag suspicious transcriptions (replacement chars, '?' inside words, very short output,
    truncated responses);
  * recompute measured USD from the log.

Usage: python summarize.py  -> prints a markdown table block and writes quality.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

import pymupdf  # noqa: E402

REPLACEMENT = "\ufffd"
MOJIBAKE_HINTS = ["Ã ", "â€", "Ã¢", "Ã´"]


def main() -> int:
    log_recs = []
    if common.LOG_PATH.exists():
        for line in common.LOG_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    log_recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    out = {}
    total_pages = 0
    for slug in common.BOOKS:
        doc = pymupdf.open(common.book_path(slug))
        n_pages = doc.page_count
        doc.close()
        total_pages += n_pages
        recs = [r for r in log_recs if r.get("book") == slug]
        fails = [r for r in recs if r.get("status") == "failed"]
        usd = sum(float(r.get("usd") or 0) for r in recs)
        blank = image_only = have = suspicious = 0
        suspicious_pages = []
        chars = 0
        for n in range(1, n_pages + 1):
            md = common.page_md(slug, n)
            if not (md.exists() and md.stat().st_size > 0):
                continue
            t = md.read_text(encoding="utf-8").strip()
            have += 1
            chars += len(t)
            if t == "[BLANK]":
                blank += 1
                continue
            if t == "[IMAGE-ONLY]":
                image_only += 1
                continue
            bad = 0
            bad += t.count(REPLACEMENT) * 5
            bad += len(re.findall(r"\w\?\w", t))
            bad += sum(t.count(h) for h in MOJIBAKE_HINTS)
            if len(t) < 40:
                bad += 3
            if bad >= 3:
                suspicious += 1
                suspicious_pages.append(n)
        out[slug] = {
            "pages_total": n_pages, "pages_with_md": have,
            "missing": n_pages - have,
            "blank": blank, "image_only": image_only,
            "suspicious": suspicious,
            "suspicious_pages": suspicious_pages[:40],
            "api_failures": len(fails),
            "failure_reasons": sorted({r.get("error", "")[:120] for r in fails})[:5],
            "truncated_retried": sum(1 for r in recs if r.get("truncated")),
            "mean_chars_per_page": round(chars / max(have, 1)),
            "usd": round(usd, 6),
        }

    tot = sum(v["usd"] for v in out.values())
    print("| book | pages | md | blank | img-only | suspicious | api-fail | mean chars | USD |")
    print("|---|---|---|---|---|---|---|---|---|")
    for slug, v in out.items():
        print(f"| {slug} | {v['pages_total']} | {v['pages_with_md']} | {v['blank']} | "
              f"{v['image_only']} | {v['suspicious']} | {v['api_failures']} | "
              f"{v['mean_chars_per_page']} | {v['usd']:.4f} |")
    print(f"| **TOTAL** | {total_pages} | "
          f"{sum(v['pages_with_md'] for v in out.values())} | "
          f"{sum(v['blank'] for v in out.values())} | "
          f"{sum(v['image_only'] for v in out.values())} | "
          f"{sum(v['suspicious'] for v in out.values())} | "
          f"{sum(v['api_failures'] for v in out.values())} | - | {tot:.4f} |")
    (common.DATA_WORK / "quality.json").write_text(
        json.dumps({"books": out, "total_usd": round(tot, 6),
                    "total_pages": total_pages}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
