"""Stage 4: merge per-page OCR output into markdown/<slug>.md with `<!-- page N -->` separators.

Pages whose transcription is exactly [BLANK] are skipped. [IMAGE-ONLY] pages are kept as a
single marker line so a reader knows a figure sits there.

Usage: python merge.py [--book SLUG ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

import pymupdf  # noqa: E402


def merge(slug: str) -> dict:
    doc = pymupdf.open(common.book_path(slug))
    n_pages = doc.page_count
    doc.close()

    parts: list[str] = [f"# {slug}\n"]
    stats = {"book": slug, "pages_total": n_pages, "pages_merged": 0,
             "blank_skipped": 0, "image_only": 0, "missing": []}
    for n in range(1, n_pages + 1):
        md = common.page_md(slug, n)
        if not md.exists() or md.stat().st_size == 0:
            stats["missing"].append(n)
            continue
        text = md.read_text(encoding="utf-8").strip()
        if text == "[BLANK]":
            stats["blank_skipped"] += 1
            continue
        if text == "[IMAGE-ONLY]":
            stats["image_only"] += 1
        parts.append(f"\n<!-- page {n} -->\n\n{text}\n")
        stats["pages_merged"] += 1

    out = common.MARKDOWN / f"{slug}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")
    stats["out"] = str(out.relative_to(common.PROJECT_ROOT))
    stats["bytes"] = out.stat().st_size
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", nargs="*", default=list(common.BOOKS))
    args = ap.parse_args()
    common.ensure_dirs()
    all_stats = []
    for slug in args.book:
        s = merge(slug)
        all_stats.append(s)
        print(f"{slug}: merged {s['pages_merged']}/{s['pages_total']} pages "
              f"(blank={s['blank_skipped']} image-only={s['image_only']} "
              f"missing={len(s['missing'])}) -> {s['out']} ({s['bytes']/1024:.0f} KiB)",
              flush=True)
    (common.DATA_WORK / "merge_stats.json").write_text(
        json.dumps(all_stats, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
