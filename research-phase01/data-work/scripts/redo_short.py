"""Stage 3b: re-transcribe pages whose first pass looked short, at a higher DPI with PROMPT2.

A page is re-OCR'd only if the new result is strictly longer than what is on disk, so a bad
retry can never destroy a good transcription.

Usage:
  python redo_short.py                  # auto-detect short pages via the PDF text layer
  python redo_short.py --book X --page N
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

import pymupdf  # noqa: E402


def wordcount(s: str) -> int:
    return len(re.findall(r"[^\s]+", s.replace("\u00a0", " ")))


def detect() -> list[tuple[str, int, str]]:
    """Pages where the vision pass kept far fewer words than the PDF text layer holds."""
    flagged: list[tuple[str, int, str]] = []
    for slug in common.BOOKS:
        doc = pymupdf.open(common.book_path(slug))
        for n in range(1, doc.page_count + 1):
            layer_words = wordcount(doc[n - 1].get_text())
            md = common.page_md(slug, n)
            if not (md.exists() and md.stat().st_size > 0):
                continue
            got = md.read_text(encoding="utf-8")
            if layer_words >= 15 and wordcount(got) < 0.6 * layer_words:
                flagged.append((slug, n,
                                f"layer={layer_words} words, ocr={wordcount(got)} words"))
        doc.close()
    return flagged


def redo(slug: str, n: int, dpi: int = 240) -> dict:
    key = common.load_key()
    doc = pymupdf.open(common.book_path(slug))
    png = common.render_page(doc, slug, n, dpi=dpi, force=True)
    doc.close()
    res = common.call_with_retry(key, png, prompt=common.PROMPT2)
    text = res["text"].strip()
    md = common.page_md(slug, n)
    old = md.read_text(encoding="utf-8").strip() if md.exists() else ""
    kept = len(text) > len(old)
    if kept:
        tmp = md.with_name(md.name + ".tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(md)
    common.append_log({"book": slug, "page": n, "status": "redo",
                       "kept": kept, "dpi": dpi, "prompt_version": common.PROMPT2_VERSION,
                       "chars_before": len(old), "chars_after": len(text),
                       "prompt_tokens": res["prompt_tokens"],
                       "completion_tokens": res["completion_tokens"],
                       "usd": round(res["usd"], 8),
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return {"book": slug, "page": n, "kept": kept, "chars_before": len(old),
            "chars_after": len(text), "usd": res["usd"], "text": text}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book")
    ap.add_argument("--page", type=int)
    ap.add_argument("--dpi", type=int, default=240)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.book and args.page:
        targets = [(args.book, args.page, "explicit")]
    else:
        targets = detect()
    if not targets:
        print("no short pages detected")
        return 0
    for slug, n, why in targets:
        print(f"redo {slug} p{n} ({why})", flush=True)
        if args.dry_run:
            continue
        r = redo(slug, n, dpi=args.dpi)
        print(f"   kept={r['kept']} {r['chars_before']} -> {r['chars_after']} chars "
              f"${r['usd']:.6f}", flush=True)
        print("   " + r["text"][:200].replace("\n", " / "), flush=True)
    common.write_cost_json()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
