"""Stage 3d: block-guided re-transcription, for pages the whole-page pass under-read.

Band crops are unsafe: on an empty strip the model invents plausible text (seen on
vn_mon_an p64, which produced an imaginary advert headline). So instead of guessing geometry we
ask the PDF where its text blocks are, crop each block's bbox at 400 DPI, OCR just that crop,
and concatenate block by block. Every pixel the model sees is known to contain text.

Requires a PDF text layer (any encoding — we only use the bounding boxes, never the characters).

Usage: python redo_blocks.py --book SLUG --page N [--dpi 400] [--dry-run]
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

BLOCK_PROMPT = ("Transcribe all printed text in this image, line by line, in the original "
                "language. Preserve Vietnamese diacritics exactly. Output only the text, "
                "no commentary.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--page", type=int, required=True)
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    slug, n = args.book, args.page
    key = common.load_key()
    doc = pymupdf.open(common.book_path(slug))
    page = doc[n - 1]
    blocks = [b for b in page.get_text("blocks")
              if b[6] == 0 and len(re.findall(r"\S", b[4])) >= 2]
    blocks.sort(key=lambda b: (round(b[1]), b[0]))
    common.SCRATCH.mkdir(parents=True, exist_ok=True)

    out: list[str] = []
    usd = 0.0
    pt = ct = 0
    for i, b in enumerate(blocks):
        clip = pymupdf.Rect(b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2) & page.rect
        pix = page.get_pixmap(dpi=args.dpi, colorspace="rgb", alpha=False, clip=clip)
        p = common.SCRATCH / f"blk_{slug}_{n:04d}_{i}.png"
        p.write_bytes(pix.tobytes("png"))
        res = common.call_with_retry(key, p, prompt=BLOCK_PROMPT)
        usd += res["usd"]
        pt += res["prompt_tokens"]
        ct += res["completion_tokens"]
        out.append(res["text"].strip())
        p.unlink(missing_ok=True)
    doc.close()

    text = "\n".join(x for x in out if x).strip()
    md = common.page_md(slug, n)
    old = md.read_text(encoding="utf-8").strip() if md.exists() else ""
    kept = len(text) > len(old)
    if kept and not args.dry_run:
        md.parent.mkdir(parents=True, exist_ok=True)
        tmp = md.with_name(md.name + ".tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(md)
    common.append_log({"book": slug, "page": n, "status": "redo_blocks", "kept": kept,
                       "chars_before": len(old), "chars_after": len(text),
                       "blocks": len(blocks), "prompt_tokens": pt,
                       "completion_tokens": ct, "usd": round(usd, 8),
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
    common.write_cost_json()
    print(f"{slug} p{n}: blocks={len(blocks)} kept={kept} {len(old)} -> {len(text)} chars "
          f"${usd:.6f}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
