"""Smoke test: render page 1 of vn_mon_an_truyen_thong and inspect the transcription.

Usage: python smoke.py [dpi]
Prints the returned text to stdout so a human can eyeball Vietnamese diacritics.
Does NOT write into ocr/ (only agent-work/dataworker + pages/ for the png).
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

import fitz  # noqa: E402

VI_WORDS = ["bánh", "chưng", "phở", "ẩm thực", "món", "ăn", "người", "Việt", "Nam", "văn hóa"]
# Characters that only appear in properly-decomposed Vietnamese text.
VI_CHARS = set("áàảãạăâđêôơưíìỉĩịóòỏõọúùủũụýỳỷỹỵÁÀẢÃẠĂÂĐÊÔƠƯ")


def mangled_score(text: str) -> int:
    """Count '?' replacement heuristics: standalone ? inside words, and '?' density."""
    return len(re.findall(r"\w\?\w", text))


def main() -> int:
    dpi = int(sys.argv[1]) if len(sys.argv) > 1 else common.DPI
    common.ensure_dirs()
    slug = "vn_mon_an_truyen_thong"
    doc = fitz.open(common.book_path(slug))
    png = common.render_page(doc, slug, 1, dpi=dpi, force=(dpi != common.DPI))
    doc.close()
    print(f"[{time.strftime('%H:%M:%S')}] rendered {png.name} at {dpi} dpi, "
          f"{png.stat().st_size/1024:.0f} KiB")

    key = common.load_key()
    res = common.call_with_retry(key, png)
    text = res["text"]
    print(f"[{time.strftime('%H:%M:%S')}] tokens in={res['prompt_tokens']} "
          f"out={res['completion_tokens']} usd={res['usd']:.6f}")
    print(f"dpi={dpi} chars={len(text)} mangled_patterns={mangled_score(text)} "
          f"vi_chars={sum(1 for c in text if c in VI_CHARS)}")
    print("hit_words:", [w for w in VI_WORDS if w.lower() in text.lower()])
    print("=" * 70)
    print(text)
    print("=" * 70)
    common.append_log({"book": slug, "page": 1, "status": "smoke", "dpi": dpi,
                       "prompt_tokens": res["prompt_tokens"],
                       "completion_tokens": res["completion_tokens"],
                       "usd": round(res["usd"], 8), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
    common.write_cost_json()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
