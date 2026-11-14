"""Stage 6b: ink-vs-text check for books with no PDF text layer (vn_van_hoa is pure scan).

For every page we measure the fraction of dark pixels in the rendered PNG. A page that is
ink-heavy but has almost no transcribed text means the vision pass dropped content. This is the
only coverage signal available when the PDF has no text layer.

Usage: python inkcheck.py  -> writes inkcheck.json, prints flagged pages
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

from PIL import Image  # noqa: E402

INK_MIN = 0.12       # a normal dense page of running text sits around 0.12-0.22
CHARS_MAX = 250      # ...with a good transcription this many characters or more


def ink_ratio(p: Path) -> float:
    im = Image.open(p).convert("L")
    im.thumbnail((600, 900))
    hist = im.histogram()
    dark = sum(hist[:128])
    total = im.width * im.height
    return dark / max(total, 1)


def main() -> int:
    out = {}
    for slug in common.BOOKS:
        flagged = []
        for png in sorted((common.PAGES / slug).glob("page-*.png")):
            n = int(png.stem.split("-")[1])
            md = common.page_md(slug, n)
            chars = len(md.read_text(encoding="utf-8").strip()) if md.exists() else 0
            r = ink_ratio(png)
            if r > INK_MIN and chars < CHARS_MAX:
                flagged.append({"page": n, "ink": round(r, 3), "chars": chars})
        out[slug] = flagged
        print(f"{slug}: flagged={len(flagged)} {flagged[:8]}", flush=True)
    (common.DATA_WORK / "inkcheck.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
