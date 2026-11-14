"""Stage 2: render every page of every book to PNG at 180 DPI.

Idempotent: a page whose PNG already exists with non-zero size is skipped.
`--force` re-renders everything. PyMuPDF releases the GIL inside get_pixmap, so a thread
pool gives near-linear speed-up on scanned PDFs.

Usage: python render.py [--force] [--workers 6] [--book SLUG ...]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

import pymupdf  # noqa: E402


def render_book(slug: str, dpi: int, force: bool, workers: int) -> tuple[str, int, int, int, float]:
    # PyMuPDF objects are not thread-safe: give every worker its own Document handle.
    import threading

    path = str(common.book_path(slug))
    probe = pymupdf.open(path)
    n_pages = probe.page_count
    probe.close()
    t0 = time.time()
    tl = threading.local()

    def one(n: int) -> bool:
        doc = getattr(tl, "doc", None)
        if doc is None:
            doc = tl.doc = pymupdf.open(path)
        return common.render_page(doc, slug, n, dpi=dpi, force=force) is not None

    todo = [n for n in range(1, n_pages + 1)
            if force or not (common.page_png(slug, n).exists()
                             and common.page_png(slug, n).stat().st_size > 0)]
    skipped = n_pages - len(todo)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    doc = getattr(tl, "doc", None)
    if doc is not None:
        doc.close()
    return slug, n_pages, len(todo), skipped, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--book", nargs="*", default=list(common.BOOKS))
    ap.add_argument("--dpi", type=int, default=common.DPI)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    common.ensure_dirs()

    grand_t0 = time.time()
    for slug in args.book:
        slug, n_pages, made, skipped, dt = render_book(
            slug, args.dpi, args.force, args.workers)
        print(f"[{time.strftime('%H:%M:%S')}] {slug}: {n_pages} pages "
              f"(rendered={made} skipped={skipped}) in {dt:.1f}s "
              f"= {made/max(dt,1e-6):.1f} pages/s", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] done in {time.time()-grand_t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
