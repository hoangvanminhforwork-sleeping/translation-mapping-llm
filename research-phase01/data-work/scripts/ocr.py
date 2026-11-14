"""Stage 3: vision-OCR every page with qwen/qwen3.7-flash, 6 in parallel.

Idempotent:
  * a page whose ocr/<slug>/page-NNNN.md already exists and is non-empty is skipped (costs $0);
  * `--force` redoes everything.

Cost guard: stop submitting new work as soon as the cumulative USD in ocr_log.jsonl passes
$0.50, and drop a marker in agent-work/dataworker/STOPPED.md.

Usage: python ocr.py [--force] [--workers N] [--book SLUG ...] [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

import pymupdf  # noqa: E402

_lock = threading.Lock()
_state = {"usd": 0.0, "stopped": False, "done": 0, "fail": 0, "blank": 0, "trunc": 0}


def tasks(books: list[str], force: bool, limit: int | None) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for slug in books:
        doc = pymupdf.open(common.book_path(slug))
        n_pages = doc.page_count
        doc.close()
        for n in range(1, n_pages + 1):
            md = common.page_md(slug, n)
            if not force and md.exists() and md.stat().st_size > 0:
                continue
            out.append((slug, n))
    if limit is not None:
        out = out[:limit]
    return out


def do_page(key: str, slug: str, n: int, force: bool) -> dict:
    md = common.page_md(slug, n)
    png = common.page_png(slug, n)
    if not png.exists() or png.stat().st_size == 0:
        doc = pymupdf.open(common.book_path(slug))
        common.render_page(doc, slug, n)
        doc.close()

    t0 = time.time()
    try:
        res = common.call_with_retry(key, png)
        if res["truncated"]:
            res = common.call_with_retry(key, png, max_tokens=16000)
            _state["trunc"] += 1
        text = res["text"]
        if not text:
            raise RuntimeError("empty transcription")
    except Exception as e:  # noqa: BLE001
        rec = {"book": slug, "page": n, "status": "failed", "error": str(e)[:300],
               "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0,
               "elapsed_s": round(time.time() - t0, 2),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with _lock:
            _state["fail"] += 1
            common.append_log(rec)
        return rec

    md.parent.mkdir(parents=True, exist_ok=True)
    tmp = md.with_name(md.name + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(md)

    status = "ok"
    if text.strip() in ("[BLANK]", "[IMAGE-ONLY]"):
        status = text.strip().strip("[]").lower()
    rec = {"book": slug, "page": n, "status": status,
           "prompt_tokens": res["prompt_tokens"], "completion_tokens": res["completion_tokens"],
           "usd": round(res["usd"], 8), "chars": len(text),
           "truncated": res["truncated"],
           "elapsed_s": round(time.time() - t0, 2),
           "prompt_version": common.PROMPT_VERSION,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with _lock:
        _state["usd"] += res["usd"]
        _state["done"] += 1
        if status == "ok":
            pass
        else:
            _state["blank"] += 1
        common.append_log(rec)
        if (_state["done"] + _state["fail"]) % 10 == 0:
            print(f"  ... {_state['done']} ok / {_state['fail']} failed, "
                  f"session ${_state['usd']:.4f}, {time.strftime('%H:%M:%S')}", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--book", nargs="*", default=list(common.BOOKS))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    common.ensure_dirs()

    key = common.load_key()
    todo = tasks(args.book, args.force, args.limit)
    if not todo:
        print("nothing to do (all pages already transcribed; use --force to redo)")
        common.write_cost_json()
        return 0

    already = common.total_cost()
    print(f"[{time.strftime('%H:%M:%S')}] {len(todo)} pages to OCR "
          f"| cumulative spend before run ${already:.4f} | workers={args.workers}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        it = iter(todo)
        # Prime the pool, then refill one slot at a time so the guard can halt quickly.
        for slug, n in it:
            futs[pool.submit(do_page, key, slug, n, args.force)] = (slug, n)
            if len(futs) >= args.workers:
                break
        while futs:
            done, _pending = wait(set(futs), return_when=FIRST_COMPLETED)
            for f in done:
                f.result()
                futs.pop(f, None)
            if not _state["stopped"] and already + _state["usd"] > common.COST_LIMIT_USD:
                _state["stopped"] = True
                common.write_stop(
                    f"cumulative cost ${already + _state['usd']:.4f} exceeded "
                    f"${common.COST_LIMIT_USD:.2f}")
                print("!! cost guard tripped, draining in-flight requests", flush=True)
            if _state["stopped"]:
                continue
            while len(futs) < args.workers:
                nxt = next(it, None)
                if nxt is None:
                    break
                slug, n = nxt
                futs[pool.submit(do_page, key, slug, n, args.force)] = (slug, n)

    common.write_cost_json({"last_run_pages": _state["done"] + _state["fail"],
                            "last_run_usd": round(_state["usd"], 6)})
    print(f"[{time.strftime('%H:%M:%S')}] run finished in {time.time()-t0:.0f}s | "
          f"ok={_state['done']} blank/image-only={_state['blank']} failed={_state['fail']} "
          f"truncated-retried={_state['trunc']} | session ${_state['usd']:.4f} | "
          f"cumulative ${common.total_cost():.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
