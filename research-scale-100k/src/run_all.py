"""One entry point for the 100k mapping experiment. Idempotent: every stage skips finished work.

    python -m src.run_all build      # 100k pairs (sha256 in data/pairs_manifest.json)
    python -m src.run_all encode     # two 100k x 768 matrices (GPU, ~5 min on RTX 3050)
    python -m src.run_all map        # ridge sweep + metrics (CPU, ~1 min)
    python -m src.run_all bleu       # 3 generation runs on the 1000-sentence test split (GPU)
    python -m src.run_all all        # the four above, in order
"""

from __future__ import annotations

import argparse
import time

from . import bleu, build_pairs, encode, mapping


def stage(name: str, fn, *a, **kw):
    t0 = time.time()
    print(f"\n=== {name} ===", flush=True)
    out = fn(*a, **kw)
    print(f"--- {name} finished in {time.time() - t0:.1f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "encode", "map", "bleu", "all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="bleu: first N test sentences")
    args = ap.parse_args()

    if args.stage in ("build", "all"):
        stage("build pairs", build_pairs.build, force=args.force)
    if args.stage in ("encode", "all"):
        stage("encode", encode.build, force=args.force)
    if args.stage in ("map", "all"):
        stage("map", mapping.run)
    if args.stage in ("bleu", "all"):
        stage("bleu vi->en", bleu.run, limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
