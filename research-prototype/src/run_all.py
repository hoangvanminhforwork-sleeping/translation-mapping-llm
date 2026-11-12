"""One entry point for every experiment. Idempotent: re-running skips finished work.

    python -m src.run_all data                      # validate + write the 100 pairs
    python -m src.run_all copy                      # 2k copy test, all four models
    python -m src.run_all copy --models vit5-base   # one model
    python -m src.run_all ppl                       # pseudo-perplexity, dish vs rest
    python -m src.run_all translate                 # zero-shot en2vi / vi2en, 100 pairs
    python -m src.run_all all                       # everything, in priority order
    python -m src.run_all all --force               # recompute from scratch

Every stage prints its own verification line; nothing is declared done by exit code.
"""

from __future__ import annotations

import argparse
import time

from . import config as C
from . import copy_eval, data, score


def _stage(name: str, fn, *a, **kw):
    t0 = time.time()
    print(f"\n=== {name} ===")
    out = fn(*a, **kw)
    print(f"--- {name} done in {time.time() - t0:.1f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data", "copy", "ppl", "translate", "all"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    keys = args.models or C.MODEL_ORDER

    if args.stage in ("data", "all"):
        _stage("data", data.build, force=True)

    if args.stage in ("copy", "all"):
        for k in keys:
            _stage(f"copy {k}", copy_eval.run_copy, k, force=args.force)
        _stage("copy rows", lambda: [copy_eval.write_rows(k) for k in keys])
        _stage("copy summary", copy_eval.write_summary, keys)

    if args.stage in ("ppl", "all"):
        for k in keys:
            _stage(f"ppl {k}", score.perplexity, k, force=args.force)
        _stage("ppl summary", score.ppl_summary)

    if args.stage in ("translate", "all"):
        # base<->base, large<->large: target language decides which model runs.
        plan = [("vit5-base", "en2vi"), ("flan-t5-base", "vi2en"), ("vit5-large", "en2vi"), ("flan-t5-large", "vi2en")]
        for model_key, direction in plan:
            if model_key in keys:
                _stage(f"translate {model_key} {direction}", score.run_translate, model_key, direction, force=args.force)
        _stage("bleu summary", score.translate_summary)


if __name__ == "__main__":
    main()
