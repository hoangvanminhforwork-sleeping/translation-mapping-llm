"""One entry point. Idempotent and resumable.

    python -m src.run_all data                 # 50k train pairs + fixed 2k eval set
    python -m src.run_all train --models vit5-base
    python -m src.run_all eval  --models vit5-base
    python -m src.run_all compare
    python -m src.run_all all --models vit5-base flan-t5-base
"""

from __future__ import annotations

import argparse
import time

from . import config as C
from . import data, eval_copy, train


def stage(name: str, fn, *a, **kw):
    t0 = time.time()
    print(f"\n=== {name} ===", flush=True)
    out = fn(*a, **kw)
    print(f"--- {name} finished in {time.time() - t0:.1f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["data", "train", "eval", "compare", "all"])
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    keys = args.models or C.ORDER

    if args.stage in ("data", "all"):
        stage("data", data.build_train, force=True)
        stage("eval set", data.build_eval, force=True)

    if args.stage in ("train", "all"):
        for k in keys:
            stage(f"train {k}", train.train, k, force=args.force)

    if args.stage in ("eval", "all"):
        for k in keys:
            if (C.ADAPTERS / k / "adapter_config.json").exists():
                stage(f"generate {k}", eval_copy.generate_all, k, force=args.force)
                stage(f"score {k}", eval_copy.score, k)
            else:
                print(f"skip {k}: no adapter")

    if args.stage in ("compare", "all"):
        stage("compare", eval_copy.compare)


if __name__ == "__main__":
    main()
