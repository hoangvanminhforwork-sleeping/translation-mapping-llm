"""COMET on the arms — does a learned metric agree with BLEU?

BLEU 2.5 -> 7.0 is far outside BLEU noise, but BLEU can reward fluent-but-unfaithful output (the
Oct-31 pilot's failure mode). COMET is reference-based and trained on human judgements, so it is the
cheapest independent check.

Writes incrementally (one entry per arm) so a killed process leaves usable partial results.

Usage
    python -m src.eval_comet [--n 500] [arm ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import cc

sys.stdout.reconfigure(encoding="utf-8")


def main(n: int, arms: list[str]):
    from comet import download_model, load_from_checkpoint
    path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(path)
    H = cc.WORK / "hyps"
    refs = json.loads((H / "refs.json").read_text(encoding="utf-8"))[:n]
    src = json.loads((cc.CACHE / "test" / "texts.json").read_text(encoding="utf-8"))["vi"][:n]
    out_p = cc.RESULTS / "comet.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {"n": n, "arms": {}}
    for arm in arms:
        p = H / f"{arm}.json"
        if not p.exists():
            continue
        hyp = json.loads(p.read_text(encoding="utf-8"))[:n]
        t0 = time.time()
        r = model.predict([{"src": s, "mt": m, "ref": rr} for s, m, rr in zip(src, hyp, refs)],
                          batch_size=8, gpus=1, progress_bar=False)
        out["arms"][arm] = {"comet": round(float(r.system_score), 4),
                            "sec": round(time.time() - t0, 1)}
        out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"{arm:16s} COMET {r.system_score:.4f}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("arms", nargs="*", default=["ridge_full", "ridge_recolor10", "ce_ridge",
                                                "ce_ridge_e6"])
    a = ap.parse_args()
    main(a.n, a.arms)
