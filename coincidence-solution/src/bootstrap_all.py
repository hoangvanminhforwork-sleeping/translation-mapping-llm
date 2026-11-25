"""Paired bootstrap of every arm against the published map, from the persisted hypotheses.

CPU-only: reads `agent-work/hyps/{arm}.json` (max_new=64 generations) and re-runs the 1 000-resample
paired bootstrap, so the CI table in REPORT.md can be regenerated without touching the GPU.

Usage
    python -m src.bootstrap_all
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import cc
from .fit_behaviour import boot_delta

sys.stdout.reconfigure(encoding="utf-8")


def main(only: list[str]):
    H = cc.WORK / "hyps"
    refs = json.loads((H / "refs.json").read_text(encoding="utf-8"))
    base = json.loads((H / "ridge_full.json").read_text(encoding="utf-8"))
    out_p = cc.RESULTS / "bootstrap.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    for p in sorted(H.glob("*.json")):
        if p.stem in ("refs", "ridge_full"):
            continue
        if only and p.stem not in only:
            continue
        h = json.loads(p.read_text(encoding="utf-8"))[:len(refs)]
        out[p.stem] = {"bleu": boot_delta(h, base, refs, "bleu", n=200)}
        out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        b = out[p.stem]["bleu"]
        print(f"{p.stem:18s} dBLEU {b['delta']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}] "
              f"p>0 {b['p_gt0']:.3f}", flush=True)
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote results/bootstrap.json")


if __name__ == "__main__":
    main(sys.argv[1:])
