"""Replication check: does the latent > string correlation gap survive on a *different* pool?

The main comparison used the frozen model's beam-8 pool. The LoRA-adapted model produces a different
pool (different candidates, different quality). Re-running the same two utilities on it tests whether
"latent agreement is the better scorer" is a property of the *signal* or of one particular pool.

Writes results/replication.json.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")

UTILS = ["chrF_bt", "lat_match_f1", "lat_match_f1_c", "lat_match_pre", "lat_cos_mean"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="feat_core_lora8.parquet")
    ap.add_argument("--comet", default="comet_core_lora8.parquet")
    ap.add_argument("--out", default="replication.json")
    a = ap.parse_args(argv)

    feat = pq.read_table(C.RESULTS / a.feat).to_pylist()
    com = pq.read_table(C.RESULTS / a.comet).to_pylist()
    cc = {(r["id"], r["rank"]): r["comet"] for r in com}
    ids = sorted({f["id"] for f in feat})
    by: dict[int, list[dict]] = {}
    for f in feat:
        if (f["id"], f["rank"]) in cc:
            by.setdefault(f["id"], []).append(f)
    for v in by.values():
        v.sort(key=lambda x: x["rank"])

    rows = [f for i in ids for f in by[i]]
    y = np.array([cc[(f["id"], f["rank"])] for f in rows], np.float32)
    out = {"n_rows": len(ids), "n_cands": len(rows), "spearman": {}}
    for k in UTILS:
        out["spearman"][k] = float(spearmanr(np.array([f[k] for f in rows], np.float32), y).statistic)
    out["delta_latent_minus_string"] = out["spearman"]["lat_match_f1"] - out["spearman"]["chrF_bt"]

    # top-1 accuracy of each utility, and the BLEU it buys, is computed by src.score on selections
    out["top1_acc"] = {}
    for k in ("chrF_bt", "lat_match_f1"):
        hit = 0
        for i in ids:
            cs = by[i]
            best = max(cc[(i, c["rank"])] for c in cs)
            pick = max(cs, key=lambda c: c[k])
            hit += cc[(i, pick["rank"])] >= best - 0.01
        out["top1_acc"][k] = hit / len(ids)

    (C.RESULTS / a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
