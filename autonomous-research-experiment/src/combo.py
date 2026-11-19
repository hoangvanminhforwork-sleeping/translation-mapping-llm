"""Can consensus and agreement be combined? (the last, cheapest question of the night)

MBR wins BLEU among reference-free rules (+0.19) and the round-trip signals lose (−0.55). They measure
different things: consensus = "what the model would say anyway", agreement = "what the source supports".
A convex combination of the two per-row z-scored utilities is the natural test of whether either
carries information the other lacks.

    u = (1 - w) * z(mbr_chrf) + w * z(latent_f1)

w = 0 reduces to MBR (sanity check), w = 1 to the latent arm.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pyarrow.parquet as pq
from sacrebleu.metrics import CHRF

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")
CHRF_M = CHRF()


def z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="pool_test_beam8.jsonl")
    ap.add_argument("--feat", default="feat_test_beam8.parquet")
    ap.add_argument("--weights", default="0,0.25,0.5,0.75,1.0")
    a = ap.parse_args(argv)

    rows = [json.loads(line) for line in open(C.RESULTS / a.pool, encoding="utf-8")]
    feats = pq.read_table(C.RESULTS / a.feat).to_pylist()
    by: dict[int, list[dict]] = {}
    for f in feats:
        by.setdefault(f["id"], []).append(f)
    for v in by.values():
        v.sort(key=lambda x: x["rank"])

    weights = [float(x) for x in a.weights.split(",")]
    picks = {w: [] for w in weights}
    for k, r in enumerate(rows):
        hyps = [c["hyp"] for c in r["cands"]]
        n = len(hyps)
        mbr = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    mbr[i] += CHRF_M.sentence_score(hyps[i], [hyps[j]]).score
        lat = np.array([f["lat_match_f1"] for f in by[r["id"]]], dtype=np.float64)
        zm, zl = z(mbr), z(lat)
        for w in weights:
            picks[w].append(int(np.argmax((1 - w) * zm + w * zl)))
        if k % 2000 == 0:
            print(f"  {k}/{len(rows)}", flush=True)

    for w in weights:
        name = f"A13_combo{str(w).replace('.', '')}"
        with open(C.RESULTS / f"sel_{name}.jsonl", "w", encoding="utf-8") as fh:
            for r, j in zip(rows, picks[w]):
                c = r["cands"][j]
                fh.write(json.dumps({"id": r["id"], "vi": r["vi"], "ref": r["ref"], "core": r["core"],
                                     "hyp": c["hyp"], "rank": j, "src": f"combo{w}"},
                                    ensure_ascii=False) + "\n")
        print(f"wrote sel_{name}.jsonl")


if __name__ == "__main__":
    main()
