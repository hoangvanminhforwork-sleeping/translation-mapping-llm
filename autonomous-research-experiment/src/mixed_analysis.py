"""Does a more diverse pool raise the ceiling, and does the selector rise with it?

Compares the 2 k core under two pools:
    beam-8 only          results/pool_test_beam8.jsonl  (subset core)
    beam-8 + sample-8    results/pool_core_mixed.jsonl
using the same teacher (COMET) for the oracle and the same rules for selection.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pyarrow.parquet as pq
from sacrebleu.metrics import BLEU, CHRF

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")
BLEU_M = BLEU(tokenize="13a")
CHRF_M = CHRF()


def stats(pool_path: str, comet_path: str) -> dict:
    pool = [json.loads(line) for line in open(C.RESULTS / pool_path, encoding="utf-8")]
    if pool_path.startswith("pool_test"):
        pool = [r for r in pool if r.get("core")]
    com = pq.read_table(C.RESULTS / comet_path).to_pylist()
    by: dict[int, list[tuple[float, int]]] = {}
    for r in com:
        by.setdefault(r["id"], []).append((r["comet"], r["rank"]))
    top = [r["cands"][0]["hyp"] for r in pool]
    orac = [r["cands"][max(by[r["id"]])[1]]["hyp"] for r in pool]
    ref = [r["ref"] for r in pool]
    uniq = float(np.mean([len({c["hyp"] for c in r["cands"]}) for r in pool]))
    return {
        "n": len(pool),
        "cands_per_row": len(pool[0]["cands"]),
        "unique_per_row": round(uniq, 2),
        "bleu_top1": round(BLEU_M.corpus_score(top, [ref]).score, 2),
        "bleu_oracle": round(BLEU_M.corpus_score(orac, [ref]).score, 2),
        "chrf_top1": round(CHRF_M.corpus_score(top, [ref]).score, 2),
        "chrf_oracle": round(CHRF_M.corpus_score(orac, [ref]).score, 2),
        "comet_top1_mean": round(float(np.mean([max([c for c, _ in [v[0]]]) for v in
                                                 [[by[r["id"]][0]] for r in pool]])), 4),
    }


def main() -> None:
    a = stats("pool_test_beam8.jsonl", "comet_test_beam8.parquet")
    a["pool"] = "beam-8"
    b = stats("pool_core_mixed.jsonl", "comet_core_mixed.parquet")
    b["pool"] = "beam-8 + sample-8"
    out = {"beam8_core": a, "mixed_core": b}
    (C.RESULTS / "mixed_pool.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
