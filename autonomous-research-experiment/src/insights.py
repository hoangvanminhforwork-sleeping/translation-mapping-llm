"""Descriptive facts a reader needs before believing any arm comparison.

1. Pool redundancy: how many *distinct* hypotheses a row actually offers, and how similar they are.
2. Misalignment proxy: MTet's known defect. For each test row, take the best COMET over the pool.
   Rows whose *best possible* candidate is still poor are either genuinely hard or misaligned; the
   share is reported (never called "misaligned" outright - the proxy cannot tell the two apart).
3. Headroom: rank-0 vs oracle, in COMET and in BLEU.
Outputs results/insights.json; the report quotes it.
"""
from __future__ import annotations

import itertools
import json
import statistics
import sys

import numpy as np
import pyarrow.parquet as pq
from sacrebleu.metrics import BLEU, CHRF

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")
BLEU_M = BLEU(tokenize="13a")
CHRF_M = CHRF()


def beam_effect() -> dict:
    """Greedy vs beam-8 top-1: how often does beam change the output, and is the change good?"""
    g = {json.loads(l)["id"]: json.loads(l) for l in
         open(C.RESULTS / "sel_A0_greedy.jsonl", encoding="utf-8")}
    b = {json.loads(l)["id"]: json.loads(l) for l in
         open(C.RESULTS / "sel_A1_beam.jsonl", encoding="utf-8")}
    ids = sorted(set(g) & set(b))
    d = np.array([CHRF_M.sentence_score(b[i]["hyp"], [b[i]["ref"]]).score
                  - CHRF_M.sentence_score(g[i]["hyp"], [g[i]["ref"]]).score for i in ids])
    lens = np.array([len(g[i]["vi"].split()) for i in ids])
    out = {"n": len(ids),
           "identical_rate": float(np.mean([b[i]["hyp"] == g[i]["hyp"] for i in ids])),
           "mean_sentence_chrf_delta": float(d.mean()),
           "share_improved": float((d > 0).mean()),
           "share_worsened": float((d < 0).mean()),
           "by_len": {}}
    for lo, hi in ((1, 10), (11, 20), (21, 30), (31, 100)):
        m = (lens >= lo) & (lens <= hi)
        if m.sum() > 50:
            out["by_len"][f"{lo}-{hi}"] = {"n": int(m.sum()), "delta": float(d[m].mean())}
    return out


def main() -> None:
    rows = [json.loads(line) for line in open(C.RESULTS / "pool_test_beam8.jsonl", encoding="utf-8")]
    com = pq.read_table(C.RESULTS / "comet_test_beam8.parquet").to_pylist()
    best: dict[int, float] = {}
    first: dict[int, float] = {}
    for r in com:
        best[r["id"]] = max(best.get(r["id"], -9), r["comet"])
        if r["rank"] == 0:
            first[r["id"]] = r["comet"]

    # redundancy
    uniq = [len({c["hyp"] for c in r["cands"]}) for r in rows]
    pair = [CHRF_M.sentence_score(c["hyp"], [d["hyp"]]).score
            for r in rows[:300] for c, d in itertools.combinations(r["cands"], 2)]

    maxc = np.array([best[r["id"]] for r in rows])
    topc = np.array([first[r["id"]] for r in rows])
    ref = [r["ref"] for r in rows]
    top1 = [r["cands"][0]["hyp"] for r in rows]
    orac = [r["cands"][int(np.argmax([best[r["id"]]]))]["hyp"] for r in rows]  # placeholder, replaced below

    # real oracle hypotheses
    by_id: dict[int, list[float]] = {}
    for r in com:
        by_id.setdefault(r["id"], []).append(r["comet"])
    orac = []
    for r in rows:
        j = int(np.argmax(by_id[r["id"]]))
        orac.append(r["cands"][j]["hyp"])

    out = {
        "n_rows": len(rows),
        "pool_unique_mean": statistics.mean(uniq),
        "pool_unique_min": min(uniq),
        "pool_pairwise_chrf_mean": statistics.mean(pair),
        "comet_best_mean": float(maxc.mean()),
        "comet_top1_mean": float(topc.mean()),
        "comet_gap_top1_to_best": float((maxc - topc).mean()),
        "share_best_comet_lt": {t: float((maxc < t).mean()) for t in (0.3, 0.4, 0.5, 0.6)},
        "bleu_top1": BLEU_M.corpus_score(top1, [ref]).score,
        "bleu_oracle": BLEU_M.corpus_score(orac, [ref]).score,
        "chrf_top1": CHRF_M.corpus_score(top1, [ref]).score,
        "chrf_oracle": CHRF_M.corpus_score(orac, [ref]).score,
    }
    out["headroom_bleu"] = out["bleu_oracle"] - out["bleu_top1"]
    out["headroom_chrf"] = out["chrf_oracle"] - out["chrf_top1"]
    out["beam_effect"] = beam_effect()
    (C.RESULTS / "insights.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
