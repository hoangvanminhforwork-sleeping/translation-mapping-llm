"""Turn one candidate pool into one selection per arm, then report what each arm bought.

Every arm reads the SAME pool (results/pool_*.jsonl) and the SAME frozen features
(results/feat_*.parquet). Only the utility changes, so differences are attributable.

Arms (reference-free unless marked)
    A0_greedy   the historical baseline (separate pool, beam=1)
    A1_beam     pool rank 0 (beam top-1)
    mbr_chrf    mean pairwise chrF against the other candidates   (MBR, no reference)
    lp          length-normalised model log-prob
    lenratio    |len_ratio - 1| penalised... i.e. closest to the source length
    rt_str      string round-trip: chrF(back-translation, source)
    rt_lat      latent round-trip: token-level nearest-neighbour agreement in A's encoder space
    rt_lat_cos  latent round-trip, pooled-cosine variant
    learned     tiny MLP trained on train-split candidates with COMET as target
    oracle      best candidate by true COMET (needs refs -> headroom only)
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


def load_pool(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_sel(rows: list[dict], picks: list[int], name: str) -> None:
    path = C.RESULTS / f"sel_{name}.jsonl"
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r, j in zip(rows, picks):
            c = r["cands"][j]
            f.write(json.dumps({"id": r["id"], "vi": r["vi"], "ref": r["ref"], "core": r["core"],
                                "hyp": c["hyp"], "rank": j, "src": c["src"]}, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"  wrote sel_{name}.jsonl")


def mbr_pick(cands: list[dict]) -> int:
    hyps = [c["hyp"] for c in cands]
    n = len(hyps)
    if n == 1:
        return 0
    util = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                util[i] += CHRF_M.sentence_score(hyps[i], [hyps[j]]).score
    return int(util.argmax())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--greedy", default="pool_test_greedy.jsonl")
    ap.add_argument("--arms", default="all")
    a = ap.parse_args(argv)

    rows = load_pool(C.RESULTS / a.pool)
    feat = pq.read_table(C.RESULTS / a.feat).to_pylist()
    by_id: dict[int, list[dict]] = {}
    for f in feat:
        by_id.setdefault(f["id"], []).append(f)
    for v in by_id.values():
        v.sort(key=lambda x: x["rank"])

    arms = a.arms.split(",") if a.arms != "all" else \
        ["beam", "lenratio", "rt_str", "rt_lat", "rt_lat_c", "mbr_chrf"]

    print(f"{len(rows)} rows, arms={arms}")
    write_sel(rows, [0] * len(rows), "A1_beam")

    if "A0_greedy" in arms or a.greedy:
        g = load_pool(C.RESULTS / a.greedy) if (C.RESULTS / a.greedy).exists() else None
        if g:
            path = C.RESULTS / "sel_A0_greedy.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for r in g:
                    f.write(json.dumps({"id": r["id"], "vi": r["vi"], "ref": r["ref"],
                                        "core": r["core"], "hyp": r["cands"][0]["hyp"],
                                        "rank": 0, "src": "greedy"}, ensure_ascii=False) + "\n")
            print("  wrote sel_A0_greedy.jsonl")

    def pick_by(fn, name):
        picks = [fn(by_id[r["id"]]) for r in rows]
        write_sel(rows, picks, name)

    if "mbr_chrf" in arms:
        print("mbr_chrf ...", flush=True)
        picks = []
        for k, r in enumerate(rows):
            picks.append(mbr_pick(r["cands"]))
            if k % 2000 == 0:
                print(f"  {k}/{len(rows)}", flush=True)
        write_sel(rows, picks, "A5_mbr_chrf")

    if "lenratio" in arms:
        pick_by(lambda v: int(np.argmin([abs(x["len_ratio"] - 1.0) for x in v])), "A3_lenratio")
    if "rt_str" in arms:
        pick_by(lambda v: int(np.argmax([x["chrF_bt"] for x in v])), "A6_rt_str")
    if "rt_lat" in arms:
        pick_by(lambda v: int(np.argmax([x["lat_match_f1"] for x in v])), "A7_rt_lat")
    if "rt_lat_c" in arms:
        pick_by(lambda v: int(np.argmax([x["lat_match_f1_c"] for x in v])), "A7c_rt_lat_centred")
    if "rt_lat_cos" in arms:
        pick_by(lambda v: int(np.argmax([x["lat_cos_mean"] for x in v])), "A7b_rt_lat_cos")


if __name__ == "__main__":
    main()
