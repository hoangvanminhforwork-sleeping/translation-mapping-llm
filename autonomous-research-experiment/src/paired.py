"""Paired contrast between two selection files (the test the pre-registration promises).

`score.py` reports each arm against A1_beam. Some hypotheses compare two *other* arms (latent vs
string, MBR vs latent), and that difference must be bootstrapped on the same sentence indices, not
computed as a difference of two independent means.

Usage
    python -m src.paired --a sel_A7_rt_lat.jsonl --b sel_A6_rt_str.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from sacrebleu.metrics import BLEU

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")
BLEU_M = BLEU(tokenize="13a")


def load(name: str) -> dict[int, dict]:
    path = C.RESULTS / (name if name.endswith(".jsonl") else f"sel_{name}.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return {r["id"]: r for r in rows}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(argv)

    A, B = load(a.a), load(a.b)
    ids = sorted(set(A) & set(B))
    hyp_a = [A[i]["hyp"] for i in ids]
    hyp_b = [B[i]["hyp"] for i in ids]
    ref = [A[i]["ref"] for i in ids]

    sa = np.asarray(BLEU_M._extract_corpus_statistics(hyp_a, [ref]), dtype=np.float64)
    sb = np.asarray(BLEU_M._extract_corpus_statistics(hyp_b, [ref]), dtype=np.float64)
    point = (BLEU_M._compute_score_from_stats(sa.sum(0)).score
             - BLEU_M._compute_score_from_stats(sb.sum(0)).score)

    rng = np.random.default_rng(C.SEED)
    n = len(ids)
    diffs = np.empty(a.boot)
    for k in range(a.boot):
        ix = rng.integers(0, n, n)
        diffs[k] = (BLEU_M._compute_score_from_stats(sa[ix].sum(0)).score
                    - BLEU_M._compute_score_from_stats(sb[ix].sum(0)).score)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(json.dumps({"a": a.a, "b": a.b, "n": n, "delta_bleu": round(point, 3),
                      "ci95": [round(float(lo), 3), round(float(hi), 3)],
                      "p_delta_le_0": round(float((diffs <= 0).mean()), 3),
                      "boot": a.boot}, indent=2))


if __name__ == "__main__":
    main()
