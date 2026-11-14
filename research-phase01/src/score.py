"""Score the generation runs: BLEU, chrF, COMET, bootstrap intervals, paired channel comparison, and
the structural diagnostics that stop a number from being read as translation quality when it is not.

Why the CIs: with 2,000 sentences, BLEU differences under ~1 point are noise. The previous phase
reported 0.141 vs 0.240 with no interval, which cannot support an ordering; here every channel gets a
paired bootstrap over sentences, and the *difference* between channels is bootstrapped with the same
resample indices (paired), so the interval answers "is mapped worse than native" directly.

Diagnostics per channel (a BLEU cannot separate "bad translation" from "token loop" from "echo"):
  degenerate    unique tokens / tokens <= 0.5
  vi_word_rate  share of output words carrying Vietnamese diacritics (references ~0.1 %)
  empty         share of empty hypotheses

Usage
    python -m src.score [--comet] [--boot 1000]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata

import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

BLEU_M = BLEU(tokenize="13a")
CHRF_M = CHRF()

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")
CONDS = ("mapped", "native", "identity")
VI_MARKS = re.compile(r"[\u00e0-\u01b0\u1ea0-\u1ef9]")


def diagnose(ref: list[str], hyp: list[str]) -> dict:
    deg = empty = words = vi_tok = 0
    for h in hyp:
        toks = h.split()
        if not toks:
            empty += 1
            continue
        if len(set(toks)) / len(toks) <= 0.5:
            deg += 1
        words += len(toks)
        vi_tok += sum(1 for w in toks if VI_MARKS.search(unicodedata.normalize("NFC", w)))
    ref_vi = sum(1 for r in ref for w in r.split() if VI_MARKS.search(unicodedata.normalize("NFC", w)))
    ref_words = sum(len(r.split()) for r in ref)
    return {"n": len(hyp), "empty_rate": empty / len(hyp), "degenerate_rate": deg / len(hyp),
            "vi_word_rate": vi_tok / max(words, 1), "ref_vi_word_rate": ref_vi / max(ref_words, 1),
            "mean_hyp_words": words / max(len(hyp) - empty, 1), "mean_ref_words": ref_words / max(len(ref), 1)}


def bleu(hyp, ref):
    return BLEU_M.corpus_score(hyp, [ref]).score


def bootstrap(hyp, ref, idx_sets):
    return np.array([bleu([hyp[i] for i in ix], [ref[i] for i in ix]) for ix in idx_sets])


def load(cond):
    rows = [json.loads(line) for line in (C.RESULTS / f"gen_{cond}.jsonl").open(encoding="utf-8")]
    return [r["vi"] for r in rows], [r["ref"] for r in rows], [r["hyp"] for r in rows]


def main(use_comet: bool, n_boot: int):
    out = {}
    data = {}
    conds = [c for c in ("mapped", "native", "identity", "baseline")
             if (C.RESULTS / f"gen_{c}.jsonl").exists()]
    for cond in conds:
        src, ref, hyp = load(cond)
        data[cond] = (src, ref, hyp)
        b = BLEU_M.corpus_score(hyp, [ref])
        c = CHRF_M.corpus_score(hyp, [ref])
        out[cond] = {"bleu": b.score, "bleu_signature": str(BLEU_M.get_signature()), "chrf": c.score,
                     **diagnose(ref, hyp)}
        print(f"{cond:9s} BLEU {b.score:6.2f}  chrF {c.score:6.2f}  empty {out[cond]['empty_rate']:.3f}  "
              f"degen {out[cond]['degenerate_rate']:.3f}  vi-words {out[cond]['vi_word_rate']:.3f}", flush=True)

    if out:
        n = len(next(iter(data.values()))[1])
        rng = np.random.default_rng(20261114)
        idx_sets = [rng.integers(0, n, n) for _ in range(n_boot)]
        boots = {c: bootstrap(v[2], v[1], idx_sets) for c, v in data.items()}
        for c, arr in boots.items():
            lo, hi = np.percentile(arr, [2.5, 97.5])
            out[c]["bleu_ci95"] = [float(lo), float(hi)]
            out[c]["bleu_boot_std"] = float(arr.std())
            print(f"  {c:9s} BLEU 95% CI [{lo:.2f}, {hi:.2f}]")
        for other in (c for c in boots if c != "mapped"):
            d = boots[other] - boots["mapped"]
            lo, hi = np.percentile(d, [2.5, 97.5])
            out[f"paired_{other}_minus_mapped"] = {
                "mean": float(d.mean()), "ci95": [float(lo), float(hi)],
                "p_other_le_mapped": float((d <= 0).mean()), "n_boot": n_boot,
                "reading": "positive mean = this channel beats mapped; P(other<=mapped) near 1 means "
                           "mapped is significantly better"}
            print(f"  {other} - mapped BLEU: {d.mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  "
                  f"P(diff<=0) {(d <= 0).mean():.3f}")

    if use_comet:
        from comet import download_model, load_from_checkpoint

        ckpt = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(ckpt)
        for cond, (src, ref, hyp) in data.items():
            res = model.predict([{"src": s, "mt": h, "ref": r} for s, h, r in zip(src, hyp, ref)],
                                batch_size=16, gpus=1)
            out[cond]["comet"] = float(res.system_score)
            if hasattr(res, "scores"):
                out[cond]["comet_mean_sentence"] = float(np.mean(res.scores))
            print(f"  {cond:9s} COMET {res.system_score:.4f}", flush=True)
        out["comet_checkpoint"] = "Unbabel/wmt22-comet-da"

    (C.RESULTS / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {C.RESULTS / 'metrics.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--comet", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    main(a.comet, a.boot)
