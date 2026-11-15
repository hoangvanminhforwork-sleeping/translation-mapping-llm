"""Score every generation channel: BLEU, chrF, COMET, paired bootstrap, and structural diagnostics.

With 5,000 sentences a BLEU difference under ~0.6 is noise, so each channel gets a bootstrap CI and
the *difference* to `mapped` is bootstrapped on the same resample indices (paired). Diagnostics keep
a low BLEU from being read as "bad translation" when it is a token loop or an echo.

Usage
    python -m src.score --direction vi_en [--comet] [--boot 1000]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata

import numpy as np
from sacrebleu.metrics import BLEU, CHRF

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")
BLEU_M, CHRF_M = BLEU(tokenize="13a"), CHRF()
VI_MARKS = re.compile(r"[\u00e0-\u01b0\u1ea0-\u1ef9]")
ORDER = ("mapped", "lgbm", "native", "identity", "copy")


def diagnose(hyp: list[str]) -> dict:
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
    return {"empty_rate": empty / len(hyp), "degenerate_rate": deg / len(hyp),
            "vi_word_rate": vi_tok / max(words, 1), "mean_hyp_words": words / max(len(hyp) - empty, 1)}


def load(direction: str, cond: str):
    p = C.RESULTS / f"gen_{direction}_{cond}.jsonl"
    rows = [json.loads(line) for line in p.open(encoding="utf-8")]
    return ([r["src"] for r in rows], [r["ref"] for r in rows], [r["hyp"] for r in rows],
            np.array([r.get("nll", np.nan) for r in rows], dtype=float))


def main(direction: str, use_comet: bool, n_boot: int):
    src_side, tgt_side = direction.split("_")
    conds = [c for c in ORDER if (C.RESULTS / f"gen_{direction}_{c}.jsonl").exists()]
    data, out = {}, {}
    for cond in conds:
        src, ref, hyp, nll = load(direction, cond)
        data[cond] = (src, ref, hyp, nll)
        b, c = BLEU_M.corpus_score(hyp, [ref]), CHRF_M.corpus_score(hyp, [ref])
        out[cond] = {"bleu": b.score, "chrf": c.score, "n": len(hyp),
                     "bleu_signature": str(BLEU_M.get_signature()),
                     "ref_vi_word_rate": sum(1 for r in ref for w in r.split()
                                             if VI_MARKS.search(unicodedata.normalize("NFC", w))) /
                                        max(sum(len(r.split()) for r in ref), 1),
                     "mean_nll": float(np.nanmean(nll)) if np.isfinite(nll).any() else None,
                     **diagnose(hyp)}
        print(f"{cond:9s} BLEU {b.score:6.2f}  chrF {c.score:6.2f}  NLL {out[cond]['mean_nll']}  "
              f"empty {out[cond]['empty_rate']:.3f}  degen {out[cond]['degenerate_rate']:.3f}", flush=True)

    if out:
        n = len(next(iter(data.values()))[1])
        rng = np.random.default_rng(20261115)
        idx_sets = [rng.integers(0, n, n) for _ in range(n_boot)]

        def boot(hyp, ref):
            return np.array([BLEU_M.corpus_score([hyp[i] for i in ix], [[ref[i] for i in ix]]).score
                             for ix in idx_sets])

        boots = {c: boot(v[2], v[1]) for c, v in data.items()}
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
                "p_other_le_mapped": float((d <= 0).mean()), "n_boot": n_boot}
            print(f"  {other} - mapped: {d.mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  "
                  f"P(diff<=0) {(d <= 0).mean():.3f}")

    if use_comet and data:
        from comet import download_model, load_from_checkpoint
        ckpt = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(ckpt)
        for cond, (src, ref, hyp, _nll) in data.items():
            k = min(C.N_COMET, len(hyp))
            res = model.predict([{"src": s, "mt": h, "ref": r} for s, h, r in zip(src[:k], hyp[:k], ref[:k])],
                                batch_size=16, gpus=1)
            out[cond]["comet"] = float(res.system_score)
            out[cond]["comet_n"] = k
            print(f"  {cond:9s} COMET {res.system_score:.4f} (n={k})", flush=True)
        out["comet_checkpoint"] = "Unbabel/wmt22-comet-da"

    p = C.RESULTS / f"metrics_{direction}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["vi_en", "en_vi"], required=True)
    ap.add_argument("--comet", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    main(a.direction, a.comet, a.boot)
