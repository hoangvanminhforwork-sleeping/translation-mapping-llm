"""Score hypotheses: BLEU / chrF / COMET + paired bootstrap over sentences.

Every arm is a *selection* from the same candidate pool, so all comparisons are paired by
construction (same sentences, same pool). COMET is opt-in because it needs the GPU.

Usage
    python -m src.score --hyp results/sel_A1_beam.jsonl ...
    python -m src.score --comet --subset core
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

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")
BLEU_M = BLEU(tokenize="13a")
CHRF_M = CHRF()
VI = re.compile(r"[\u00e0-\u01b0\u1ea0-\u1ef9]")


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def diagnose(ref: list[str], hyp: list[str]) -> dict:
    empty = deg = words = vi_tok = 0
    for h in hyp:
        toks = h.split()
        if not toks:
            empty += 1
            continue
        if len(set(toks)) / len(toks) <= 0.5:
            deg += 1
        words += len(toks)
        vi_tok += sum(1 for w in toks if VI.search(unicodedata.normalize("NFC", w)))
    return {"n": len(hyp), "empty_rate": empty / len(hyp), "degenerate_rate": deg / len(hyp),
            "vi_word_rate": vi_tok / max(words, 1),
            "mean_hyp_words": words / max(len(hyp) - empty, 1),
            "mean_ref_words": sum(len(r.split()) for r in ref) / len(ref)}


def bleu(hyp, ref) -> float:
    return BLEU_M.corpus_score(hyp, [ref]).score


def bootstrap(hyp, ref, idx_sets):
    """Bootstrap over sentences using sacrebleu's per-sentence sufficient statistics.

    Re-tokenising 10 k sentences for every resample costs ~30 min per arm; summing pre-computed
    sentence statistics costs ~1 s per arm and is numerically identical.
    """
    stats = np.asarray(BLEU_M._extract_corpus_statistics(hyp, [ref]), dtype=np.float64)
    return np.array([BLEU_M._compute_score_from_stats(stats[ix].sum(0)).score for ix in idx_sets])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", nargs="+", required=True, help="jsonl files with id/vi/ref/hyp")
    ap.add_argument("--comet", action="store_true")
    ap.add_argument("--subset", default="all", choices=["all", "core", "notcore"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--comet-batch", type=int, default=32)
    ap.add_argument("--out", default="metrics.json")
    a = ap.parse_args(argv)

    data = {}
    for path in a.hyp:
        name = path.split("/")[-1].split("\\")[-1].replace("sel_", "").replace(".jsonl", "")
        rows = load_jsonl(path)
        if a.subset == "core":
            rows = [r for r in rows if r.get("core")]
        elif a.subset == "notcore":
            rows = [r for r in rows if not r.get("core")]
        data[name] = rows

    out = {}
    for name, rows in data.items():
        src = [r["vi"] for r in rows]
        ref = [r["ref"] for r in rows]
        hyp = [r["hyp"] for r in rows]
        b = BLEU_M.corpus_score(hyp, [ref])
        c = CHRF_M.corpus_score(hyp, [ref])
        out[name] = {"bleu": b.score, "bleu_signature": str(BLEU_M.get_signature()), "chrf": c.score,
                     **diagnose(ref, hyp)}
        print(f"{name:28s} BLEU {b.score:6.2f}  chrF {c.score:6.2f}  "
              f"degen {out[name]['degenerate_rate']:.3f}  vi {out[name]['vi_word_rate']:.4f}", flush=True)

    # paired bootstrap: identical resample indices across arms
    first = next(iter(data.values()))
    n = len(first)
    rng = np.random.default_rng(C.SEED)
    idx = [rng.integers(0, n, n) for _ in range(a.boot)]
    boots = {}
    for name, rows in data.items():
        hyp = [r["hyp"] for r in rows]
        ref = [r["ref"] for r in rows]
        arr = bootstrap(hyp, ref, idx)
        boots[name] = arr
        lo, hi = np.percentile(arr, [2.5, 97.5])
        out[name]["bleu_ci95"] = [float(lo), float(hi)]
        print(f"  {name:26s} 95% CI [{lo:.2f}, {hi:.2f}]")

    base = "A1_beam"
    if base in boots:
        for name, arr in boots.items():
            if name == base:
                continue
            d = arr - boots[base]
            lo, hi = np.percentile(d, [2.5, 97.5])
            out[name]["paired_vs_" + base] = {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
                                              "p_le_0": float((d <= 0).mean())}
            print(f"  {name:26s} vs {base}: {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] "
                  f"P(<=0) {(d <= 0).mean():.3f}")

    if a.comet:
        from comet import load_from_checkpoint, download_model

        ckpt = download_model(C.COMET_ID)
        cmodel = load_from_checkpoint(ckpt)
        for name, rows in data.items():
            payload = [{"src": r["vi"], "mt": r["hyp"], "ref": r["ref"]} for r in rows]
            res = cmodel.predict(payload, batch_size=a.comet_batch, gpus=1 if __import__("torch").cuda.is_available() else 0)
            out[name]["comet"] = float(res.system_score)
            out[name]["comet_sent"] = [float(x) for x in res.scores]
            print(f"  {name:26s} COMET {res.system_score:.4f}", flush=True)
        out["comet_checkpoint"] = C.COMET_ID

    (C.RESULTS / a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {C.RESULTS / a.out}")


if __name__ == "__main__":
    main()
