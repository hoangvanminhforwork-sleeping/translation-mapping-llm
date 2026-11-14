"""Geometry of the fitted map: per-token error, sentence-level cosine, retrieval@1, and controls.

The controls are the point. Predicting the dataset mean scores cosine ~0.70 for free on these
spaces (measured at 100k in research-scale-100k), so cosine cannot separate a real map from a
constant; retrieval@1 and the identity control can.

retrieval@1 is the standard paired-retrieval setup: query = mapped vit5 states of sentence i,
gallery = flan states of the English side of the *same* 2,000 test pairs; a correct map returns i.
Chance is 1/2000 = 0.05 %. This is the same protocol as the 100k run, so the numbers are comparable.

Outputs: results/mapping_metrics.csv, results/mapping_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import config as C
from .fit_tok import Encoders, iter_slices
from .tokenbatch import AlignedBatcher

sys.stdout.reconfigure(encoding="utf-8")


def collect(enc: Encoders, split: str, n: int, batch: int):
    """Return token-level (X, Y) and sentence-pooled (Sx, Sy) arrays in pair order."""
    for _idx, vi, en in iter_slices(split, n):
        xs, ys, sx, sy = [], [], [], []
        got = 0
        for _s, nb, X, Y, lens in AlignedBatcher(enc, batch).batches(vi, en):
            Xn, Yn = X.cpu().numpy(), Y.cpu().numpy()
            ln = lens.cpu().numpy()
            seg = np.repeat(np.arange(len(ln)), ln)
            B = len(ln)
            L = ln.clip(min=1).astype(np.float32)[:, None]
            sums_x = np.zeros((B, X.shape[1]), np.float32)
            sums_y = np.zeros((B, Y.shape[1]), np.float32)
            np.add.at(sums_x, seg, Xn)
            np.add.at(sums_y, seg, Yn)
            xs.append(Xn)
            ys.append(Yn)
            sx.append(sums_x / L)
            sy.append(sums_y / L)
            got += nb
        return (np.concatenate(xs), np.concatenate(ys), np.concatenate(sx), np.concatenate(sy), got)
    raise SystemExit(f"no rows for split {split}")


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True).clip(min=1e-9)


def main(batch: int = 256):
    enc = Encoders()
    print("encoding test split …", flush=True)
    Xte, Yte, Sx_te, Sy_te, n_te = collect(enc, "test", C.N_TEST, batch)
    print(f"  {n_te} pairs, {len(Xte):,} token pairs", flush=True)

    out = {"n_test_pairs": n_te, "n_test_tokens": int(len(Xte)), "chance_r1": 1.0 / n_te,
           "protocol": "paired retrieval: query = mapped source tokens; gallery = target tokens of the "
                       "same 2,000 pairs", "directions": {}}
    for direction, map_path, src_tok, src_sent, tgt_tok, tgt_sent in (
            ("vi_en", C.MAP_VI_EN, Xte, Sx_te, Yte, Sy_te),
            ("en_vi", C.DATA / "map_en_vi.npz", Yte, Sy_te, Xte, Sx_te)):
        if not map_path.exists():
            print(f"{direction}: {map_path.name} missing, skipped")
            continue
        z = np.load(map_path)
        W, b = z["W"].astype(np.float32), z["b"].astype(np.float32)
        variants = {
            "map": (src_tok @ W + b, src_sent @ W + b),
            "identity": (src_tok, src_sent),
            "mean": (np.broadcast_to(tgt_tok.mean(0), src_tok.shape),
                     np.broadcast_to(tgt_sent.mean(0), src_sent.shape)),
        }
        g_unit = unit(tgt_sent)
        idx = np.arange(len(tgt_sent))
        rows = []
        for name, (P_tok, P_sent) in variants.items():
            tok_cos = float((unit(P_tok) * unit(tgt_tok)).sum(-1).mean())
            tok_mse = float(((P_tok - tgt_tok) ** 2).sum(-1).mean())
            sent_cos = float((unit(P_sent) * unit(tgt_sent)).sum(-1).mean())
            sent_mse = float(((P_sent - tgt_sent) ** 2).sum(-1).mean())
            sims = unit(P_sent) @ g_unit.T
            r1 = float((sims.argmax(1) == idx).mean())
            correct = sims[idx, idx]
            wrong = sims.copy()
            wrong[idx, idx] = -2.0
            rows.append({"variant": name, "token_cosine": tok_cos, "token_mse": tok_mse,
                         "sentence_cosine": sent_cos, "sentence_mse": sent_mse,
                         "retrieval@1": r1, "margin_mean": float((correct - wrong.max(1)).mean()),
                         "margin_frac_positive": float((correct > wrong.max(1)).mean())})
            print(f"  {direction} {name:9s} tok_cos {tok_cos:.4f}  sent_cos {sent_cos:.4f}  R@1 {r1:.3f}  "
                  f"margin+ {rows[-1]['margin_frac_positive']:.3f}", flush=True)
        out["directions"][direction] = {
            "lambda": float(z["lam"]), "map_shards": int(z["shards"]),
            "map_pairs_seen": int(z["n_pairs"]), "rows": rows}

    (C.RESULTS / "mapping_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with (C.RESULTS / "mapping_metrics.csv").open("w", encoding="utf-8") as f:
        f.write("direction,variant,token_cosine,token_mse,sentence_cosine,sentence_mse,"
                "retrieval_at_1,margin_mean,margin_frac_positive\n")
        for d, dd in out["directions"].items():
            for r in dd["rows"]:
                f.write(f"{d},{r['variant']},{r['token_cosine']:.4f},{r['token_mse']:.6f},"
                        f"{r['sentence_cosine']:.4f},{r['sentence_mse']:.6f},{r['retrieval@1']:.4f},"
                        f"{r['margin_mean']:.4f},{r['margin_frac_positive']:.4f}\n")
    print(json.dumps({d: {k: v for k, v in dd.items() if k != "rows"} for d, dd in out["directions"].items()},
                     indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    main(a.batch)
