"""Geometry of the fitted maps: token/sentence cosine, retrieval@1, and the controls.

The controls are the point. Predicting the dataset mean already scores cosine ~0.32 per token on
these spaces (phase 01 measured 0.32 / 0.70 with 768-d models), so cosine cannot separate a real map
from a constant. retrieval@1 can: query = mapped source states of sentence i, gallery = target
states of the *same* n test pairs, chance = 1/n.

Outputs: results/geometry_{direction}.{csv,json}, plus per-sentence error for the neural-mining join.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import config as C
from .embed import Encoders, load_chosen
from .tokenbatch import AlignedBatcher
from .train_decoder import direction_sides

sys.stdout.reconfigure(encoding="utf-8")


def unit(a):
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9, None)


def collect(enc, direction: str, n: int, batch: int):
    src, tgt = direction_sides(direction)
    d = load_chosen(("vi", "en", "split"))
    te = [i for i, s in enumerate(d["split"]) if s == "test"][:n]
    src_texts = [d[src][i] for i in te]
    tgt_texts = [d[tgt][i] for i in te]
    Xs, Ys, sx, sy = [], [], [], []
    for _s, nb, X, Y, lens in AlignedBatcher(enc, batch, src=src, tgt=tgt).batches(src_texts, tgt_texts):
        Xn, Yn = X.cpu().numpy(), Y.cpu().numpy()
        ln = lens.cpu().numpy()
        seg = np.repeat(np.arange(len(ln)), ln)
        B = len(ln)
        L = ln.clip(min=1).astype(np.float32)[:, None]
        bx = np.zeros((B, X.shape[1]), np.float32)
        by = np.zeros((B, Y.shape[1]), np.float32)
        np.add.at(bx, seg, Xn)
        np.add.at(by, seg, Yn)
        Xs.append(Xn); Ys.append(Yn); sx.append(bx / L); sy.append(by / L)
    return (np.concatenate(Xs), np.concatenate(Ys), np.concatenate(sx), np.concatenate(sy),
            np.array(te), src_texts, tgt_texts)


def variant_rows(name, P_tok, P_sent, tgt_tok, tgt_sent):
    tok_cos = (unit(P_tok) * unit(tgt_tok)).sum(-1)
    sent_cos = (unit(P_sent) * unit(tgt_sent)).sum(-1)
    sims = unit(P_sent) @ unit(tgt_sent).T
    idx = np.arange(len(tgt_sent))
    r1 = float((sims.argmax(1) == idx).mean())
    correct = sims[idx, idx]
    wrong = sims.copy(); wrong[idx, idx] = -2.0
    return {"variant": name,
            "token_cosine": float(tok_cos.mean()), "token_mse": float(((P_tok - tgt_tok) ** 2).sum(-1).mean()),
            "sentence_cosine": float(sent_cos.mean()), "sentence_mse": float(((P_sent - tgt_sent) ** 2).sum(-1).mean()),
            "retrieval@1": r1, "margin_mean": float((correct - wrong.max(1)).mean()),
            "margin_frac_positive": float((correct > wrong.max(1)).mean())}, tok_cos, sent_cos


def main(direction: str, n: int, batch: int):
    enc = Encoders()
    src, tgt = direction_sides(direction)
    Xte, Yte, Sx, Sy, te_ids, vi, en = collect(enc, direction, n, batch)
    print(f"{direction}: {len(Sx):,} test pairs, {len(Xte):,} token pairs", flush=True)

    z = np.load(C.MAP[direction])
    W, b = z["W"].astype(np.float32), z["b"].astype(np.float32)
    variants = {"map": (Xte @ W + b, Sx @ W + b),
                "identity": (Xte, Sx),
                "mean": (np.broadcast_to(Yte.mean(0), Xte.shape), np.broadcast_to(Sy.mean(0), Sx.shape))}
    if (C.DATA / "lgbm" / f"{direction}.pkl.gz").exists():
        from .fit_lgbm import LgbmMap
        lm = LgbmMap(direction)
        variants["lgbm"] = (lm(Xte), lm(Sx))

    rows, per_sent = [], {}
    for name, (Pt, Ps) in variants.items():
        r, tok_cos, sent_cos = variant_rows(name, Pt, Ps, Yte, Sy)
        rows.append(r)
        print(f"  {name:9s} tok_cos {r['token_cosine']:.4f}  sent_cos {r['sentence_cosine']:.4f}  "
              f"R@1 {r['retrieval@1']:.3f}  tok_mse {r['token_mse']:.4f}", flush=True)
        if name in ("map", "lgbm"):
            # per-sentence sentence cosine, aligned to `te_ids` (used by mine.py)
            per_sent[name] = {"sent_cos": sent_cos.tolist()}

    out = {"direction": direction, "n_pairs": int(len(Sx)), "n_tokens": int(len(Xte)),
           "chance_r1": 1.0 / len(Sx), "lambda": float(z["lam"]), "rows": rows,
           "test_ids": te_ids.tolist()}
    (C.RESULTS / f"geometry_{direction}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with (C.RESULTS / f"geometry_{direction}.csv").open("w", encoding="utf-8") as f:
        f.write("direction,variant,token_cosine,token_mse,sentence_cosine,sentence_mse,retrieval_at_1,"
                "margin_mean,margin_frac_positive\n")
        for r in rows:
            f.write(f"{direction},{r['variant']},{r['token_cosine']:.4f},{r['token_mse']:.6f},"
                    f"{r['sentence_cosine']:.4f},{r['sentence_mse']:.6f},{r['retrieval@1']:.4f},"
                    f"{r['margin_mean']:.4f},{r['margin_frac_positive']:.4f}\n")
    # per-sentence sentence-cosine for mining (join key = id in gen_*.jsonl)
    np.savez_compressed(C.DATA / f"per_sent_{direction}.npz", ids=te_ids,
                        **{f"{k}_sent_cos": np.asarray(v["sent_cos"]) for k, v in per_sent.items()})
    print(f"wrote geometry_{direction}.{{csv,json}}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["vi_en", "en_vi"], required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    main(a.direction, a.n, a.batch)
