"""Diagnostic — does the folded recolor equal the literal E9 transform, and how big is it?

Two questions, both cheap:
  1. `fold_recolor` must be *identical* to `m -> (m - muM) @ A + muY` blended with t. If not, the
     +0.03 BLEU measured for `ridge_recolor01` is a coding artefact, not a failed replication.
  2. E9 reports z-distances (mapped 6.31, genuine 22.77, full recolor 61.59). Reproduce those
     units so the arms table is comparable with `ANOMALY.md`.

Usage
    python -m src.diag_recolor
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import cc
from .fit_behaviour import fold_recolor

sys.stdout.reconfigure(encoding="utf-8")


def zstats(H, am, mu, sd):
    """Median per-token distance to mu, raw and standardised (E9's 'z-distance')."""
    z = (H - mu) / sd
    d_raw, d_z = [], []
    for i in range(H.shape[0]):
        m = am[i] > 0
        d_raw.append(np.linalg.norm(H[i][m] - mu, axis=1))
        d_z.append(np.linalg.norm(z[i][m], axis=1))
    return {"median_raw": float(np.median(np.concatenate(d_raw))),
            "median_z": float(np.median(np.concatenate(d_z))),
            "mean_raw": float(np.mean(np.concatenate(d_raw)))}


def main():
    d = cc.CACHE / "fit"
    E_ref, m_ref = np.load(d / "E_ref.npy", mmap_mode="r"), np.load(d / "m_ref.npy", mmap_mode="r")
    acc, acc2, cnt = np.zeros(512), np.zeros(512), 0
    for s in range(0, len(m_ref), 512):
        h = np.asarray(E_ref[s:s + 512]).astype(np.float64)
        mk = np.asarray(m_ref[s:s + 512]) > 0
        acc += (h * mk[..., None]).sum((0, 1))
        acc2 += (h * h * mk[..., None]).sum((0, 1))
        cnt += mk.sum()
    mu = acc / cnt
    sd = np.sqrt(np.maximum(acc2 / cnt - mu ** 2, 1e-12))
    print(f"target space: mu norm {np.linalg.norm(mu):.2f}, per-dim sd {sd.mean():.3f} "
          f"-> per-token norm of a genuine state ~ {np.sqrt((sd ** 2).sum()):.2f}")

    z = np.load(cc.WORK / "maps" / "ridge_sub.npz")
    W, b = z["W"], z["b"]
    st = {k: z[k] for k in ("mx", "my", "Cx", "Cy")}

    t = cc.CACHE / "test"
    E = np.asarray(np.load(t / "E_vi.npy", mmap_mode="r")[:200]).astype(np.float64)
    am = np.asarray(np.load(t / "m_vi.npy", mmap_mode="r")[:200])

    def literal(M, t_):
        # E9's fit_recolor, reimplemented inline
        def sqrtm(S):
            w, V = np.linalg.eigh(S)
            return (V * np.sqrt(np.clip(w, 0, None))) @ V.T

        def inv_sqrtm(S):
            w, V = np.linalg.eigh(S)
            return (V / np.sqrt(np.clip(w, 1e-8, None))) @ V.T

        Cm = W.T @ st["Cx"] @ W
        Cm = Cm + np.eye(512) * 1e-3 * np.trace(Cm) / 512
        Cy = st["Cy"] + np.eye(512) * 1e-3 * np.trace(st["Cy"]) / 512
        A = sqrtm(Cy) @ inv_sqrtm(Cm)
        muM = st["mx"] @ W + b
        rec = (M - muM) @ A + st["my"]
        return (1 - t_) * M + t_ * rec

    M = E @ W + b
    rows = {}
    for t_ in (0.0, 0.1, 1.0):
        Wf, bf = fold_recolor(W.astype(np.float32), b.astype(np.float32), st, t_)
        Mf = E @ Wf.astype(np.float64) + bf.astype(np.float64)
        Ml = literal(M, t_)
        rows[f"t={t_}"] = {"max_abs_diff_fold_vs_literal": float(np.abs(Mf - Ml).max()),
                           "rel_diff": float(np.abs(Mf - Ml).max() / max(np.abs(Ml).max(), 1e-9)),
                           **zstats(Mf, am, mu, sd)}
        print(f"t={t_:<4} fold-vs-literal max|d| {rows[f't={t_}']['max_abs_diff_fold_vs_literal']:.3e}"
              f"  median raw {rows[f't={t_}']['median_raw']:.2f}"
              f"  median z {rows[f't={t_}']['median_z']:.2f}")

    Hnat = np.asarray(np.load(t / "E_nat.npy", mmap_mode="r")[:200]).astype(np.float64)
    amn = np.asarray(np.load(t / "m_nat.npy", mmap_mode="r")[:200])
    rows["genuine(native)"] = zstats(Hnat, amn, mu, sd)
    rows["genuine(ref)"] = zstats(np.asarray(np.load(t / "E_ref.npy", mmap_mode="r")[:200])
                                 .astype(np.float64),
                                 np.asarray(np.load(t / "m_ref.npy", mmap_mode="r")[:200]), mu, sd)
    for k in ("genuine(native)", "genuine(ref)"):
        print(f"{k:<16} median raw {rows[k]['median_raw']:.2f}  median z {rows[k]['median_z']:.2f}")

    (cc.RESULTS / "diag_recolor.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("wrote results/diag_recolor.json")


if __name__ == "__main__":
    main()
