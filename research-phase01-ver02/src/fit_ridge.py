"""Per-token ridge, both directions, with a closed-form per-dimension R^2 decomposition.

The mission asks for "512 linear regressions, vector -> one dimension". Ridge fitted jointly is
*exactly* those 512 regressions: with centred moments A = X^T X, B = X^T Y, C = Y^T Y,

    W = (A + lam I)^-1 B ,   b = mean(Y) - mean(X) W

and column k of W is the least-squares coefficient vector of the regression y_k ~ x. Solving once
is therefore not a shortcut, it is the same estimator with 512 right-hand sides. The per-dimension
R^2 that a 512-times loop would report is recovered from the moments without touching activations:

    SSE_k = (W^T A W - 2 W^T B + C)_kk ,  SST_k = C_kk ,  R^2_k = 1 - SSE_k / SST_k

Usage
    python -m src.fit_ridge
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from . import config as C
from .embed import LAMBDAS, Moments

sys.stdout.reconfigure(encoding="utf-8")
D = 512


def _solve(m: Moments, lam: float):
    n = float(m.n)
    mx, my = m.sx.cpu() / n, m.sy.cpu() / n
    Ac = (m.A.cpu() - n * torch.outer(mx, mx)).numpy()
    Bc = (m.B.cpu() - n * torch.outer(mx, my)).numpy()
    W = np.linalg.solve(Ac + lam * np.eye(D), Bc)
    b = (my - mx @ torch.from_numpy(W)).numpy()
    return W, b


def _mse(m: Moments, W, b):
    n = float(m.n)
    mx, my = m.sx.cpu() / n, m.sy.cpu() / n
    Wt, bt = torch.from_numpy(W), torch.from_numpy(b)
    Ac = m.A.cpu() - n * torch.outer(mx, mx)
    Bc = m.B.cpu() - n * torch.outer(mx, my)
    Cc = m.C.cpu() - n * torch.outer(my, my)
    tr = torch.trace(Wt.T @ Ac @ Wt) - 2 * torch.trace(Wt.T @ Bc) + torch.trace(Cc)
    res = bt + mx @ Wt - my
    return float((tr / n).item() + (res @ res).item())


def per_dim_r2(m: Moments, W, b):
    """R^2 for each of the 512 regressions, from the moments only."""
    n = float(m.n)
    mx, my = m.sx.cpu() / n, m.sy.cpu() / n
    Wt, bt = torch.from_numpy(W), torch.from_numpy(b)
    Ac = m.A.cpu() - n * torch.outer(mx, mx)
    Bc = m.B.cpu() - n * torch.outer(mx, my)
    Cc = m.C.cpu() - n * torch.outer(my, my)
    M = Wt.T @ Ac @ Wt - 2 * Wt.T @ Bc + Cc
    sse = torch.diagonal(M)
    res = bt + mx @ Wt - my
    sse = sse + n * res * res
    sst = torch.diagonal(Cc)
    return (1.0 - sse / sst.clamp(min=1e-12)).numpy()


def swapped(m: Moments) -> Moments:
    s = Moments(device="cpu")
    s.A, s.B, s.C = m.C, m.B.T.clone(), m.A
    s.sx, s.sy, s.n, s.pairs = m.sy, m.sx, m.n, m.pairs
    return s


def main():
    stats = C.DATA / "stats"
    tr_p = next(stats.glob("*_train.npz"), None)
    va_p = next(stats.glob("*_val.npz"), None)
    if tr_p is None or va_p is None:
        raise SystemExit("missing moments; run `python -m src.embed moments --file <shard>` first")
    tr, va = Moments.load(tr_p), Moments.load(va_p)
    print(f"train: {tr.pairs:,} pairs, {tr.n:,} token pairs ({tr_p.name})")
    print(f"val:   {va.pairs:,} pairs, {va.n:,} token pairs ({va_p.name})")

    rows, per_dim = [], {}
    for direction, (t, v, out_path) in {
        "vi_en": (tr, va, C.MAP["vi_en"]),
        "en_vi": (swapped(tr), swapped(va), C.MAP["en_vi"]),
    }.items():
        best, base = None, None
        for lam in LAMBDAS:
            W, b = _solve(t, lam)
            mse_tr, mse_va = _mse(t, W, b), _mse(va, W, b)
            rows.append((direction, lam, mse_tr, mse_va))
            if base is None:
                base = mse_va
            print(f"  {direction} lambda {lam:>8g}  mse_train {mse_tr:.6f}  mse_val {mse_va:.6f}")
            if best is None or mse_va < best[1]:
                best = (lam, mse_va, W, b)
        lam, mse_va, W, b = best
        gain = (base - mse_va) / base
        if gain < 1e-5:
            lam, W, b = LAMBDAS[0], *_solve(t, LAMBDAS[0])
            mse_va = _mse(va, W, b)
            print(f"{direction}: sweep flat (gain {gain:.2e}) -> lambda=0")
        else:
            print(f"{direction}: lambda={lam} (val mse {mse_va:.6f}, gain {gain:.2e})")
        np.savez_compressed(out_path, W=W.astype(np.float32), b=b.astype(np.float32), lam=lam,
                            n_pairs=t.pairs, n_tokens=t.n, direction=direction)
        r2_va = per_dim_r2(v, W, b)
        r2_tr = per_dim_r2(t, W, b)
        per_dim[direction] = {"r2_train": r2_tr.tolist(), "r2_val": r2_va.tolist()}
        print(f"{direction}: saved {out_path.name}; per-dim R2 val mean {r2_va.mean():.4f} "
              f"median {np.median(r2_va):.4f} min {r2_va.min():.4f} max {r2_va.max():.4f}")

    (C.RESULTS / "ridge_sweep.csv").write_text(
        "direction,lambda,mse_train,mse_val\n"
        + "\n".join(f"{d},{a},{x:.8f},{y:.8f}" for d, a, x, y in rows) + "\n", encoding="utf-8")
    (C.RESULTS / "per_dim_r2.json").write_text(json.dumps(per_dim), encoding="utf-8")
    print(f"wrote {C.RESULTS / 'ridge_sweep.csv'} and per_dim_r2.json")


if __name__ == "__main__":
    main()
