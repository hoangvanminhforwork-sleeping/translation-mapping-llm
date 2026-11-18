"""Mission 4 addendum - is a PARTIAL recolor enough?

Full covariance matching (t = 1) overcorrected: zdist_med 6.31 -> 61.59 against a
genuine 22.77, and retrieval fell. This sweeps t in [0, 1] on
    M_t = (1 - t) * M + t * recolor(M)
and reports where the distribution lands relative to the genuine shell, since
the target is zdist_med ~ 22.8 rather than zdist_med = maximum.

Fit on train tokens, evaluated on held-out val tokens, as in fix_maps.

Usage:
    ..\\research-scale-100k\\.venv\\Scripts\\python.exe -m src.recolor_sweep
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
VER02 = PROJ.parent / "research-phase01-ver02"
sys.path.insert(0, str(PROJ))

from src.fix_maps import (  # noqa: E402
    CACHE_TRN, CACHE_VAL, MAX_TOK, OUT, SEED, cos, embed, fit_procrustes,
    fit_recolor, retrieval_at_1, subsample)

TS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def stats(M, Y, mu_y, sd_y):
    mu_m = M.mean(0)
    d = np.linalg.norm((M - mu_y) / sd_y, axis=1)
    return {
        "cos_centre": round(cos(M - mu_m, Y - mu_y), 4),
        "zdist_med": round(float(np.median(d)), 3),
        "zdist_p05": round(float(np.quantile(d, 0.05)), 3),
        "zdist_p95": round(float(np.quantile(d, 0.95)), 3),
        "ret_at_1": round(retrieval_at_1(M, Y), 4),
    }


def main():
    t0 = time.time()
    Xtr, Ytr = embed("train", CACHE_TRN)
    Xva, Yva = embed("val", CACHE_VAL)
    Xtr, Ytr = subsample(Xtr, Ytr, MAX_TOK, SEED + 1)
    Xva, Yva = subsample(Xva, Yva, MAX_TOK, SEED)
    Xtr, Ytr = Xtr.astype(np.float64), Ytr.astype(np.float64)
    Xva, Yva = Xva.astype(np.float64), Yva.astype(np.float64)
    mu_y, sd_y = Ytr.mean(0), Ytr.std(0) + 1e-9

    mp = np.load(VER02 / "data" / "map_vi_en.npz")
    W, b = mp["W"].astype(np.float64), mp["b"].astype(np.float64)

    rows = []
    for name, fit_fn, apply_fn in (
        ("ridge", lambda: (W, b), lambda X: X @ W + b),
        ("procrustes", lambda: fit_procrustes(Xtr, Ytr),
         lambda X: X @ fit_procrustes(Xtr, Ytr)[0].astype(np.float64) + fit_procrustes(Xtr, Ytr)[1].astype(np.float64)),
    ):
        Mtr = apply_fn(Xtr)
        Mva = apply_fn(Xva)
        A, muY_r, muM_r = fit_recolor(Mtr, Ytr)
        Rtr = (Mtr - muM_r) @ A.astype(np.float64) + muY_r
        Rva = (Mva - muM_r) @ A.astype(np.float64) + muY_r
        print(f"\n{name}:")
        print(f"{'t':>5}{'cos_centre':>12}{'zdist_med':>11}{'p05':>8}{'p95':>8}{'ret@1':>8}")
        for t in TS:
            Mt = (1 - t) * Mva + t * Rva
            s = stats(Mt, Yva, mu_y, sd_y)
            rows.append({"map": name, "t": t, **s})
            print(f"{t:5.1f}{s['cos_centre']:12.4f}{s['zdist_med']:11.3f}"
                  f"{s['zdist_p05']:8.3f}{s['zdist_p95']:8.3f}{s['ret_at_1']:8.4f}")

    genuine = float(np.median(np.linalg.norm((Yva - mu_y) / sd_y, axis=1)))
    best = {}
    for name in ("ridge", "procrustes"):
        cand = [r for r in rows if r["map"] == name]
        # closest to the genuine shell, tie-broken by retrieval
        cand.sort(key=lambda r: (abs(r["zdist_med"] - genuine), -r["ret_at_1"]))
        best[name] = cand[0]
    print(f"\ngenuine zdist_med = {genuine:.3f}")
    for name, r in best.items():
        print(f"best {name:11} at t={r['t']:.1f}: zdist {r['zdist_med']:.2f}, "
              f"cos_centre {r['cos_centre']:.4f}, ret@1 {r['ret_at_1']:.4f}")

    (OUT / "recolor_sweep.json").write_text(json.dumps(
        {"genuine_zdist_med": round(genuine, 3), "sweep": rows,
         "best": best, "elapsed_s": round(time.time() - t0, 1)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote", OUT / "recolor_sweep.json", f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
