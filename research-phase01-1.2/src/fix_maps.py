"""Mission 4 - diagnose the mapped representation, then try two cheap repairs.

Diagnosis (not a "fix"): ver02's mapped vectors are COLLAPSED toward the target
mean. Median z-distance from the target mean is ~6.3 against ~22.8 for genuine
vectors, and a constant vector already reproduces ~80% of the reported token
cosine. This script quantifies that and asks whether linear algebra can undo it.

Repairs compared, all fitted on TRAIN tokens and scored on the held-out VAL
tokens so nothing is fitted on what it is measured against:

  ridge      the existing ver02 map (data/map_vi_en.npz)
  procrustes orthogonal R fitted by SVD (Zhou 2026: Procrustes beats affine and
             MLP on retrieval for every model pair; an orthogonal map cannot
             shrink lengths, so it cannot collapse)
  recolor    ridge followed by a covariance-matching transform
             M' = Sigma_Y^1/2 Sigma_M^-1/2 (M - mu_M) + mu_Y
  proc+color procrustes followed by the same recolor

Controls: `constant` (predict mu_Y for every token) and `identity` (raw source).

Metrics, on held-out tokens only:
  cos_abs      mean cosine(M, Y)                    - the number ver02 reports
  cos_centre   mean cosine(M-mu, Y-mu)              - directional signal only
  const_cos    mean cosine(mu_M broadcast, Y)       - how much a constant gets
  zdist_med    median distance from mu_Y in the target z-frame
  zdist_p05/p95
  ret_at_1     token retrieval@1: nearest target vector is the true counterpart

Usage:
    ..\\research-scale-100k\\.venv\\Scripts\\python.exe -m src.fix_maps
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
VER02 = PROJ.parent / "research-phase01-ver02"

_spec = importlib.util.spec_from_file_location(
    "v2", VER02 / "src" / "__init__.py",
    submodule_search_locations=[str(VER02 / "src")])
_v2 = importlib.util.module_from_spec(_spec)
sys.modules["v2"] = _v2
_spec.loader.exec_module(_v2)

OUT = PROJ / "results"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_VAL = PROJ / "data" / "anomaly_vectors.npz"
CACHE_TRN = PROJ / "data" / "train_vectors.npz"

N_SENT = 6000
SEED = 7
MAX_TOK = 30000        # cap tokens to keep peak RAM under ~1 GB
RET_N = 4000           # retrieval@1 bank size


# ------------------------------------------------------------- embeddings
def embed(split: str, cache: Path):
    if cache.exists():
        d = np.load(cache)
        print(f"{split}: cached X={d['X'].shape} Y={d['Y'].shape}")
        return d["X"], d["Y"]
    import pandas as pd
    from v2.config import DATA as V2DATA
    from v2.embed import Encoders
    from v2.tokenbatch import AlignedBatcher

    df = pd.read_parquet(V2DATA / "chosen.parquet", columns=["split", "vi", "en"])
    part = df[df["split"] == split]
    rng = np.random.default_rng(SEED if split == "val" else SEED + 1)
    idx = np.sort(rng.choice(len(part), size=min(N_SENT, len(part)), replace=False))
    part = part.iloc[idx]
    vi, en = part["vi"].tolist(), part["en"].tolist()
    enc = Encoders()
    Xs, Ys, t0 = [], [], time.time()
    for start, n, Xb, Yb, lens in AlignedBatcher(enc, batch=64, src="vi", tgt="en").batches(vi, en):
        Xs.append(Xb.cpu().numpy())
        Ys.append(Yb.cpu().numpy())
        if (start // 64) % 25 == 0:
            print(f"  {start}/{len(vi)} sentences, {sum(len(a) for a in Xs)} tokens, "
                  f"{time.time() - t0:.0f}s", flush=True)
    X = np.concatenate(Xs).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.float32)
    np.savez_compressed(cache, X=X, Y=Y)
    print(f"{split}: {len(X)} token pairs in {time.time() - t0:.0f}s")
    return X, Y


def subsample(X, Y, n, seed):
    if len(X) <= n:
        return X, Y
    rng = np.random.default_rng(seed)
    i = np.sort(rng.choice(len(X), size=n, replace=False))
    return X[i], Y[i]


# ------------------------------------------------------------------- maps
def fit_procrustes(X, Y):
    """Orthogonal R minimising ||Xc R - Yc||, with centring."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
    R = U @ Vt
    b = my - mx @ R
    return R.astype(np.float32), b.astype(np.float32)


def fit_recolor(M, Y):
    """Covariance-matching transform so cov(M') ~ cov(Y)."""
    mu_m, mu_y = M.mean(0), Y.mean(0)
    Cm = np.cov(M - mu_m, rowvar=False)
    Cy = np.cov(Y - mu_y, rowvar=False)
    Cm += np.eye(len(mu_m)) * 1e-3 * np.trace(Cm) / len(mu_m)
    Cy += np.eye(len(mu_y)) * 1e-3 * np.trace(Cy) / len(mu_y)

    def sqrtm_inv(C):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-8, None)
        return (V / np.sqrt(w)) @ V.T

    def sqrtm(C):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 0, None)
        return (V * np.sqrt(w)) @ V.T

    A = sqrtm(Cy) @ sqrtm_inv(Cm)
    return A.astype(np.float32), mu_y.astype(np.float32), mu_m.astype(np.float32)


# ---------------------------------------------------------------- metrics
def cos(A, B):
    return float(np.mean(np.sum(A * B, 1) /
                         (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12)))


def retrieval_at_1(M, Y, n=RET_N, seed=SEED):
    """Nearest target vector in the val bank == the true counterpart?"""
    rng = np.random.default_rng(seed)
    if len(M) > n:
        q = np.sort(rng.choice(len(M), size=n, replace=False))
        M2, Y2 = M[q], Y[q]
    else:
        M2, Y2 = M, Y
    M2 = M2 / (np.linalg.norm(M2, axis=1, keepdims=True) + 1e-12)
    Y2 = Y2 / (np.linalg.norm(Y2, axis=1, keepdims=True) + 1e-12)
    best = None
    for i in range(0, len(M2), 500):
        sim = M2[i:i + 500] @ Y2.T
        idx = sim.argmax(1)
        hit = (idx == np.arange(i, i + len(sim)))
        best = hit if best is None else np.concatenate([best, hit])
    return float(best.mean())


def evaluate(name, M, Y, mu_y, sd_y, extra=None):
    mu_m = M.mean(0)
    z = (M - mu_y) / sd_y
    zy = (Y - mu_y) / sd_y
    d = np.linalg.norm(z, axis=1)
    dy = np.linalg.norm(zy, axis=1)
    r = {
        "map": name,
        "cos_abs": round(cos(M, Y), 4),
        "cos_centre": round(cos(M - mu_m, Y - mu_y), 4),
        "const_cos": round(cos(np.broadcast_to(mu_m, Y.shape), Y), 4),
        "zdist_med": round(float(np.median(d)), 3),
        "zdist_p05": round(float(np.quantile(d, 0.05)), 3),
        "zdist_p95": round(float(np.quantile(d, 0.95)), 3),
        "genuine_zdist_med": round(float(np.median(dy)), 3),
        "ret_at_1": round(retrieval_at_1(M, Y), 4),
    }
    if extra:
        r.update(extra)
    return r


def main():
    t0 = time.time()
    Xtr, Ytr = embed("train", CACHE_TRN)
    Xva, Yva = embed("val", CACHE_VAL)
    Xtr, Ytr = subsample(Xtr, Ytr, MAX_TOK, SEED + 1)
    Xva, Yva = subsample(Xva, Yva, MAX_TOK, SEED)
    Xtr = Xtr.astype(np.float64)
    Ytr = Ytr.astype(np.float64)
    Xva = Xva.astype(np.float64)
    Yva = Yva.astype(np.float64)
    print(f"fit tokens={len(Xtr)}  eval tokens={len(Xva)}")

    mu_y, sd_y = Ytr.mean(0), Ytr.std(0) + 1e-9

    mp = np.load(VER02 / "data" / "map_vi_en.npz")
    W = mp["W"].astype(np.float64)
    b = mp["b"].astype(np.float64)
    M_ridge_tr = Xtr @ W + b
    M_ridge = Xva @ W + b

    R, br = fit_procrustes(Xtr, Ytr)
    M_proc_tr = Xtr @ R.astype(np.float64) + br.astype(np.float64)
    M_proc = Xva @ R.astype(np.float64) + br.astype(np.float64)

    A, muY_r, muM_r = fit_recolor(M_ridge_tr, Ytr)
    M_ridge_col = (M_ridge - muM_r) @ A.astype(np.float64) + muY_r
    A2, muY_p, muM_p = fit_recolor(M_proc_tr, Ytr)
    M_proc_col = (M_proc - muM_p) @ A2.astype(np.float64) + muY_p

    rows = [
        evaluate("constant (mu_Y)", np.broadcast_to(mu_y, Yva.shape).copy(), Yva, mu_y, sd_y),
        evaluate("identity (no map)", Xva, Yva, mu_y, sd_y),
        evaluate("ridge  [ver02]", M_ridge, Yva, mu_y, sd_y),
        evaluate("ridge + recolor", M_ridge_col, Yva, mu_y, sd_y),
        evaluate("procrustes", M_proc, Yva, mu_y, sd_y),
        evaluate("procrustes + recolor", M_proc_col, Yva, mu_y, sd_y),
    ]

    hdr = ["map", "cos_abs", "cos_centre", "const_cos", "zdist_med",
           "zdist_p05", "zdist_p95", "ret_at_1"]
    print("\n" + "".join(f"{h:>20}" for h in hdr))
    for r in rows:
        print("".join(f"{str(r[h]):>20}" for h in hdr))
    print(f"\ngenuine zdist median = {rows[0]['genuine_zdist_med']}")

    out = {
        "protocol": {
            "fit_on": "train tokens (6000 sentences, subsampled)",
            "eval_on": "held-out val tokens (6000 sentences, subsampled)",
            "n_fit_tokens": int(len(Xtr)), "n_eval_tokens": int(len(Xva)),
            "direction": "vi->en", "seed": SEED,
            "note": "cos_abs is the metric ver02 reports. const_cos is what a "
                    "constant vector scores on the same metric.",
        },
        "results": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "map_repairs.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote", OUT / "map_repairs.json", f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

