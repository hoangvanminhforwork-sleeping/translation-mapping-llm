"""512 LightGBM regressions: one boosted-tree model per target dimension.

Ridge fits all 512 output dimensions at once (W has 512 columns). LightGBM has no multi-output
mode, so the mission's "512 regressions" must be literal here: 512 independent boosters, each
predicting one coordinate of the target token vector from the 512-d source token vector.

Why this experiment exists (mission §3.1): a cosine of ~0.7 between mapped and true vectors is an
angle of ~45 deg, and a decoder should not tolerate that much. If the residual is a *nonlinear*
function of x, trees will find it and the geometry will improve; if the residual is irreducible
cross-lingual variance, trees will match ridge and stop — which is itself the answer.

Cost control: 120k sampled token pairs, 200 trees, 15 leaves, feature_fraction 0.3, 8 threads
(one per booster, so no memory duplication). Threads, not processes: LightGBM releases the GIL.

Usage
    python -m src.fit_lgbm [--tokens 120000] [--threads 8]
"""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import lightgbm as lgb
import numpy as np

from . import config as C
from .embed import Encoders, load_chosen
from .tokenbatch import AlignedBatcher

sys.stdout.reconfigure(encoding="utf-8")
D = 512
PARAMS = dict(objective="regression", metric="l2", learning_rate=0.1, num_leaves=15,
              min_data_in_leaf=100, feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1,
              verbose=-1, num_threads=1, seed=20261115)
MAX_BIN = 63
N_ROUNDS = 150


def _fit_block(dims, Xtr, Ytr, Xva, Yva, out):
    """One worker: bin the features once, then swap the label for each dimension.

    Binning 108k x 512 costs ~6 s and boosting costs ~16 s, so re-creating the Dataset per
    dimension would make binning the dominant term (measured: 40 s vs 16.5 s per dimension).
    """
    first = dims[0]
    ds = lgb.Dataset(Xtr, label=Ytr[:, first], free_raw_data=False, params={"max_bin": MAX_BIN})
    ds.construct()
    for k in dims:
        ds.set_label(Ytr[:, k])
        b = lgb.train(PARAMS, ds, num_boost_round=N_ROUNDS)
        pred = b.predict(Xva, num_threads=1)
        sse = float(((pred - Yva[:, k]) ** 2).sum())
        sst = float(((Yva[:, k] - Yva[:, k].mean()) ** 2).sum())
        imp = b.feature_importance("gain")
        out[k] = (b, 1.0 - sse / max(sst, 1e-12), int(b.num_trees()), int((imp > 0).sum()))


def fit_all(Xtr, Ytr, Xva, Yva, threads):
    out: dict[int, tuple] = {}
    chunks = [list(range(i, D, threads)) for i in range(threads)]  # strided: balanced
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(_fit_block, c, Xtr, Ytr, Xva, Yva, out) for c in chunks]
        for f in futs:
            f.result()
    return out


def collect_tokens(n_pairs: int, n_tokens: int, batch: int = 256, seed: int = 20261115):
    """Random pairs from the chosen shard's train slice -> aligned (X, Y) token matrices."""
    d = load_chosen(("vi", "en", "split"))
    idx = [i for i, s in enumerate(d["split"]) if s == "train"]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(idx), size=min(n_pairs, len(idx)), replace=False)
    vi = [d["vi"][idx[j]] for j in pick]
    en = [d["en"][idx[j]] for j in pick]
    print(f"sampling tokens from {len(vi):,} train pairs", flush=True)
    enc = Encoders()
    Xs, Ys, got = [], [], 0
    for s, nb, X, Y, _l in AlignedBatcher(enc, batch).batches(vi, en):
        Xs.append(X.cpu().numpy())
        Ys.append(Y.cpu().numpy())
        got += len(X)
        if got >= n_tokens:
            break
    X = np.concatenate(Xs)[:n_tokens]
    Y = np.concatenate(Ys)[:n_tokens]
    print(f"collected {len(X):,} token pairs")
    return X, Y


def main(n_tokens: int, threads: int):
    chosen = json.loads((C.DATA / "chosen_shard.json").read_text(encoding="utf-8"))["file"]
    out_dir = C.DATA / "lgbm"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_p = C.RESULTS / "lgbm_metrics.json"
    if meta_p.exists():
        print(f"{meta_p.name} exists — reusing (delete it to refit)")
        return

    X, Y = collect_tokens(n_pairs=12_000, n_tokens=n_tokens)
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(X))
    cut = int(0.9 * len(X))
    tr, va = perm[:cut], perm[cut:]
    Xtr, Ytr, Xva, Yva = X[tr], Y[tr], X[va], Y[va]
    print(f"fit {len(Xtr):,} / val {len(Xva):,} tokens, {D} boosters, {threads} threads", flush=True)

    t0 = time.time()
    res = fit_all(Xtr, Ytr, Xva, Yva, threads)
    r2 = np.array([res[k][1] for k in range(D)])
    ntree = np.array([res[k][2] for k in range(D)], dtype=np.int32)
    nfeat = np.array([res[k][3] for k in range(D)], dtype=np.int32)
    boosters = [res[k][0] for k in range(D)]
    print(f"512 boosters in {(time.time() - t0) / 60:.1f} min", flush=True)

    # per-dim comparison with ridge on the same tokens
    z = np.load(C.MAP["vi_en"])
    W, b0 = z["W"], z["b"]
    ridge_pred = Xva @ W + b0
    ridge_r2 = np.array([1 - ((ridge_pred[:, k] - Yva[:, k]) ** 2).sum() /
                         max(((Yva[:, k] - Yva[:, k].mean()) ** 2).sum(), 1e-12) for k in range(D)])
    mse_l = float(((np.stack([boosters[k].predict(Xva, num_threads=1) for k in range(D)], 1) - Yva) ** 2).sum(1).mean())
    mse_r = float(((ridge_pred - Yva) ** 2).sum(1).mean())
    cos_l = float((_unit(np.stack([boosters[k].predict(Xva, num_threads=1) for k in range(D)], 1)) * _unit(Yva)).sum(1).mean())
    cos_r = float((_unit(ridge_pred) * _unit(Yva)).sum(1).mean())
    print(f"token MSE  ridge {mse_r:.4f}  lgbm {mse_l:.4f}   cosine ridge {cos_r:.4f}  lgbm {cos_l:.4f}")

    with gzip.open(out_dir / "vi_en.pkl.gz", "wb", compresslevel=6) as f:
        pickle.dump(boosters, f, protocol=5)
    meta = {"shard": chosen, "n_tokens_train": int(len(Xtr)), "n_tokens_val": int(len(Xva)),
            "n_rounds": N_ROUNDS, "params": {**PARAMS, "max_bin": MAX_BIN}, "direction": "vi_en",
            "r2_lgbm_mean": float(r2.mean()), "r2_ridge_mean": float(ridge_r2.mean()),
            "r2_lgbm_per_dim": r2.tolist(), "r2_ridge_per_dim": ridge_r2.tolist(),
            "trees_per_dim": ntree.tolist(), "features_used_per_dim": nfeat.tolist(),
            "token_mse_ridge": mse_r, "token_mse_lgbm": mse_l,
            "token_cosine_ridge": cos_r, "token_cosine_lgbm": cos_l,
            "wall_min": round((time.time() - t0) / 60, 1)}
    meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with (C.RESULTS / "lgbm_vs_ridge.csv").open("w", encoding="utf-8") as f:
        f.write("dim,r2_ridge,r2_lgbm,delta,trees,features_used\n")
        for k in range(D):
            f.write(f"{k},{ridge_r2[k]:.6f},{r2[k]:.6f},{r2[k] - ridge_r2[k]:+.6f},{ntree[k]},{nfeat[k]}\n")
    print(f"wrote {meta_p} and lgbm_vs_ridge.csv in {meta['wall_min']} min")


def _unit(a):
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)


class LgbmMap:
    """Loads the 512 boosters and applies them as a vector map (used by generate.py)."""

    def __init__(self, direction: str = "vi_en"):
        p = C.DATA / "lgbm" / f"{direction}.pkl.gz"
        with gzip.open(p, "rb") as f:
            self.boosters = pickle.load(f)
        self.d = len(self.boosters)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        # prediction parallelises over rows, so 4 threads cuts the wall time ~3x
        return np.stack([b.predict(X, num_threads=4) for b in self.boosters], 1).astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=120_000)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    main(a.tokens, a.threads)
