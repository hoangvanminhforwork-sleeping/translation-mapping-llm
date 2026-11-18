"""Mission 5 - one-class anomaly detection on the mapped representation.

Hypothesis under test (missions.md section 5): the vector produced by the linear
map is NOT in the decoder's familiar distribution. A one-class detector trained
only on genuine target-side vectors should therefore flag mapped vectors far
more often than it flags held-out genuine vectors.

Protocol, exactly as the mission specifies
    * fit on 90% of NORMAL vectors (real target-side encoder states)
    * threshold calibrated so that 10% of the fitted set is flagged
    * test A: the held-out 10% of normal vectors  -> expect ~10% flagged
    * test B: ALL transformed vectors              -> the number of interest

Detectors: Isolation Forest, Gaussian Mixture, and a small autoencoder
(reconstruction error). Everything runs on CPU except the embedding pass.

Controls, so the number is interpretable rather than merely large:
    identity  - the source vector used unchanged (no map)
    mean      - the source vector replaced by the target mean
A good result is: mapped > identity >> mean, i.e. the map does not remove the
shift, but it is not the cause of it either.

Usage (from research-phase01-1.2):
    ..\\research-scale-100k\\.venv\\Scripts\\python.exe -m src.anomaly
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

# research-phase01-ver02/src and research-phase01-1.2/src are both called `src`,
# so the old package is loaded under the alias `v2`; its modules use relative
# imports (`from . import config`), which only work inside a real package.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "v2", VER02 / "src" / "__init__.py",
    submodule_search_locations=[str(VER02 / "src")])
_v2 = importlib.util.module_from_spec(_spec)
sys.modules["v2"] = _v2
_spec.loader.exec_module(_v2)

OUT = PROJ / "results"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = PROJ / "data" / "anomaly_vectors.npz"

N_SENT = 6000          # held-out sentences to embed
SEED = 7


# --------------------------------------------------------------- embeddings
def build_vectors():
    """(X source, Y target) aligned token vectors on the held-out split."""
    if CACHE.exists():
        d = np.load(CACHE)
        print(f"cached vectors: X={d['X'].shape} Y={d['Y'].shape}")
        return d["X"], d["Y"]
    import pandas as pd
    import torch

    from v2.config import DATA as V2DATA
    from v2.embed import Encoders
    from v2.tokenbatch import AlignedBatcher

    df = pd.read_parquet(V2DATA / "chosen.parquet", columns=["split", "vi", "en"])
    val = df[df["split"] == "val"]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(val), size=min(N_SENT, len(val)), replace=False)
    val = val.iloc[np.sort(idx)]
    vi, en = val["vi"].tolist(), val["en"].tolist()
    print(f"held-out sentences: {len(vi)}")

    enc = Encoders()
    Xs, Ys, nx = [], [], 0
    t0 = time.time()
    # batches yields (start, n, X, Y, lens) where X and Y are ALREADY flattened
    # to (n_tokens_in_batch, 512) by h_src[keep] / Y[keep] inside the batcher.
    for start, n, Xb, Yb, lens in AlignedBatcher(enc, batch=64, src="vi", tgt="en").batches(vi, en):
        Xs.append(Xb.cpu().numpy())
        Ys.append(Yb.cpu().numpy())
        nx += len(Xb)
        if (start // 64) % 20 == 0:
            print(f"  {start}/{len(vi)} sentences, {nx} token pairs, "
                  f"{time.time() - t0:.0f}s", flush=True)
    X = np.concatenate(Xs).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.float32)
    print(f"token pairs: {len(X)}  ({time.time() - t0:.0f}s)")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, X=X, Y=Y)
    return X, Y


# ---------------------------------------------------------------- detectors
def calibrate(scores_fit, frac=0.10):
    """Threshold that flags `frac` of the fitting scores."""
    return float(np.quantile(scores_fit, 1.0 - frac))


def flag_rate(scores, thr):
    return float((scores > thr).mean())


def run_iforest(train, tests, seed=SEED):
    from sklearn.ensemble import IsolationForest
    m = IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1,
                        max_samples=min(256, len(train)))
    m.fit(train)
    out = {}
    fit_s = -m.score_samples(train)
    thr = calibrate(fit_s)
    for name, t in tests.items():
        out[name] = flag_rate(-m.score_samples(t), thr)
    return out


def run_gmm(train, tests, comps=8, seed=SEED):
    from sklearn.mixture import GaussianMixture
    m = GaussianMixture(n_components=comps, covariance_type="diag",
                        random_state=seed, max_iter=200, reg_covar=1e-4)
    m.fit(train)
    fit_s = -m.score_samples(train)
    thr = calibrate(fit_s)
    return {name: flag_rate(-m.score_samples(t), thr) for name, t in tests.items()}


def run_autoencoder(train, tests, seed=SEED, epochs=40):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    d = train.shape[1]
    net = nn.Sequential(
        nn.Linear(d, 256), nn.GELU(),
        nn.Linear(256, 64), nn.GELU(),
        nn.Linear(64, 256), nn.GELU(),
        nn.Linear(256, d),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    T = torch.from_numpy(train)
    n = len(T)
    bs = 512
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            b = T[idx]
            opt.zero_grad()
            out = net(b)
            loss = lossf(out, b)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if ep % 10 == 9:
            print(f"    ae epoch {ep + 1}: loss {tot / n:.5f}", flush=True)

    def recon(x):
        with torch.no_grad():
            return ((net(torch.from_numpy(x)) - torch.from_numpy(x)) ** 2).mean(1).numpy()

    fit_s = recon(train)
    thr = calibrate(fit_s)
    return {name: flag_rate(recon(t), thr) for name, t in tests.items()}


def main():
    t0 = time.time()
    X, Y = build_vectors()
    n = len(Y)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_fit = int(0.9 * n)
    fit_idx, hold_idx = perm[:n_fit], perm[n_fit:]

    mp = np.load(VER02 / "data" / "map_vi_en.npz")
    W, b = mp["W"].astype(np.float32), mp["b"].astype(np.float32)
    M = X @ W.T + b                        # the mapped representation
    ident = X                              # control: no map at all
    meanv = np.broadcast_to(Y.mean(0, keepdims=True), X.shape).astype(np.float32)

    train = Y[fit_idx]
    tests = {
        "normal_holdout": Y[hold_idx],
        "mapped": M,
        "identity_src": ident,
        "mean_baseline": meanv,
    }

    # standardise with genuine-vector statistics; a real decoder sees none of it,
    # but it makes the three detectors comparable
    mu, sd = train.mean(0), train.std(0) + 1e-6
    f = lambda a: ((a - mu) / sd).astype(np.float32)
    train_s = f(train)
    tests_s = {k: f(v) for k, v in tests.items()}

    results = {}
    for name, fn in (("isolation_forest", run_iforest),
                     ("gmm", run_gmm),
                     ("autoencoder", run_autoencoder)):
        print(f"  fitting {name} ...", flush=True)
        r = fn(train_s, tests_s)
        results[name] = r
        print(f"    {name}: " + "  ".join(f"{k}={v:.3f}" for k, v in r.items()),
              flush=True)

    summary = {
        "protocol": {
            "n_tokens": int(n), "n_fit": int(n_fit), "n_holdout": int(n - n_fit),
            "detector_fit_on": "genuine target-side encoder states (vi->en)",
            "threshold": "quantile of fit scores, calibrated to flag 10% of fit",
            "standardisation": "z-scored with genuine-vector mean/std",
            "map": str(VER02 / "data" / "map_vi_en.npz"),
        },
        "flag_rates": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "anomaly_detection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nwrote", OUT / "anomaly_detection.json")
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

