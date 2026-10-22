"""CLI for experiment 02.  Every stage is idempotent and skippable.

  python -m src.pipeline cache     # encoders -> artifacts/vec_*.npy
  python -m src.pipeline maps      # ridge + MLP per direction x pooling
  python -m src.pipeline invert    # Vector2 -> Token2 readout (the experiment)
  python -m src.pipeline metrics   # scores -> results/
  python -m src.pipeline all
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import data as D
from . import encode as E
from . import invert as I
from . import maps as M
from . import metrics as MET

# Vietnamese on a cp1252 console: reconfigure once.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- cache
def stage_cache(force: bool = False):
    df = D.pairs()
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    info = {"models": {"en": C.EN_MODEL, "vi": C.VI_MODEL}, "poolings": {}, "pairs_sha256": D.sha256(C.PAIRS)}
    for lang, name in (("en", C.EN_MODEL), ("vi", C.VI_MODEL)):
        tok, model = E.load(name, dtype=torch.float16)
        try:
            for pooling in C.POOLINGS:
                path = E.vector_path(lang, pooling)
                v = E.cached(df[lang].tolist(), lang, pooling, tok, model, force=force)
                info["poolings"][f"{lang}/{pooling}"] = {
                    "shape": list(v.shape), "norm_mean": float(np.linalg.norm(v, axis=1).mean())}
                log(f"  {lang}/{pooling} -> {path.name} {v.shape}")
        finally:
            E.free(tok, model)
    (C.RESULTS / "cache_summary.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


# ---------------------------------------------------------------- maps
def map_paths(direction, pooling):
    return (C.MAPS / f"{direction}_{pooling}_ridge.npz",
            C.MAPS / f"{direction}_{pooling}_ann.pt")


def stage_maps(force: bool = False):
    df = D.pairs()
    train, test = D.split(df)
    C.MAPS.mkdir(parents=True, exist_ok=True)
    rows = []
    for direction, src, tgt in C.DIRECTIONS:
        for pooling in C.POOLINGS:
            rp, ap = map_paths(direction, pooling)
            Xtr = np.load(E.vector_path(src, pooling))[: C.N_TRAIN]
            Ytr = np.load(E.vector_path(tgt, pooling))[: C.N_TRAIN]
            Xte = np.load(E.vector_path(src, pooling))[C.N_TRAIN:]
            Yte = np.load(E.vector_path(tgt, pooling))[C.N_TRAIN:]
            nval = int(len(Xtr) * C.VAL_FRACTION)
            sl = slice(0, len(Xtr) - nval)
            vl = slice(len(Xtr) - nval, len(Xtr))

            if rp.exists() and not force:
                ridge = M.ridge_load(rp)
            else:
                ridge = M.ridge_fit(Xtr[sl], Ytr[sl], C.RIDGE_ALPHA)
                M.ridge_save(rp, ridge)
                log(f"  ridge fit {direction}/{pooling}")
            if ap.exists() and not force:
                mlp, scalers = M.mlp_load(ap)
            else:
                mlp, scalers, curve = M.train_mlp(Xtr[sl], Ytr[sl], Xtr[vl], Ytr[vl], verbose=False)
                M.mlp_save(ap, mlp, scalers)
                pd.DataFrame(curve).to_csv(C.RESULTS / f"ann_curve_{direction}_{pooling}.csv", index=False)
                log(f"  ann fit {direction}/{pooling} (epochs={len(curve)})")

            for method, pred in (("ridge", M.ridge_apply(Xte, ridge)),
                                 ("ann", M.apply_mlp(mlp, scalers, Xte))):
                pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
                yn = Yte / (np.linalg.norm(Yte, axis=1, keepdims=True) + 1e-12)
                cos = float((pn * yn).sum(1).mean())
                rows.append({"direction": direction, "pooling": pooling, "method": method,
                             "cosine_test": cos,
                             "norm_ratio": float(np.linalg.norm(pred, axis=1).mean()
                                                 / np.linalg.norm(Yte, axis=1).mean())})
                log(f"  {direction}/{pooling}/{method} test cosine {cos:.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS / "map_sanity.csv", index=False)
    return out


# ---------------------------------------------------------------- invert
def arm_targets(direction, pooling, n_eval: int):
    src, tgt = direction[:2], direction[3:]
    ridge_path, ann_path = map_paths(direction, pooling)
    Xte = np.load(E.vector_path(src, pooling))[C.N_TRAIN:][:n_eval]
    Yte = np.load(E.vector_path(tgt, pooling))[C.N_TRAIN:][:n_eval]
    ridge = M.ridge_load(ridge_path)
    mlp, scalers = M.mlp_load(ann_path)
    perm = D.shuffled_index(len(Xte))
    return {
        "roundtrip": Yte,
        "identity": Xte,
        "ridge": M.ridge_apply(Xte, ridge).astype(np.float32),
        "ann": M.apply_mlp(mlp, scalers, Xte),
        "shuffled": M.ridge_apply(Xte[perm], ridge).astype(np.float32),
    }, Yte


def stage_invert(n_eval: int = C.N_EVAL, steps: int = C.INV_STEPS, force: bool = False):
    df = D.pairs()
    test = df[df.split == "test"].reset_index(drop=True).iloc[:n_eval]
    rows = []
    for direction, src, tgt in C.DIRECTIONS:
        tok, model = E.load(C.EN_MODEL if tgt == "en" else C.VI_MODEL, dtype=torch.float32)
        try:
            for pooling in C.POOLINGS:
                out = C.RESULTS / f"hyp_{direction}_{pooling}.parquet"
                if out.exists() and not force:
                    log(f"  skip {out.name} (exists)")
                    rows.append(pd.read_parquet(out))
                    continue
                tgts, _ = arm_targets(direction, pooling, n_eval)
                mu = np.load(E.vector_path(tgt, pooling))[: C.N_TRAIN].mean(0)
                log(f"  invert {direction}/{pooling}: {len(test)} items x {len(tgts)} arms "
                    f"x {steps} steps")
                readout = I.Readout(tok, model, pooling)
                texts, disc = readout.invert(tgts, test[tgt].tolist(), mu=mu, steps=steps)
                for a in ("roundtrip", "ridge"):
                    log(f"    [{a}] ref: {test[tgt].iloc[0][:70]}")
                    log(f"    [{a}] hyp: {texts[a][0][:70]}")
                part = pd.DataFrame({
                    "pair_id": np.repeat(test.pair_id.values, len(C.ARMS)),
                    "direction": direction, "pooling": pooling,
                    "arm": np.tile(C.ARMS, len(test)),
                    "hyp": sum((texts[a] for a in C.ARMS), []),
                    "ref": np.tile(test[tgt].values, len(C.ARMS)),
                    "disc_cos": sum((disc[a].tolist() for a in C.ARMS), []),
                })
                part.to_parquet(out, index=False)
                for a in C.ARMS:
                    sub = part[part.arm == a]
                    log(f"    {a:9s} disc_cos {sub.disc_cos.mean():.3f}  empty {sub.hyp.str.strip().eq('').mean():.2f}")
                rows.append(part)
        finally:
            E.free(tok, model)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------- metrics
def stage_nn(n_eval: int = C.N_EVAL):
    """Nearest-neighbour readout: a real 'translation memory' baseline.

    Gallery = the 10,000 TRAIN target sentences (open set w.r.t. test), so the
    hypothesis is never the gold sentence.  Cheap alternative to inversion.
    """
    df = D.pairs()
    test = df[df.split == "test"].reset_index(drop=True).iloc[:n_eval]
    res = []
    for direction, src, tgt in C.DIRECTIONS:
        for pooling in C.POOLINGS:
            G = np.load(E.vector_path(tgt, pooling))[: C.N_TRAIN]
            mu = G.mean(0)
            Gn = (G - mu) / (np.linalg.norm(G - mu, axis=1, keepdims=True) + 1e-12)
            gtext = df[df.split == "train"].reset_index(drop=True)[tgt].tolist()
            tgts, _ = arm_targets(direction, pooling, n_eval)
            for arm in C.ARMS:
                Q = tgts[arm] - mu
                Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
                sims = Qn @ Gn.T
                best = sims.argmax(1)
                hyp = [gtext[i] for i in best]
                row = {"direction": direction, "pooling": pooling, "arm": arm,
                       "retrieval_cos": float(sims[np.arange(len(best)), best].mean())}
                row.update(MET.compute(hyp, test[tgt].tolist(), C.TGT_LANG[direction]))
                res.append(row)
                log(f"  nn {direction}/{pooling}/{arm:9s} BLEU {row['bleu']:5.2f} "
                    f"chrF {row['chrf']:5.2f} EM {row['exact_match']:.3f}")
    out = pd.DataFrame(res)
    out.to_csv(C.RESULTS / "nn_metrics.csv", index=False)
    return out


def stage_metrics():
    res = []
    for direction, src, tgt in C.DIRECTIONS:
        for pooling in C.POOLINGS:
            path = C.RESULTS / f"hyp_{direction}_{pooling}.parquet"
            if not path.exists():
                log(f"  missing {path.name}")
                continue
            h = pd.read_parquet(path)
            for arm in C.ARMS:
                sub = h[h.arm == arm]
                row = {"direction": direction, "pooling": pooling, "arm": arm,
                       "disc_cos": float(sub.disc_cos.mean())}
                row.update(MET.compute(sub.hyp.tolist(), sub.ref.tolist(), C.TGT_LANG[direction]))
                res.append(row)
                log(f"  {direction}/{pooling}/{arm:9s} BLEU {row['bleu']:6.2f}  "
                    f"chrF {row['chrf']:6.2f}  EM {row['exact_match']:.3f}")
    out = pd.DataFrame(res)
    out.to_csv(C.RESULTS / "token_metrics.csv", index=False)
    return out


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    args = sys.argv[1:]
    stage = args[0] if args else "all"
    n_eval = int(next((a.split("=")[1] for a in args if a.startswith("--n=")), C.N_EVAL))
    steps = int(next((a.split("=")[1] for a in args if a.startswith("--steps=")), C.INV_STEPS))
    force = "--force" in args

    t0 = time.time()
    if stage in ("cache", "all"):
        log("cache"); stage_cache(force=force)
    if stage in ("maps", "all"):
        log("maps"); stage_maps(force=force)
    if stage in ("nn", "all"):
        log("nn readout"); stage_nn(n_eval=n_eval)
    if stage in ("invert", "all-fast"):
        log("invert"); stage_invert(n_eval=n_eval, steps=steps, force=force)
    if stage in ("metrics", "all"):
        log("metrics"); stage_metrics()
    log(f"done in {time.time() - t0:.0f}s")
