"""Fit the linear maps between the two representation spaces.

Two directions, one closed-form ridge fit each, swept over lambda on the val split:

    f: vit5_vi -> flan_en   (vi -> en)
    g: flan_en -> vit5_vi   (en -> vi)

Reported per direction, because a small MSE does not imply a usable vector:
    mse, cosine, retrieval@1 (val queries against the 97,743 train targets), and two controls
    (identity = use the source vector unchanged; mean = predict the target mean).
"""

from __future__ import annotations

import csv
import json

import numpy as np

from . import config as C


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def _metrics(pred: np.ndarray, gold: np.ndarray, gallery: np.ndarray) -> dict:
    p, g = _unit(pred.astype(np.float64)), _unit(gold.astype(np.float64))
    cos = float((p * g).sum(1).mean())
    mse = float(((pred - gold) ** 2).mean())
    # Retrieval: does the prediction's nearest gallery vector belong to the same sentence?
    sims = p @ _unit(gallery.astype(np.float64)).T          # (q, n)
    rank = (sims > (p * g).sum(1, keepdims=True)).sum(1)
    return {"mse": round(mse, 5), "cosine": round(cos, 4),
            "retrieval_at_1": round(float((rank == 0).mean()), 4),
            "n_query": int(len(pred)), "n_gallery": int(len(gallery))}


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    a = xc.T @ xc
    a[np.diag_indices_from(a)] += lam * len(xc) / 1000.0  # scale-free lambda
    w = np.linalg.solve(a, xc.T @ yc)
    return w, ym - xm @ w


def run() -> dict:
    meta = json.loads((C.DATA / "pairs_manifest.json").read_text())
    split = np.array(
        __import__("pyarrow.parquet", fromlist=["read_table"]).read_table(
            C.PAIRS, columns=["split"]
        )["split"].to_pylist()
    )
    x_vi = np.load(C.DATA / "enc_vit5_vi.npy")   # vit5 space
    x_en = np.load(C.DATA / "enc_flan_en.npy")   # flan space
    tr, va, te = (split == "train"), (split == "val"), (split == "test")

    summary = {"pairs_sha256": meta["sha256"], "rows": {k: int(m.sum()) for k, m in
               (("train", tr), ("val", va), ("test", te))}, "directions": {}}
    sweep_rows, metric_rows = [], []

    for name, x, y in (("vi_en", x_vi, x_en), ("en_vi", x_en, x_vi)):
        best = None
        for lam in C.RIDGE_LAMBDAS:
            w, b = fit_ridge(x[tr], y[tr], lam)
            m = _metrics(x[va] @ w + b, y[va], y[tr])
            sweep_rows.append([name, lam, m["mse"], m["cosine"], m["retrieval_at_1"]])
            score = m["cosine"] - 0.5 * m["mse"]
            if best is None or score > best[0]:
                best = (score, lam, w, b, m)

        _, lam, w, b, val_m = best
        test_m = _metrics(x[te] @ w + b, y[te], y[tr])
        ident_m = _metrics(x[te], y[te], y[tr])
        mean_m = _metrics(np.repeat(y[tr].mean(0, keepdims=True), int(te.sum()), 0), y[te], y[tr])
        np.savez(C.DATA / f"map_{name}.npz", w=w.astype(np.float32), b=b.astype(np.float32),
                 lambda_=lam)
        summary["directions"][name] = {"lambda": lam, "val": val_m, "test": test_m,
                                       "control_identity_test": ident_m,
                                       "control_mean_test": mean_m}
        for tag, m in (("val", val_m), ("test", test_m), ("identity", ident_m), ("mean", mean_m)):
            metric_rows.append([name, tag, m["mse"], m["cosine"], m["retrieval_at_1"]])

    with (C.RESULTS / "ridge_sweep.csv").open("w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["direction", "lambda", "mse", "cosine", "retrieval_at_1"])
        w_.writerows(sweep_rows)
    with (C.RESULTS / "mapping_metrics.csv").open("w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["direction", "split", "mse", "cosine", "retrieval_at_1"])
        w_.writerows(metric_rows)
    (C.RESULTS / "mapping_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
