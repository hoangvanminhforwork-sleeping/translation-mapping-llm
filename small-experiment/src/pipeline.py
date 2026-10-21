"""Every stage as a function; the notebooks are thin wrappers around these.

    python -m src.pipeline sample|encode|verify|mapping|probe|all

Each stage is idempotent: sampling reuses its frozen artifact, SQL uses
CREATE TABLE IF NOT EXISTS + upserts, and every artifact is rewritten wholesale.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import config as C
from . import data, db, encode, mapping, manifest, probing

# Windows consoles default to cp1252, which cannot print Vietnamese at all.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def _dump(name: str, obj) -> None:
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), "utf-8")


# ---------------------------------------------------------------- 1. sampling
def stage_sample(force: bool = False) -> pd.DataFrame:
    pairs = data.sample_pairs(force=force)
    titles = data.probe_titles(force=force)
    print(f"pairs: {len(pairs)}  train={(pairs.split == 'train').sum()} "
          f"test={(pairs.split == 'test').sum()}   probe titles: {len(titles)}")
    print(pairs.head(3).to_string())
    return pairs


# ---------------------------------------------------------------- 2. encoding
def stage_encode(force_pairs: bool = False) -> dict:
    pairs = data.sample_pairs(force=force_pairs)
    print("mysql", db.ensure_schema())
    print("pairs upserted:", db.upsert_pairs(pairs))

    used, inserted = {}, 0
    for lang, column, candidates in (("en", "en", C.EN_ENCODERS), ("vi", "vi", C.VI_ENCODERS)):
        name, tok, model = encode.load_encoder(candidates)
        used[lang] = name
        print(f"[{lang}] {name}")
        reps = encode.encode(pairs[column].tolist(), tok, model)
        rows = [(pid, lang, name, pooling, vec)
                for pooling in C.POOLINGS
                for pid, vec in zip(pairs.pair_id.to_numpy(), reps[pooling])]
        inserted += db.upsert_embeddings(rows)
        del reps, rows
        encode.free(model)

    stats = db.counts()
    groups = {f"{lang}/{pooling}": n for (lang, pooling), n in stats["embeddings"].items()}
    _dump("encode_summary.json", {"models": used, "inserted": inserted,
                                  "db": {"pairs": stats["pairs"], "embeddings": groups}})
    manifest.record("encode", models=used, inserted=inserted,
                    poolings=list(C.POOLINGS), max_tokens=C.MAX_TOKENS,
                    db_counts={"pairs": stats["pairs"], "embeddings": groups})
    print(stats)
    return {"models": used, "db": stats}


# ---------------------------------------------------------------- 3. verify
def stage_verify(sample: int = 50) -> dict:
    """Invariants: row counts, dims, storage fidelity, pair alignment."""
    stats = db.counts()
    expected = len(data.sample_pairs())
    assert stats["pairs"] == expected, f"pairs {stats['pairs']} != {expected}"
    groups = {f"{k[0]}/{k[1]}": v for k, v in stats["embeddings"].items()}
    assert expected * len(C.POOLINGS) * 2 == sum(groups.values()), groups

    ids_en, X_en = db.fetch_vectors("en", "mean", "test")
    ids_vi, X_vi = db.fetch_vectors("vi", "mean", "test")
    assert np.array_equal(ids_en, ids_vi), "EN/VI test rows are not aligned"
    assert X_en.shape == X_vi.shape == (C.N_TEST, C.DIM), (X_en.shape, X_vi.shape)

    cos = (mapping._l2(X_en) * mapping._l2(X_vi)).sum(1)

    # storage fidelity: we wrote 6 decimals, so every coordinate must come back
    # on a 1e-6 grid (anything else means the column mangled the value)
    grid = float(np.abs(X_vi[:sample] * 1e6 - np.round(X_vi[:sample] * 1e6)).max() / 1e6)

    # alignment: a stored vector must be its own nearest neighbour
    sims = mapping._l2(X_vi[:sample]) @ mapping._l2(X_vi).T
    top1 = ids_vi[np.argmax(sims, axis=1)]
    self_hits = float(np.mean(top1 == ids_vi[:sample]))

    out = {
        "pairs": stats["pairs"], "embedding_rows": groups,
        "test_shape": list(X_en.shape), "dim": C.DIM,
        "paired_cosine_mean": float(cos.mean()),
        "quantisation_error_max": float(grid),
        "self_retrieval_top1": self_hits,
    }
    _dump("verify.json", out)
    manifest.record("verify", **out)
    print(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------- 4. mapping
def _split_vectors(pooling: str):
    ids_tr, Xtr = db.fetch_vectors("en", pooling, "train")
    ids_tr_vi, Ytr = db.fetch_vectors("vi", pooling, "train")
    ids_te, Xte = db.fetch_vectors("en", pooling, "test")
    ids_te_vi, Yte = db.fetch_vectors("vi", pooling, "test")
    assert np.array_equal(ids_tr, ids_tr_vi) and np.array_equal(ids_te, ids_te_vi)
    return Xtr, Ytr, Xte, Yte


def _fit_val_split(n: int, seed: int = C.SEED):
    idx = np.random.default_rng(seed).permutation(n)
    cut = max(1, int(n * C.VAL_FRACTION))
    return idx[cut:], idx[:cut]


def stage_mapping() -> pd.DataFrame:
    rows, curves = [], []
    for pooling in C.POOLINGS:
        print(f"[{pooling}] loading vectors")
        Xtr, Ytr, Xte, Yte = _split_vectors(pooling)
        fit, val = _fit_val_split(len(Xtr))

        rows.append({"pooling": pooling, "method": "identity", "hp": "-",
                     **mapping.metrics(Xte, Yte)})

        xmu, xsd = Xtr[fit].mean(0, keepdims=True), Xtr[fit].std(0, keepdims=True) + 1e-6
        ymu, ysd = Ytr[fit].mean(0, keepdims=True), Ytr[fit].std(0, keepdims=True) + 1e-6
        z = lambda M, mu, sd: (M - mu) / sd                      # noqa: E731

        best = None
        for alpha in C.RIDGE_ALPHAS:
            params = mapping.ridge_fit(z(Xtr[fit], xmu, xsd), z(Ytr[fit], ymu, ysd), alpha)
            pred_val = mapping.ridge_apply(z(Xtr[val], xmu, xsd), params) * ysd + ymu
            val_mse = float(((pred_val - Ytr[val]) ** 2).mean())
            if best is None or val_mse < best[1]:
                best = (alpha, val_mse, params)
        alpha, val_mse, params = best
        pred_ridge = mapping.ridge_apply(z(Xte, xmu, xsd), params) * ysd + ymu
        rows.append({"pooling": pooling, "method": "ridge", "hp": f"alpha={alpha}",
                     "val_mse_std": val_mse, **mapping.metrics(pred_ridge.astype(np.float32), Yte)})
        print(f"  ridge alpha={alpha} r@1={rows[-1]['r@1']:.4f}")

        model, scalers, curve = mapping.train_mlp(Xtr[fit], Ytr[fit], Xtr[val], Ytr[val])
        pred_mlp = mapping.apply_mlp(model, scalers, Xte)
        rows.append({"pooling": pooling, "method": "mlp",
                     "hp": f"h={C.HIDDEN},epochs={len(curve)}",
                     **mapping.metrics(pred_mlp, Yte)})
        print(f"  mlp  epochs={len(curve)} r@1={rows[-1]['r@1']:.4f}")
        curves += [{"pooling": pooling, **c} for c in curve]

        del Xtr, Ytr, Xte, Yte, pred_ridge, pred_mlp

    df = pd.DataFrame(rows)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RESULTS / "mapping_metrics.csv", index=False)
    pd.DataFrame(curves).to_csv(C.RESULTS / "mlp_training_curve.csv", index=False)
    _plots(df, pd.DataFrame(curves))
    manifest.record("mapping", models=_models_used(), rows=len(df),
                    results=str((C.RESULTS / "mapping_metrics.csv").name),
                    sha256=manifest.sha256(C.RESULTS / "mapping_metrics.csv"))
    print(df[["pooling", "method", "mse", "cosine", "r@1", "r@5", "r@10"]].to_string(index=False))
    return df


def _models_used() -> dict:
    try:
        doc = json.loads((C.AGENT_WORK / "manifest.json").read_text("utf-8"))
        return doc.get("encode", {}).get("models", {})
    except Exception:
        return {}


def _plots(df: pd.DataFrame, curves: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    methods = ["identity", "ridge", "mlp"]
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, method in enumerate(methods):
        sub = df[df.method == method].set_index("pooling").reindex(C.POOLINGS)
        ax.bar(np.arange(len(sub)) + j * width, sub["r@1"], width, label=method)
    ax.set_xticks(np.arange(len(C.POOLINGS)) + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(C.POOLINGS)
    ax.set_ylabel("Recall@1 (closed set, n=2000)")
    ax.set_title("EN->VI mapping: retrieval after mapping")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.RESULTS / "retrieval_at1.png", dpi=150)
    plt.close(fig)

    if len(curves):
        fig, ax = plt.subplots(figsize=(7, 4))
        for pooling, sub in curves.groupby("pooling"):
            ax.plot(sub["epoch"], sub["val_mse_std"], label=pooling)
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation MSE (standardised space)")
        ax.set_title("MLP mapping training")
        ax.legend()
        fig.tight_layout()
        fig.savefig(C.RESULTS / "mlp_training_curve.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------- 5. probing
def stage_probe() -> dict:
    pairs = data.sample_pairs()
    positives = data.probe_titles()["title"].tolist()
    negatives = data.probe_negatives(pairs)

    name, tok, model = encode.load_encoder(C.EN_ENCODERS)
    X_pos = encode.encode(positives, tok, model)["mean"]
    X_neg = encode.encode(negatives, tok, model)["mean"]
    encode.free(model)

    result = {"encoder": name, "pooling": "mean", **probing.run(X_pos, X_neg)}
    _dump("probe_food.json", result)
    pd.DataFrame([result]).to_csv(C.RESULTS / "probe_food.csv", index=False)
    manifest.record("probe", encoder=name, positives=str(data.TITLES_ARTIFACT.name),
                    **{k: result[k] for k in ("n_pos", "n_neg", "accuracy", "f1",
                                              "balanced_accuracy", "roc_auc")})

    keys = ["accuracy", "f1", "balanced_accuracy", "roc_auc"]
    print(json.dumps({k: round(result[k], 4) for k in keys} |
                     {"majority_baseline": round(result["majority_baseline"], 4)}, indent=2))
    return result


# ---------------------------------------------------------------- CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["sample", "encode", "verify", "mapping", "probe", "all"])
    ap.add_argument("--force", action="store_true", help="re-draw the sample artifacts")
    args = ap.parse_args(argv)

    if args.stage in ("sample", "all"):
        stage_sample(force=args.force)
    if args.stage in ("encode", "all"):
        stage_encode(force_pairs=args.force)
    if args.stage in ("verify", "all"):
        stage_verify()
    if args.stage in ("mapping", "all"):
        stage_mapping()
    if args.stage in ("probe", "all"):
        stage_probe()


if __name__ == "__main__":
    main()

