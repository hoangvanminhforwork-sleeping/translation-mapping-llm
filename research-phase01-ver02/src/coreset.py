"""Coreset selection: which shard is the most representative and the most diverse?

Protocol (mission §3.3)
  1. Encode a stratified 50k-row sample of every shard with the Vietnamese model (mean-pooled, 512-d).
  2. Centre, L2-normalise, and (by default) drop the first principal direction. Raw mean-pooled
     transformer states are strongly anisotropic: one direction carries most of the variance and
     hides domain structure, which is the same anisotropy that makes "cosine 0.7" look impressive
     in the mapping results.
  3. MiniBatchKMeans with K = 4 (the number of MTet domains = number of dataset fields); K = 32 is a
     sensitivity check.
  4. Per shard: representativeness (1 - mean distance to the assigned centroid), diversity
     (normalised cluster entropy), outlier_rate (share beyond the global 95th pct), coverage
     (clusters with >= 0.5 % of the shard), and JS divergence from the pooled cluster histogram.
  5. score = mean of the four z-scores across shards; a 100x within-shard bootstrap gives the
     sampling SD of that score, so a 0.1 difference can be called noise instead of a finding.

The LoRA shard is ineligible: an earlier agent already sampled it for copy fine-tuning.

Outputs: results/coreset.csv, results/coreset.json, data/chosen_shard.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")


def unit(a: np.ndarray) -> np.ndarray:
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)


def preprocess(X: np.ndarray, drop_pc: int = 1) -> np.ndarray:
    """Centre, remove the top principal directions, L2-normalise. X: (n,512) float32."""
    X = X - X.mean(0, keepdims=True)
    if drop_pc > 0:
        # top directions of the covariance, via randomised SVD on a subsample for speed
        sub = X[::max(1, len(X) // 50_000)]
        _u, _s, vt = np.linalg.svd(sub, full_matrices=False)
        P = vt[:drop_pc].T
        X = X - (X @ P) @ P.T
    return unit(X)


def assign(X: np.ndarray, km: MiniBatchKMeans, chunk: int = 20_000):
    lab = np.empty(len(X), dtype=np.int16)
    dist = np.empty(len(X), dtype=np.float32)
    for s in range(0, len(X), chunk):
        d = km.transform(X[s:s + chunk])
        lab[s:s + chunk] = d.argmin(1)
        dist[s:s + chunk] = d[np.arange(len(d)), lab[s:s + chunk]]
    return lab, dist


def metrics_for(mask, lab, dist, k, p95, pooled, n_boot=0, rng=None):
    def one(ix):
        l, d = lab[ix], dist[ix]
        n = len(ix)
        h = np.bincount(l, minlength=k).astype(np.float64) / n
        ent = float(-(h[h > 0] * np.log(h[h > 0])).sum() / np.log(k))
        m = h > 0
        js = float(0.5 * (h[m] * np.log(h[m] / pooled[m])).sum() +
                   0.5 * (pooled[m] * np.log(pooled[m] / h[m])).sum())
        return {"rep": float(1.0 - d.mean()), "diversity": ent,
                "outlier_rate": float((d > p95).mean()),
                "coverage": float((h >= 0.005).sum() / k), "js": js, "mean_dist": float(d.mean())}

    base = one(np.flatnonzero(mask))
    if n_boot:
        idx = np.flatnonzero(mask)
        boots = [one(rng.choice(idx, size=len(idx), replace=True)) for _ in range(n_boot)]
        base["sd"] = {c: float(np.std([b[c] for b in boots])) for c in ("rep", "diversity", "outlier_rate", "coverage")}
    return base


def zscore(v):
    v = np.asarray(v, dtype=np.float64)
    return (v - v.mean()) / (v.std() + 1e-12)


def run(X, files, k, eligible, n_boot=100, seed=20261115, drop_pc=1):
    rng = np.random.default_rng(seed)
    Xp = preprocess(X, drop_pc)
    fit_idx = rng.choice(len(Xp), size=min(60_000, len(Xp)), replace=False)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10, batch_size=4096,
                         max_iter=200, reassignment_ratio=0.01).fit(Xp[fit_idx])
    lab, dist = assign(Xp, km)
    p95 = float(np.percentile(dist, 95))
    pooled = np.bincount(lab, minlength=k).astype(np.float64) / len(lab)
    rows = []
    for f in sorted(set(files.tolist())):
        m = files == f
        r = {"file": f, "n": int(m.sum()), **metrics_for(m, lab, dist, k, p95, pooled, n_boot, rng)}
        rows.append(r)
    comps = {c: zscore([r[c] for r in rows]) for c in ("rep", "diversity", "outlier_rate", "coverage")}
    for i, r in enumerate(rows):
        r["eligible"] = r["file"] in eligible
        r["score"] = float(np.mean([comps[c][i] for c in comps])) if r["eligible"] else float("nan")
        # propagate the bootstrap SD through the same z-scaling
        r["score_sd"] = float(np.sqrt(sum((r["sd"][c] / (np.std([q[c] for q in rows]) + 1e-12)) ** 2
                                          for c in comps)) / 4) if "sd" in r else None
    rows.sort(key=lambda r: (-1e9 if not r["eligible"] else r["score"]), reverse=True)
    return rows, {"k": k, "drop_pc": drop_pc, "n_sample": int(len(X)), "p95_dist": p95,
                  "inertia": float(km.inertia_), "n_boot": n_boot}


def main():
    X = np.load(C.DATA / "sent_sample_fp16.npy").astype(np.float32)
    meta = pq.read_table(C.DATA / "sent_sample_meta.parquet").to_pydict()
    files = np.asarray(meta["file"], dtype=object)
    print(f"sample {X.shape}, {len(set(files.tolist()))} shards")
    all_files = sorted(set(files.tolist()))
    eligible = {f for f in all_files if f != C.LORA_FILE}
    out = {"lora_file": C.LORA_FILE, "eligible": sorted(eligible), "protocol": __doc__, "runs": {}}

    for k in (C.K_MAIN, C.K_SENS):
        rows, info = run(X, files, k, eligible)
        out["runs"][f"k{k}"] = {"info": info, "rows": rows, "chosen": rows[0]["file"]}
        print(f"\nK={k} (drop_pc={info['drop_pc']}): chosen {rows[0]['file']}")
        print(f"{'file':26s} {'rep':>6s} {'div':>6s} {'out':>6s} {'cov':>5s} {'js':>6s} "
              f"{'score':>7s} {'sd':>6s}")
        for r in rows:
            print(f"{r['file']:26s} {r['rep']:.3f} {r['diversity']:.3f} {r['outlier_rate']:.3f} "
                  f"{r['coverage']:.3f} {r['js']:.4f} {r['score']:7.3f} {r['score_sd']:6.3f}")

    out["chosen"] = out["runs"][f"k{C.K_MAIN}"]["chosen"]
    (C.RESULTS / "coreset.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with (C.RESULTS / "coreset.csv").open("w", encoding="utf-8") as f:
        f.write("k,file,n,representativeness,diversity,outlier_rate,coverage,js,mean_dist,"
                "score,score_sd,eligible,chosen\n")
        for k in (C.K_MAIN, C.K_SENS):
            for r in out["runs"][f"k{k}"]["rows"]:
                f.write(f"{k},{r['file']},{r['n']},{r['rep']:.5f},{r['diversity']:.5f},"
                        f"{r['outlier_rate']:.5f},{r['coverage']:.5f},{r['js']:.6f},{r['mean_dist']:.5f},"
                        f"{r['score']:.4f},{r['score_sd']:.4f},{int(r['eligible'])},"
                        f"{int(r['file'] == out['chosen'])}\n")
    k4 = out["runs"][f"k{C.K_MAIN}"]
    spread = float(np.nanmax([r["score"] for r in k4["rows"]]) - np.nanmin([r["score"] for r in k4["rows"]]))
    (C.DATA / "chosen_shard.json").write_text(json.dumps(
        {"file": out["chosen"],
         "rule": "max mean z-score of rep/diversity/outlier/coverage, K=4, top PC removed",
         "score": k4["rows"][0]["score"], "score_sd": k4["rows"][0]["score_sd"],
         "runner_up": {"file": k4["rows"][1]["file"], "score": k4["rows"][1]["score"]},
         "score_spread_across_shards": spread,
         "exchangeable": bool(spread < 3 * (k4["rows"][0]["score_sd"] or 0.0)),
         "k32_chosen": out["runs"][f"k{C.K_SENS}"]["chosen"]}, indent=2), encoding="utf-8")
    print(f"\nwrote {C.RESULTS / 'coreset.csv'}")


if __name__ == "__main__":
    main()
