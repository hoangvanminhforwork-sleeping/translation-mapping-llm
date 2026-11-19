"""The novel arm: a tiny ranker that predicts translation quality from reference-free features,
plus the diagnostics that decide whether LATENT agreement earns its place.

Two models are trained, identical except for the feature set:
    string   chrF(back-translation, source), char-3gram Jaccard, len_ratio, lp
    latent   the above + pooled cosine + token-level nearest-neighbour precision/recall/F1/std

Reported for every utility (string, latent, learned variants, raw baselines):
    spearman   candidate-level Spearman against COMET (the teacher)
    top1_acc   share of rows where argmax(utility) is the true best candidate (within 0.01 COMET)
    capture    (BLEU(arm) - BLEU(beam@top1)) / (BLEU(oracle) - BLEU(beam@top1))  -> "share of the
               available headroom the utility actually converts"
If latent agreement is a usable signal, `latent` must beat `string` on all three, ideally by more
than the seed noise of the tiny MLP (reported as mean +- sd over 3 seeds).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sacrebleu.metrics import BLEU

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")
BLEU_M = BLEU(tokenize="13a")

STRING_F = ["chrF_bt", "char3_bt", "len_ratio", "hyp_len"]
LATENT_F = ["lat_cos_mean", "lat_match_pre", "lat_match_rec", "lat_match_f1", "lat_match_std",
            "lat_len_ratio",
            "lat_cos_mean_c", "lat_match_pre_c", "lat_match_rec_c", "lat_match_f1_c",
            "lat_match_std_c"]


def load_feats(path) -> tuple[np.ndarray, dict]:
    t = pq.read_table(C.RESULTS / path)
    rows = t.to_pylist()
    by_id: dict[int, list[dict]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
    for v in by_id.values():
        v.sort(key=lambda x: x["rank"])
    return rows, by_id


class Ranker(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(),
                                 nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit(feats: np.ndarray, y: np.ndarray, seed: int, epochs: int = 60) -> Ranker:
    torch.manual_seed(seed)
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    x = torch.tensor((feats - mu) / sd, dtype=torch.float32, device="cuda")
    t = torch.tensor(y, dtype=torch.float32, device="cuda")
    m = Ranker(x.shape[1]).to("cuda")
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(x), device="cuda")
        for s in range(0, len(x), 4096):
            idx = perm[s:s + 4096]
            loss = nn.functional.mse_loss(m(x[idx]), t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    m.mu, m.sd = mu, sd
    return m


def predict(m: Ranker, feats: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.tensor((feats - m.mu) / m.sd, dtype=torch.float32, device="cuda")
        return m(x).cpu().numpy()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-test", default="feat_test_beam8.parquet")
    ap.add_argument("--comet-test", default="comet_test_beam8.parquet")
    ap.add_argument("--feat-train", default="feat_train_beam8.parquet")
    ap.add_argument("--comet-train", default="comet_train_beam8.parquet")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="rank_diag.json")
    a = ap.parse_args(argv)

    # ---- load
    _, test_by_id = load_feats(a.feat_test)
    test_comet = pq.read_table(C.RESULTS / a.comet_test).to_pylist()
    tc: dict[tuple[int, int], float] = {(r["id"], r["rank"]): r["comet"] for r in test_comet}

    _, train_by_id = load_feats(a.feat_train)
    train_comet = pq.read_table(C.RESULTS / a.comet_train).to_pylist()
    trc: dict[tuple[int, int], float] = {(r["id"], r["rank"]): r["comet"] for r in train_comet}

    def matrix(by_id, labels, ids):
        F, y, order = [], [], []
        for i in ids:
            for c in by_id[i]:
                F.append([c[k] for k in STRING_F + LATENT_F])
                y.append(labels[(i, c["rank"])])
                order.append((i, c["rank"]))
        return np.asarray(F, np.float32), np.asarray(y, np.float32), order

    train_ids = sorted({i for i, _ in trc})
    test_ids = sorted({i for i, _ in tc})
    Ft, yt, _ = matrix(train_by_id, trc, train_ids)
    Fs, ys, test_order = matrix(test_by_id, tc, test_ids)
    print(f"train candidates {len(Ft)}, test candidates {len(Fs)}")

    FEATS = STRING_F + LATENT_F
    s_idx = [FEATS.index(k) for k in STRING_F]
    l_idx = [FEATS.index(k) for k in FEATS]

    diag = {"n_test_rows": len(test_ids), "n_test_cands": int(len(Fs)),
            "n_train_rows": len(train_ids), "n_train_cands": int(len(Ft))}

    # ---- univariate utilities: Spearman + top-1 accuracy on the test pool
    def top1_acc(by_id, labels, ids, key):
        hit = 0
        for i in ids:
            cs = by_id[i]
            best = max(labels[(i, c["rank"])] for c in cs)
            pick = max(cs, key=lambda c: c[key])
            hit += labels[(i, pick["rank"])] >= best - 0.01
        return hit / len(ids)

    diag["utilities"] = {}
    for key in ["chrF_bt", "char3_bt", "lat_cos_mean", "lat_match_f1", "lat_match_pre",
                "lat_match_rec", "lat_cos_mean_c", "lat_match_f1_c", "lat_match_pre_c",
                "lat_match_rec_c"]:
        vals = np.array([c[key] for i in test_ids for c in test_by_id[i]], np.float32)
        rho = spearmanr(vals, ys).statistic
        acc = top1_acc(test_by_id, tc, test_ids, key)
        diag["utilities"][key] = {"spearman": float(rho), "top1_acc": float(acc)}
        print(f"  {key:16s} spearman {rho:+.3f}  top1 {acc:.3f}")

    # ---- paired bootstrap of the H1 gap: Spearman(chrF_bt) - Spearman(lat_match_f1)
    # resample SENTENCES (not candidates) so within-row correlation is preserved
    rng = np.random.default_rng(C.SEED)
    lens = np.array([len(test_by_id[i]) for i in test_ids])
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    vec = {k: np.array([c[k] for i in test_ids for c in test_by_id[i]], np.float32)
           for k in ("chrF_bt", "lat_match_f1", "lat_match_f1_c")}
    diffs, diffs_c = [], []
    for _ in range(300):
        pick = rng.integers(0, len(test_ids), len(test_ids))
        idx = np.concatenate([np.arange(starts[i], starts[i] + lens[i]) for i in pick])
        ys_b = ys[idx]
        s_str = spearmanr(vec["chrF_bt"][idx], ys_b).statistic
        s_lat = spearmanr(vec["lat_match_f1"][idx], ys_b).statistic
        s_latc = spearmanr(vec["lat_match_f1_c"][idx], ys_b).statistic
        diffs.append(s_lat - s_str)
        diffs_c.append(s_latc - s_str)
    diag["h1_bootstrap"] = {
        "delta_spearman_latent_minus_string": {
            "mean": float(np.mean(diffs)),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]},
        "delta_spearman_centred_minus_string": {
            "mean": float(np.mean(diffs_c)),
            "ci95": [float(np.percentile(diffs_c, 2.5)), float(np.percentile(diffs_c, 97.5))]},
        "n_boot": 300, "resampling_unit": "sentence",
    }
    print(f"  H1 delta (latent-string): {np.mean(diffs):+.3f} "
          f"[{np.percentile(diffs, 2.5):+.3f}, {np.percentile(diffs, 97.5):+.3f}]")

    # ---- length control: is the signal just "length matching"? report Spearman inside length bands
    diag["length_bands"] = {}
    src_len = np.array([test_by_id[i][0]["src_len"] for i in test_ids])
    qs = np.quantile(src_len, [0.25, 0.5, 0.75])
    band_of = np.digitize(src_len, qs)
    for key in ("chrF_bt", "lat_match_f1", "lat_match_f1_c"):
        vals = np.array([c[key] for i in test_ids for c in test_by_id[i]], np.float32)
        per_band = []
        for b in range(4):
            sel = np.repeat(band_of == b, [len(test_by_id[i]) for i in test_ids])
            per_band.append(float(spearmanr(vals[sel], ys[sel]).statistic) if sel.sum() > 30 else None)
        diag["length_bands"][key] = {"overall": float(spearmanr(vals, ys).statistic),
                                    "by_band": per_band}
        print(f"  {key:16s} bands {per_band}")

    # ---- learned rankers (string-only vs string+latent), 3 seeds
    for name, idx in (("string", s_idx), ("latent", l_idx)):
        preds, rhos, accs = [], [], []
        for seed in [int(s) for s in a.seeds.split(",")]:
            m = fit(Ft[:, idx], yt, seed)
            p = predict(m, Fs[:, idx])
            preds.append(p)
            rhos.append(spearmanr(p, ys).statistic)
            # top-1 accuracy computed on the mean prediction below
        p = np.mean(preds, axis=0)
        pred_by = {k: float(v) for k, v in zip(test_order, p)}
        hit = 0
        for i in test_ids:
            cs = test_by_id[i]
            best = max(tc[(i, c["rank"])] for c in cs)
            pick = max(cs, key=lambda c: pred_by[(i, c["rank"])])
            hit += tc[(i, pick["rank"])] >= best - 0.01
        diag[f"learned_{name}"] = {
            "spearman_mean": float(np.mean(rhos)), "spearman_sd": float(np.std(rhos)),
            "spearman_pooled": float(spearmanr(p, ys).statistic),
            "top1_acc": hit / len(test_ids),
            "n_params": int(sum(x.numel() for x in Ranker(len(idx)).parameters())),
        }
        # write the selection for scoring
        path = C.RESULTS / f"sel_A8_learned_{name}.jsonl"
        with open(C.RESULTS / a.feat_test.replace("feat_", "pool_").replace(".parquet", ".jsonl"),
                  encoding="utf-8") as f:
            pool = [json.loads(line) for line in f]
        with path.open("w", encoding="utf-8") as f:
            for r in pool:
                cs = test_by_id[r["id"]]
                pick = max(cs, key=lambda c: pred_by[(r["id"], c["rank"])])
                c = r["cands"][pick["rank"]]
                f.write(json.dumps({"id": r["id"], "vi": r["vi"], "ref": r["ref"], "core": r["core"],
                                    "hyp": c["hyp"], "rank": pick["rank"], "src": c["src"]},
                                   ensure_ascii=False) + "\n")
        print(f"  learned_{name:8s} spearman {np.mean(rhos):+.3f}+-{np.std(rhos):.3f}  "
              f"top1 {diag[f'learned_{name}']['top1_acc']:.3f}  params "
              f"{diag[f'learned_{name}']['n_params']}")

    # ---- oracle arm
    path = C.RESULTS / "sel_A9_oracle.jsonl"
    with open(C.RESULTS / a.feat_test.replace("feat_", "pool_").replace(".parquet", ".jsonl"),
              encoding="utf-8") as f:
        pool = [json.loads(line) for line in f]
    with path.open("w", encoding="utf-8") as f:
        for r in pool:
            cs = test_by_id[r["id"]]
            pick = max(cs, key=lambda c: tc[(r["id"], c["rank"])])
            c = r["cands"][pick["rank"]]
            f.write(json.dumps({"id": r["id"], "vi": r["vi"], "ref": r["ref"], "core": r["core"],
                                "hyp": c["hyp"], "rank": pick["rank"], "src": c["src"]},
                               ensure_ascii=False) + "\n")
    print("  wrote sel_A9_oracle.jsonl")

    (C.RESULTS / a.out).write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
