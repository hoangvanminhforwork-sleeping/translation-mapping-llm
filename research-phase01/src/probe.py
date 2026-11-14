"""§3.1 - logistic-regression probe on flan-t5-base embeddings (before LoRA).

What is probed: whether the frozen encoder already separates "food name / food phrase" from ordinary
prose. Positive = RecipeNLG title+ingredients, negative = MTET English sentence containing no term
from a RecipeNLG-derived food lexicon. The labels are weak and automatic; the domain confound
(recipes are formulaic) is reported in REPORT.md, and §3.3 only uses the probe cross-domain.

Trained with torch (AdamW, BCE) instead of sklearn: 200k x 768 float64 copies do not fit in this
laptop's 7.7 GB, and the GPU is idle during this stage.

Outputs
    data/probe_feats.npy      (N,768) float32 pooled vectors  (gitignored)
    data/probe_labels.npz     y, split, and the weight/bias of the fitted probe
    results/probe_metrics.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")

RECIPE = C.PROJECT / "data" / "RecipeNLG.csv"
N_POS = 70_000
N_VAL = 15_000
N_TEST = 15_000
C_GRID = [0.01, 0.1, 1.0]


def load_recipes() -> list[dict]:
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    rows = con.sql(
        f"""
        SELECT title, ingredients
        FROM read_csv_auto('{RECIPE.as_posix()}', sample_size=20000, header=true)
        WHERE title IS NOT NULL AND ingredients IS NOT NULL
        ORDER BY md5(title)
        LIMIT {(N_POS + N_VAL + N_TEST) * 2}
        """
    ).fetchall()
    out, seen = [], set()
    for title, ing in rows:
        if title in seen:
            continue
        seen.add(title)
        if " . " not in title:
            s = re.sub(r"[\[\]']", "", str(ing))
            text = f"{title} . {s}"
        else:
            text = str(ing)
        k = len(out)
        split = "train" if k < N_POS else "val" if k < N_POS + N_VAL else "test"
        out.append({"text": text, "y": 1, "split": split})
        if len(out) == N_POS + N_VAL + N_TEST:
            break
    return out


def food_lexicon(recipes: list[dict]) -> set[str]:
    """Single-word food terms from the ingredient strings (set membership beats a 3000-way regex)."""
    cnt: Counter[str] = Counter()
    for r in recipes[:N_POS]:
        for w in set(re.findall(r"[a-z]{4,}", r["text"].lower())):
            cnt[w] += 1
    stop = set("""
    with without until into over then than that this these those from your each other some very
    minutes hour hours while about after before large small medium high low heat cook cooked
    place well make made more most just also have will when what which their there they them
    add added stir water salt pepper serve serving servings combine mixture ingredients chopped
    diced sliced fresh dried ground whole half cup cups tablespoon tablespoons teaspoon
    """.split())
    return {w for w, _ in cnt.most_common(6000) if w not in stop}


def load_negatives(lex: set[str], want: int) -> list[str]:
    """MTET English sentences with no food term, read from the already-deduplicated pairs file.

    The first version scanned the raw 8.3M-row corpus and ran two regexp_replace per row for language
    detection; that alone took >10 min of CPU. pairs.parquet already has the Vietnamese side removed
    and duplicates collapsed, so the same sample costs seconds.
    """
    con = duckdb.connect()
    con.execute("PRAGMA threads=8")
    rows = con.sql(
        f"SELECT en FROM read_parquet('{C.PAIRS.as_posix()}') "
        f"WHERE length(en) BETWEEN 20 AND 300 USING SAMPLE {want * 4} ROWS (reservoir, 42)"
    ).fetchall()
    out = []
    for (en,) in rows:
        words = set(re.findall(r"[a-z]{4,}", en.lower()))
        if words & lex:
            continue
        out.append(en)
        if len(out) >= want:
            break
    return out


@torch.no_grad()
def embed(texts: list[str], batch: int = 256, max_len: int = 64) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(C.MODELS["flan-t5-base"]["hf_id"], use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        C.MODELS["flan-t5-base"]["hf_id"], dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda").eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), 768), np.float16)  # float16 halves peak RAM; cast per batch at fit time
    t0 = time.time()
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to("cuda")
        h = model.encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        out[idx] = ((h.float() * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy().astype(np.float16)
        if (s // batch) % 50 == 0:
            print(f"  embed {s + len(idx)}/{len(order)}  {(s + len(idx)) / (time.time() - t0):.0f} seq/s", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def fit_probe(X, y, split, lam: float, epochs: int = 12, batch: int = 4096):
    tr = np.flatnonzero(split == "train")
    Xtr = X[tr].astype(np.float32)
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-6
    del Xtr
    w = torch.zeros(768, device="cuda", requires_grad=True)
    b = torch.zeros(1, device="cuda", requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=2e-3, weight_decay=lam)
    yt = torch.from_numpy(y.astype(np.float32))
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        perm = np.random.default_rng(ep).permutation(tr)
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            xb = torch.from_numpy((X[idx].astype(np.float32) - mu) / sd).cuda()
            loss = lossf(xb @ w + b, yt[idx].cuda())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return w.detach().cpu().numpy(), b.detach().cpu().numpy(), mu, sd


def scores(X, idx, w, b, mu, sd, batch: int = 16384):
    out = np.empty(len(idx), np.float32)
    for s in range(0, len(idx), batch):
        j = idx[s:s + batch]
        out[s:s + len(j)] = ((X[j].astype(np.float32) - mu) / sd) @ w + b[0]
    return out


def main(force: bool = False):
    feats = C.DATA / "probe_feats.npy"
    lab = C.DATA / "probe_labels.npz"
    if feats.exists() and lab.exists() and not force:
        X = np.load(feats)
        z = np.load(lab)
        y, split = z["y"], z["split"]
        print(f"reusing {feats.name}: X{X.shape}")
    else:
        pos = load_recipes()
        lex = food_lexicon(pos)
        neg = load_negatives(lex, len(pos))
        print(f"positives {len(pos):,}  negatives {len(neg):,}")
        # negatives get the same 70/15/15 proportions, by position in the deterministic sample
        n = len(neg)
        bounds = [int(0.70 * n), int(0.85 * n)]
        neg_rows = [{"text": t, "y": 0,
                     "split": "train" if i < bounds[0] else "val" if i < bounds[1] else "test"}
                    for i, t in enumerate(neg)]
        rows = pos + neg_rows
        texts = [r["text"] for r in rows]
        y = np.array([r["y"] for r in rows], np.int8)
        split = np.array([r["split"] for r in rows])
        X = embed(texts)
        np.save(feats, X)
        np.savez_compressed(lab, y=y, split=split)

    va = np.flatnonzero(split == "val")
    te = np.flatnonzero(split == "test")
    tr = np.flatnonzero(split == "train")
    print(f"train {len(tr):,}  val {len(va):,}  test {len(te):,}")

    best = None
    for lam in C_GRID:
        w, b, mu, sd = fit_probe(X, y, split, lam)
        auc = roc_auc_score(y[va], scores(X, va, w, b, mu, sd))
        print(f"  weight_decay={lam}: val AUC {auc:.4f}", flush=True)
        if best is None or auc > best[0]:
            best = (auc, lam, w, b, mu, sd)
    auc_va, lam, w, b, mu, sd = best
    s_te = scores(X, te, w, b, mu, sd)
    metrics = {
        "n": {"train": len(tr), "val": len(va), "test": len(te)},
        "classes": {"1": "RecipeNLG title+ingredients", "0": "MTET sentence, no food lexeme"},
        "weight_decay": lam,
        "val_auc": float(auc_va),
        "test_auc": float(roc_auc_score(y[te], s_te)),
        "test_accuracy": float(accuracy_score(y[te], (s_te > 0).astype(int))),
        "test_f1": float(f1_score(y[te], (s_te > 0).astype(int))),
        "embedding": "flan-t5-base encoder, masked-mean, max_len 64, no LoRA adapter",
    }
    np.savez_compressed(C.DATA / "probe_model.npz", w=w, b=b, mu=mu, sd=sd,
                        threshold=np.array(0.0), lam=np.array(lam))
    (C.RESULTS / "probe_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.force)
