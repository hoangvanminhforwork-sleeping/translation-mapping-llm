"""§3.3 - does the logistic probe agree with a decision model about "is this about food"?

Neither side is ground truth, so the output is an *agreement* analysis, not an accuracy. The weak
referee is a lexicon built from the RecipeNLG ingredient vocabulary: a sentence "contains a food name"
if it contains one of those words. That referee is itself noisy, so it is reported as a third column
and its disagreement with both is discussed in REPORT.md rather than being called an error.

Probe (this repo): flan-t5-base masked-mean embedding of *sliding windows* (8 words, stride 4), the
sentence score is the max over its windows — the mission specifies window-level probing.
Decision model (API): morph/reflex `noul` decision, one call per sentence.

Usage
    python -m src.decision_check --n 200 [--backend reflex|qwen]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import requests
import torch

from . import config as C
from .probe import embed

sys.stdout.reconfigure(encoding="utf-8")
ENV = C.ROOT / ".env"
WINDOW, STRIDE = 8, 4


def api_key() -> str:
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("DataWorker"):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("DataWorker key not found in research-phase01/.env")


def sample_sentences(n: int, lex: set[str]) -> list[dict]:
    """Half the sample deliberately contains a food lexeme, half does not, otherwise agreement is
    dominated by the easy majority class."""
    con = duckdb.connect()
    rows = con.sql(
        f"SELECT en FROM read_parquet('{C.PAIRS.as_posix()}') "
        f"WHERE length(en) BETWEEN 30 AND 220 USING SAMPLE {n * 20} ROWS (reservoir, 7)"
    ).fetchall()
    with_food, without = [], []
    for (en,) in rows:
        words = set(re.findall(r"[a-z]{4,}", en.lower()))
        hit = bool(words & lex)
        (with_food if hit else without).append({"text": en, "lexicon_food": int(hit)})
    k = min(n // 2, len(with_food), len(without))
    out = with_food[:k] + without[:k]
    print(f"sampled {len(out)} sentences ({k} with a food lexeme, {k} without)")
    return out


def probe_score(texts: list[str], win: int = WINDOW, stride: int = STRIDE) -> np.ndarray:
    z = np.load(C.DATA / "probe_model.npz")
    w, b, mu, sd = z["w"], z["b"], z["mu"], z["sd"]
    windows, owner = [], []
    for i, t in enumerate(texts):
        ws = t.split()
        if len(ws) <= win:
            windows.append(t)
            owner.append(i)
        else:
            for s in range(0, len(ws) - win + 1, stride):
                windows.append(" ".join(ws[s:s + win]))
                owner.append(i)
    X = embed(windows).astype(np.float32)
    s = ((X - mu) / sd) @ w + b[0]
    out = np.full(len(texts), -np.inf, np.float32)
    for o, v in zip(owner, s):
        out[o] = max(out[o], float(v))
    return out


def reflex_one(text: str, key: str) -> float | None:
    body = {
        "model": "morph/reflex",
        "state": [text],
        "questions": {
            "is_food": {
                "type": "noul",
                "instructions": "Does this sentence name or describe a food, a dish, or an ingredient?",
                "criteria": {"true": "A food, dish or ingredient is named",
                             "false": "No food, dish or ingredient is named"},
            }
        },
    }
    r = requests.post("https://openrouter.ai/api/alpha/decisions", json=body,
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                      timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
    js = r.json()
    ans = js.get("answers", {}).get("is_food", {})
    return float(ans.get("noul")) if "noul" in ans else None


def qwen_one(text: str, key: str, model: str = "qwen/qwen3.7-flash") -> float | None:
    """Fallback decision call: constrained yes/no, probability read from the logprob of the answer."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer with exactly one word: yes or no."},
            {"role": "user", "content": "Does this sentence name or describe a food, a dish, or an "
                                       f"ingredient? Sentence: {text}"},
        ],
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 5,
        "temperature": 0,
        "reasoning": {"enabled": False},
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                      timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
    js = r.json()
    try:
        top = js["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return None
    py = pn = None
    for item in top:
        tok = str(item.get("token", "")).strip().lower()
        if tok == "yes":
            py = float(np.exp(item["logprob"]))
        elif tok == "no":
            pn = float(np.exp(item["logprob"]))
    if py is None and pn is None:
        return None
    py = py or 0.0
    pn = pn or 0.0
    return py / max(py + pn, 1e-9)


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / max(1 - pe, 1e-9)


def main(n: int, backend: str, workers: int):
    from .probe import food_lexicon, load_recipes

    lex = food_lexicon(load_recipes())
    rows = sample_sentences(n, lex)
    texts = [r["text"] for r in rows]

    print("probing windows …", flush=True)
    s_probe = probe_score(texts)
    rows = rows[: len(s_probe)]
    texts = texts[: len(s_probe)]

    key = api_key()
    fn = reflex_one if backend == "reflex" else qwen_one
    t0 = time.time()
    probs = [None] * len(texts)
    errs = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, t, key): i for i, t in enumerate(texts)}
        for f, i in futs.items():
            try:
                probs[i] = f.result()
            except Exception as e:  # noqa: BLE001
                errs += 1
                if errs <= 3:
                    print("  API error:", str(e)[:160])
    ok = [i for i, p in enumerate(probs) if p is not None]
    print(f"{backend}: {len(ok)}/{len(texts)} answered in {time.time() - t0:.0f}s, {errs} errors")
    if not ok:
        raise SystemExit("no decision-model answers; nothing to compare")

    p_probe = (s_probe > 0).astype(int)
    p_dm = np.array([1 if probs[i] > 0.5 else 0 for i in ok])
    pr = p_probe[ok]
    ref = np.array([rows[i]["lexicon_food"] for i in ok])
    out = {
        "n": len(ok),
        "backend": backend,
        "probe_threshold": 0.0,
        "decision_threshold": 0.5,
        "agreement_probe_vs_decision": float((pr == p_dm).mean()),
        "kappa_probe_vs_decision": kappa(pr, p_dm),
        "probe_vs_lexicon": {"accuracy": float((pr == ref).mean()), "kappa": kappa(pr, ref)},
        "decision_vs_lexicon": {"accuracy": float((p_dm == ref).mean()), "kappa": kappa(p_dm, ref)},
        "confusion_probe_decision": {
            "both_food": int(((pr == 1) & (p_dm == 1)).sum()),
            "both_not": int(((pr == 0) & (p_dm == 0)).sum()),
            "probe_only": int(((pr == 1) & (p_dm == 0)).sum()),
            "decision_only": int(((pr == 0) & (p_dm == 1)).sum()),
        },
        "mean_decision_prob": float(np.mean([probs[i] for i in ok])),
        "mean_probe_margin": float(np.mean(s_probe[ok])),
        "windows_per_sentence": round(sum(max(1, (len(t.split()) - WINDOW) // STRIDE + 1) for t in texts) / len(texts), 2),
        "api_errors": errs,
    }
    (C.RESULTS / "food_consistency.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with (C.RESULTS / "food_consistency_sample.jsonl").open("w", encoding="utf-8") as f:
        for i in ok[:200]:
            f.write(json.dumps({"text": rows[i]["text"], "lexicon_food": rows[i]["lexicon_food"],
                                "probe_margin": float(s_probe[i]), "decision_prob": probs[i]},
                               ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--backend", choices=["reflex", "qwen"], default="reflex")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    main(a.n, a.backend, a.workers)
