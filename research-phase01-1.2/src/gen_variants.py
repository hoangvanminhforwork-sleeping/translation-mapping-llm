"""Mission 4 (final step) - does any repaired map actually translate better?

Takes the two diagnostics winners plus the untransformed baselines through the
frozen decoder with ver02's translation LoRA, and scores them with ver02's own
metrics so the numbers are directly comparable to its published table.

Channels
    ridge            ver02's map, unchanged              (cond "mapped")
    ridge+recolor.1  ridge blended 10% toward full covariance matching
    procrustes       orthogonal map
    procrustes+.3    procrustes blended 30% toward full covariance matching
    native           the target model's own encoder      (cond "native")
    identity         raw source states, no map           (cond "identity")

Transforms are fitted on train tokens with the same subsample and seed as
`src.fix_maps.py`, so what is generated here is exactly what was measured there.

Usage:
    ..\\research-scale-100k\\.venv\\Scripts\\python.exe -m src.gen_variants --n 1000
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
import unicodedata
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
CACHE_TRN = PROJ / "data" / "train_vectors.npz"
MAX_TOK = 30000
SEED = 7
DIRECTION = "vi_en"


# ---------------------------------------------------------------- transforms
def fit_transforms():
    from src.fix_maps import fit_procrustes, fit_recolor

    d = np.load(CACHE_TRN)
    X, Y = d["X"], d["Y"]
    rng = np.random.default_rng(SEED + 1)
    if len(X) > MAX_TOK:
        i = np.sort(rng.choice(len(X), size=MAX_TOK, replace=False))
        X, Y = X[i], Y[i]
    X, Y = X.astype(np.float64), Y.astype(np.float64)

    mp = np.load(VER02 / "data" / "map_vi_en.npz")
    W, b = mp["W"].astype(np.float64), mp["b"].astype(np.float64)

    R, br = fit_procrustes(X, Y)
    R, br = R.astype(np.float64), br.astype(np.float64)
    A_r, muY_r, muM_r = fit_recolor(X @ W + b, Y)
    A_p, muY_p, muM_p = fit_recolor(X @ R + br, Y)
    for arr in (A_r, A_p, muY_r, muM_r, muY_p, muM_p):
        arr.setflags(write=False)
    return dict(W=W, b=b, R=R, br=br, A_r=A_r.astype(np.float64), muY_r=muY_r,
                muM_r=muM_r, A_p=A_p.astype(np.float64), muY_p=muY_p, muM_p=muM_p)


def make_mappers(T):
    W, b, R, br = T["W"], T["b"], T["R"], T["br"]

    def ridge_color(t):
        def f(x):
            m = x.astype(np.float64) @ W + b
            rec = (m - T["muM_r"]) @ T["A_r"] + T["muY_r"]
            return ((1 - t) * m + t * rec).astype(np.float32)
        return f

    def proc():
        def f(x):
            return (x.astype(np.float64) @ R + br).astype(np.float32)
        return f

    def proc_color(t):
        def f(x):
            m = x.astype(np.float64) @ R + br
            rec = (m - T["muM_p"]) @ T["A_p"] + T["muY_p"]
            return ((1 - t) * m + t * rec).astype(np.float32)
        return f

    return {
        "ridge+recolor.1": ridge_color(0.1),
        "procrustes": proc(),
        "procrustes+.3": proc_color(0.3),
    }


# ------------------------------------------------------------------- scoring
VI = re.compile(r"[ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ]", re.I)


def diagnose(hyp):
    """ver02's definition, so degen rates are comparable to its 31.6 %."""
    deg = empty = 0
    for h in hyp:
        toks = h.split()
        if not toks:
            empty += 1
            continue
        if len(set(toks)) / len(toks) <= 0.5:
            deg += 1
    return {"empty_rate": empty / len(hyp), "degenerate_rate": deg / len(hyp),
            "mean_hyp_words": float(np.mean([len(h.split()) for h in hyp if h.split()] or [0]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=48)
    a = ap.parse_args()

    import torch
    import sacrebleu
    from peft import PeftModel
    from transformers import T5ForConditionalGeneration

    from v2 import config as C
    from v2.generate import nll, run, test_pairs
    from v2.train_decoder import Channels, DT, direction_sides

    src, tgt = direction_sides(DIRECTION)
    adapter = C.ADAPTERS[f"trans_{tgt}"]
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"missing adapter {adapter}")
    base = T5ForConditionalGeneration.from_pretrained(
        str(C.MODELS[tgt]["path"]), dtype=DT, low_cpu_mem_usage=True).to("cuda")
    model = PeftModel.from_pretrained(base, str(adapter)).eval()
    ch = Channels(model, DIRECTION)

    T = fit_transforms()
    mappers = make_mappers(T)

    pairs = test_pairs(DIRECTION, a.n)
    refs = [p[1] for p in pairs]
    print(f"{DIRECTION}: {len(pairs)} test pairs, batch {a.batch}, max_new {a.max_new}", flush=True)

    jobs = [
        ("native", "native", None),
        ("identity", "identity", None),
        ("ridge", "mapped", None),
        ("ridge+recolor.1", "lgbm", mappers["ridge+recolor.1"]),
        ("procrustes", "lgbm", mappers["procrustes"]),
        ("procrustes+.3", "lgbm", mappers["procrustes+.3"]),
    ]

    rows = []
    for name, cond, mapper in jobs:
        t0 = time.time()
        hyps, nlls = run(model, ch, pairs, cond, a.batch, a.max_new, lgbm=mapper)
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
        d = diagnose(hyps)
        row = {"channel": name, "bleu": round(bleu, 3), "chrf": round(chrf, 3),
               "nll": round(float(np.mean(nlls)), 3),
               **{k: round(v, 4) for k, v in d.items()},
               "sec": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"  {name:16} BLEU {row['bleu']:6.2f}  chrF {row['chrf']:6.2f}  "
              f"NLL {row['nll']:5.3f}  degen {row['degenerate_rate']:.3f}  "
              f"empty {row['empty_rate']:.3f}  ({row['sec']}s)", flush=True)
        with (OUT / f"gen_{DIRECTION}_{name.replace('+','p').replace('.','')}.jsonl").open(
                "w", encoding="utf-8") as f:
            for (s, r), h, nl in zip(pairs, hyps, nlls):
                f.write(json.dumps({"src": s, "ref": r, "hyp": h, "nll": float(nl)},
                                   ensure_ascii=False) + "\n")

    (OUT / f"gen_variants_{DIRECTION}.json").write_text(json.dumps(
        {"n": len(pairs), "direction": DIRECTION, "max_new": a.max_new,
         "protocol": "identical pairs, one adapter, greedy; metrics = ver02 score.py",
         "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'channel':16}{'BLEU':>8}{'chrF':>8}{'NLL':>8}{'degen':>8}")
    for r in rows:
        print(f"{r['channel']:16}{r['bleu']:8.2f}{r['chrf']:8.2f}{r['nll']:8.3f}"
              f"{r['degenerate_rate']:8.3f}")
    print("\nwrote", OUT / f"gen_variants_{DIRECTION}.json")


if __name__ == "__main__":
    main()
