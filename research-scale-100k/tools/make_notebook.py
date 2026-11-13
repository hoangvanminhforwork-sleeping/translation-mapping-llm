"""Build 01_mapping_100k.ipynb. Regenerating it never destroys results: the notebook only reads them.

    python tools/make_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "01_mapping_100k.ipynb"

cells = [
    ("markdown", """# 01 — Linear mapping vit5-base → flan-t5-base on 100k sentence pairs

**Question.** Can a linear map from one language model's sentence representation into another's let
the second model generate a translation?

**Setup.** 97,743 train / 1,000 val / 1,000 test EN–VI pairs (`src/build_pairs.py`, seed 42).
Representation = masked-mean of the T5 encoder's last hidden state (768-d, same width on both sides).
`vit5-base` (Vietnamese) is frozen base; `flan-t5-base` carries the LoRA adapter trained in
`finetune-and-lora/` (copy circuit, 96.6 % exact copies).

**Two directions.** `f: vit5_vi → flan_en` (vi→en) and `g: flan_en → vit5_vi` (en→vi).
Both are closed-form ridge fits on 97,743 rows; λ is chosen on val.

**Why this notebook is short.** Encoding 200k sequences and generating 3,000 translations are GPU
jobs that belong in `src/encode.py` and `src/bleu.py`; the notebook reads their outputs so the
numbers here are the real ones, not a demo. Generation of the 100k training sentences is not
attempted — BLEU is measured on the 1,000-sentence test split and that limit is stated in the report.
"""),
    ("code", """import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src import config as C, mapping

pd.set_option("display.width", 160)
print("root     ", ROOT.name)
print("pairs    ", C.PAIRS.name, C.PAIRS.stat().st_size // 1024, "KB")
print("results  ", sorted(p.name for p in C.RESULTS.glob("*.json")) + sorted(p.name for p in C.RESULTS.glob("*.csv")))"""),
    ("markdown", "## 1. The data"),
    ("code", """meta = json.loads((C.DATA / "pairs_manifest.json").read_text())
pairs = pd.read_parquet(C.PAIRS)
print(json.dumps({k: meta[k] for k in ("rows", "split", "seed", "sha256")}, indent=2))
pairs.groupby("split").size().rename("rows").to_frame()"""),
    ("code", """pairs.sample(5, random_state=0)[["vi", "en"]].assign(
    vi_len=pairs["vi"].str.split().str.len(), en_len=pairs["en"].str.split().str.len())"""),
    ("markdown", "## 2. Encoding throughput (what it actually cost)"),
    ("code", """enc = [json.loads((C.DATA / f"enc_{t}.json").read_text())
       for t in ("vit5_vi", "flan_en", "flan_vi_test")]
pd.DataFrame(enc)[["tag", "model", "adapter", "rows", "dim", "seconds"]].assign(
    seq_per_s=lambda d: (d.rows / d.seconds).round(1))"""),
    ("markdown", """## 3. Fit the map and score it

`mse` and `cosine` are geometry; `retrieval@1` asks whether the mapped vector's nearest neighbour in
the target space is the same sentence. The `identity` control answers the obvious objection: how much
does the map buy over just reusing the source vector as-is?"""),
    ("code", """summary = mapping.run()
rows = []
for d, v in summary["directions"].items():
    for tag in ("val", "test", "control_identity_test", "control_mean_test"):
        m = v[tag]
        rows.append({"direction": d, "split": tag, "lambda": v["lambda"], **m})
metrics = pd.DataFrame(rows)
metrics"""),
    ("code", """sweep = pd.read_csv(C.RESULTS / "ridge_sweep.csv")
ax = sweep.pivot(index="lambda", columns="direction", values="cosine").plot(
    logx=True, marker="o", ylabel="cosine(mapped, gold) on val", title="ridge sweep")
ax.grid(alpha=.3);"""),
    ("markdown", "## 4. Does the mapped vector survive the decoder? (BLEU, vi→en)"),
    ("code", """bleu = json.loads((C.RESULTS / "bleu_vi_en.json").read_text())
print("test sentences:", bleu["n"], "| mean reference length: %.2f tokens" % bleu["mean_ref_tokens"])
pd.DataFrame(bleu["scores"]).T"""),
    ("markdown", """`flan_encoder` is the control: flan's own encoder reads the **Vietnamese** sentence and the same
decoder generates. If `mapped` ≪ `flan_encoder`, the map is what loses the information. If both are
low, a single pooled vector is not enough for this decoder — a different conclusion entirely."""),
    ("code", """gens = {n: [json.loads(l) for l in (C.RESULTS / f"gen_{n}.jsonl").read_text(encoding="utf-8").splitlines()]
        for n in ("mapped", "flan_encoder", "zeroshot_vit5")}
tbl = pd.DataFrame({"vi": [g["vi"] for g in gens["mapped"]],
                    "reference": pairs[pairs.split == "test"]["en"].to_list()[:len(gens["mapped"])],
                    **{n: [g["hyp"] for g in gens[n]] for n in gens}})
tbl.head(10)"""),
    ("markdown", """### Reading the BLEU table

Read it as a comparison, not as a translation-quality claim:

* corpus BLEU on 1,000 legal/general-domain sentences, beam = 1, no length-penalty tuning;
* the decoder sees **one pooled vector per sentence**, with no token alignment, so it cannot copy
  word order from its input. That is the architecture as proposed in `paperOverview.md` §3.3, and it
  is the main suspect if `mapped` scores poorly;
* a low score is a result about *this* representation choice (pooled encoder state, single token),
  not proof that representation mapping cannot work."""),
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


nb = {
    "cells": [md(t) if k == "markdown" else code(t) for k, t in cells],
    "metadata": {
        "kernelspec": {"display_name": "research-scale-100k", "language": "python",
                       "name": "research-scale-100k"},
        "language_info": {"name": "python", "version": "3.12.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {NB.name}: {len(nb['cells'])} cells")
