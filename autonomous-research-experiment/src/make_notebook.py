"""Builds NOTEBOOK.ipynb from the files in results/ (a readable, re-runnable report).

Run:  python -m src.make_notebook
Then: jupyter nbconvert --to notebook --execute NOTEBOOK.ipynb   (optional)
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip().splitlines(True)}

cells = [
    md("""
# One night, two frozen 77 M models, one question

**Is *latent* agreement a usable reference-free quality signal for MT — and does it beat the *string*
version of the same idea?**

Pre-registration: `HYPOTHESES.md`. Run log (including mistakes): `PROGRESS.md`.
This notebook only *reads* what the scripts produced; nothing here retrains or regenerates.
"""),
    code("""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, str(Path.cwd()))
from src import config as C

pages = pq.read_table(C.SPLIT, columns=["split"]).to_pandas()
print(pages["split"].value_counts())
"""),
    md("""
## 1. The baseline ladder (BLEU / chrF on 10 k test, COMET on the 2 k historical core)

Every row is a *selection from the same candidate pool* (except A0, which is greedy). Numbers come
from `results/metrics_*.json` — this cell does not compute anything, it only formats.
"""),
    code("""
from src.tables import load_all, pick
m = load_all()
rows = [{"arm": a, "BLEU": pick(e, "bleu"), "chrF": pick(e, "chrf"),
         "COMET": pick(e, "comet", ("lora_core", "core", "all", "10k", "lora"))}
        for a, e in m.items()]
df = pd.DataFrame(rows).dropna(subset=["BLEU"]).sort_values("BLEU", ascending=False).round(3)
df
"""),
    md("""
## 2. Is the signal *usable*? Spearman vs COMET, top-1 accuracy, and headroom captured

`top-1 accuracy` = share of sentences where the utility picks the candidate that COMET likes best.
A utility can correlate well and still be useless if it always agrees on the easy sentences.
"""),
    code("""
diag = json.load(open(C.RESULTS / "rank_diag.json", encoding="utf-8"))
ut = pd.DataFrame(diag["utilities"]).T
for k in ("learned_string", "learned_latent"):
    if k in diag:
        ut.loc[k] = [diag[k]["spearman_mean"], diag[k]["top1_acc"]]
ut
"""),
    md("""
## 3. Does a bigger pool help? (beam-8 vs beam-8 + 8 samples, 2 k core)

`results/mixed_pool.json` and `results/metrics_mixed.json`. If the ceiling rises with candidate count,
selection was pool-limited; if it does not, selection is selector-limited.
"""),
    code("""
mixed = json.load(open(C.RESULTS / "mixed_pool.json", encoding="utf-8"))
mx = json.load(open(C.RESULTS / "metrics_mixed.json", encoding="utf-8"))
pool_tbl = pd.DataFrame([{"pool": v["pool"], "cands/row": v["cands_per_row"],
                          "unique/row": v["unique_per_row"], "top-1 BLEU": v["bleu_top1"],
                          "oracle BLEU": v["bleu_oracle"]} for v in mixed.values()])
arm_tbl = pd.DataFrame([{"arm": k, "BLEU": round(v["bleu"], 2), "chrF": round(v["chrf"], 2)}
                        for k, v in mx.items() if isinstance(v, dict) and "bleu" in v])
print(pool_tbl.to_string(index=False))
print()
print(arm_tbl.to_string(index=False))
"""),
    md("""
## 4. Replication on a second pool (the adapted model's, 2 k / 16 k candidates)

If the latent−string gap were a quirk of one pool, it would vanish here. `results/replication.json`.
"""),
    code("""
print(json.dumps(json.load(open(C.RESULTS / "replication.json", encoding="utf-8")), indent=2))
"""),
    md("""
## 5. The picture: latent agreement vs true quality, per candidate

One point per candidate hypothesis (sampled). If latent agreement were a clean quality signal, this
cloud would be a tight increasing band. Its shape is the answer to the research question.
"""),
    code("""
import matplotlib.pyplot as plt
feat = pq.read_table(C.RESULTS / "feat_test_beam8.parquet").to_pandas()
com  = pq.read_table(C.RESULTS / "comet_test_beam8.parquet").to_pandas()
df = feat.merge(com, on=["id", "rank"])
s = df.sample(min(20000, len(df)), random_state=0)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, col in zip(ax, ["chrF_bt", "lat_match_f1", "lat_match_f1_c"]):
    a.scatter(s[col], s["comet"], s=2, alpha=0.15)
    a.set_xlabel(col); a.set_ylabel("COMET (teacher)"); a.set_title(col)
plt.tight_layout()
"""),
    md("""
## 4. What a reader should check before trusting any of this

- `results/rank_diag.json` → `n_test_rows`, `n_test_cands`: the pool size bounds everything.
- `HYPOTHESES.md` → the deviation log: what was dropped and why (`lp` was dropped, see PROGRESS).
- `PROGRESS.md` → the two self-inflicted stalls; they cost minutes, not results, but they are there.
- The LoRA control (`results/lora_control.json`, `sel_A5_lora*`) is **not** part of the claim; it
  exists so the reader can see how much of the gap is ordinary domain adaptation.
"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"},
                                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
(ROOT / "NOTEBOOK.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote NOTEBOOK.ipynb")
