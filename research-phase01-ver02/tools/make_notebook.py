"""Build 01_results.ipynb — the readable artefact of ver02.

The notebook does not run the pipeline (that needs ~1 h of GPU). It reads the artefacts in
`results/` and `data/`, so every number in it is a measured one, and it fails loudly if a file is
missing instead of inventing a table.
"""
from __future__ import annotations

import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "01_results.ipynb"


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


cells = [
    md("""
# ver02 — 512-d cross-model mapping with 60 M models

Pipeline: `token(vi) → t5-small-vietnamese encoder → 512×512 map → t5-small decoder → token(en)`.

This notebook is the report of record for the numbers. It reads what the pipeline wrote; it does not
recompute it. The command that produced each artefact is printed above its cell.
"""),
    md("## 0. Environment as measured"),
    code("""
import json, sys, platform
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path.cwd()
RES = ROOT / "results"
print(platform.platform(), "| python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("vram MB", torch.cuda.get_device_properties(0).total_memory // 2**20 if torch.cuda.is_available() else 0)
"""),
    md("## 1. Corpus and the 80/10/10 split"),
    md("`python -m src.build_pairs`"),
    code("""
man = json.loads((ROOT / "data" / "pairs_manifest.json").read_text(encoding="utf-8"))
print("rows:", f"{man['rows']:,}", "| dedup:", man["dedup"], "| id rule:", man["id_rule"])
print("lora shard (dirty, never in the map corpus):", man["lora_file"])
print("excluded prior artefacts:", list(man["excluded_from"]))
print(pd.Series(man["per_file"]).sort_index().to_string())
"""),
    md("## 2. Coreset — which shard, and why"),
    md("`python -m src.embed sentence_sample` then `python -m src.coreset`"),
    code("""
cs = pd.read_csv(RES / "coreset.csv")
chosen = json.loads((ROOT / "data" / "chosen_shard.json").read_text(encoding="utf-8"))
print("chosen:", chosen["file"], "| rule:", chosen["rule"])
display(cs[cs.k == cs.k.min()].sort_values("score", ascending=False)
        .round(3).reset_index(drop=True))
"""),
    md("## 3. Ridge — 512 linear regressions, and the λ sweep"),
    md("`python -m src.embed moments --file <shard>` then `python -m src.fit_ridge`"),
    code("""
display(pd.read_csv(RES / "ridge_sweep.csv").round(6))
for d in ("vi_en", "en_vi"):
    m = np.load(ROOT / "data" / f"map_{d}.npz")
    print(d, "lambda", float(m["lam"]), "| pairs", f"{int(m['n_pairs']):,}",
          "| tokens", f"{int(m['n_tokens']):,}", "| W", m["W"].shape)
"""),
    md("## 4. LightGBM vs ridge on identical tokens"),
    md("`python -m src.fit_lgbm --tokens 120000 --threads 8`"),
    code("""
lg = json.loads((RES / "lgbm_metrics.json").read_text(encoding="utf-8"))
print({k: lg[k] for k in ("n_tokens_train", "n_tokens_val", "r2_ridge_mean", "r2_lgbm_mean",
                          "token_mse_ridge", "token_mse_lgbm", "token_cosine_ridge",
                          "token_cosine_lgbm", "wall_min")})
display(pd.read_csv(RES / "lgbm_vs_ridge.csv").describe().round(4))
"""),
    md("## 5. Copy gate — can the decoder reproduce a vector it was given?"),
    md("`python -m src.copy_gate eval --side en` then `... train --side en` then `... eval --adapter`"),
    code("""
for side in ("en", "vi"):
    for tag in ("", "_lora"):
        p = RES / f"copy_gate_{side}{tag}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            print(f"{side:3s}{tag:6s} exact {d['exact_match']:.3f}  empty {d['empty_rate']:.3f}  "
                  f"words {d['mean_hyp_words']:.1f}/{d['mean_ref_words']:.1f}")
"""),
    md("## 6. Generation — five channels, one decoder"),
    md("`python -m src.train_decoder --direction vi_en --pairs 80000` then `python -m src.generate` then `python -m src.score --comet`"),
    code("""
for d in ("vi_en", "en_vi"):
    p = RES / f"metrics_{d}.json"
    if not p.exists():
        continue
    met = json.loads(p.read_text(encoding="utf-8"))
    rows = [{"channel": k, **v} for k, v in met.items() if isinstance(v, dict) and "bleu" in v]
    tbl = pd.DataFrame(rows).set_index("channel")
    cols = ["bleu", "bleu_ci95", "chrf", "comet", "mean_nll", "degenerate_rate", "empty_rate",
            "vi_word_rate", "mean_hyp_words"]
    print("==", d)
    display(tbl[[c for c in cols if c in tbl.columns]].round(4))
    for k, v in met.items():
        if k.startswith("paired_"):
            print(k, v)
"""),
    md("### 6.1 Examples (same source, one row per channel)"),
    code("""
import itertools
for d in ("vi_en",):
    gens = {c: [json.loads(l) for l in (RES / f"gen_{d}_{c}.jsonl").open(encoding="utf-8")]
            for c in ("mapped", "lgbm", "native", "identity", "copy")
            if (RES / f"gen_{d}_{c}.jsonl").exists()}
    for i in (0, 1, 2):
        print("SRC:", gens["mapped"][i]["src"])
        print("REF:", gens["mapped"][i]["ref"])
        for c, rows in gens.items():
            print(f"  {c:9s}: {rows[i]['hyp']}")
        print()
"""),
    md("## 7. Representation geometry, with the controls that make cosine interpretable"),
    md("`python -m src.eval_geometry --direction vi_en`"),
    code("""
for d in ("vi_en", "en_vi"):
    p = RES / f"geometry_{d}.csv"
    if p.exists():
        print("==", d)
        display(pd.read_csv(p).round(4))
"""),
    md("""
The identity row is the point: without a map the two spaces are nearly orthogonal. The mean row is the
trap: predicting the dataset mean earns cosine for free, which is why retrieval@1 carries the claim.
"""),
    md("## 8. Neural mining (M1–M4)"),
    md("`python -m src.mine`"),
    code("""
print((RES / "mining.md").read_text(encoding="utf-8"))
"""),
    md("## 9. Figures"),
    code("""
from IPython.display import Image, display as d2
for p in sorted((RES / "figures").glob("*.png")):
    print(p.name)
    d2(Image(filename=str(p), width=640))
"""),
    md("""
## 10. What this does not show

* BLEU is measured on 5,000 test sentences of **one shard**; the shard was chosen for
  representativeness, but it is still one shard, and the 95 % CI does not cover shard variance.
* The decoder is 60 M and trained one epoch on 80 k pairs; the absolute BLEU ceiling is set by that,
  not by the map. The claim is the *gap between channels*, not the absolute number.
* `copy` is a readout diagnostic, not a translation system: it is trained with T_out = T_in and can
  learn a length-copy shortcut. Its BLEU is reported to be inspected, not to be trusted.
* COMET is scored on the first 2,000 sentences only (it runs at ~7 sent/s).
"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "research-ver02", "language": "python", "name": "research-ver02"},
    "language_info": {"name": "python", "version": "3.12.10"},
}
nbf.write(nb, OUT)
print(f"wrote {OUT}")
