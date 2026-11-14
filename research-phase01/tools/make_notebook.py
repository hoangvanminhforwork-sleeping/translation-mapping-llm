"""Build 01_results.ipynb — the readable artefact of phase 1.

The notebook does not run the pipeline (that would need 3 h of GPU and would make the numbers
unreproducible from a clean checkout). It reads the artefacts in `results/` and `data/`, so every
number in it is a measured one, and it fails loudly if a file is missing instead of inventing a table.
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "01_results.ipynb"


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


cells = [
    md("""
# Phase 1 — cross-model linear mapping, vi → en

Pipeline: `token(vi) → vit5 encoder → per-token linear map W → flan decoder → token(en)`.

This notebook is the report of record for the numbers. It reads what the pipeline wrote; it does not
recompute it. Regenerate any artefact with the command printed above the cell.
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
    md("## 1. Corpus — what '8M sentences' actually is"),
    md("`python -m src.build_pairs`"),
    code("""
man = json.loads((ROOT / "data" / "pairs_manifest.json").read_text(encoding="utf-8"))
print("rows kept:", f"{man['rows']:,}")
print("split:", man["split"])
print("rule:", man["split_rule"])
print("vi side:", man["vi_side_rule"])
"""),
    md("## 2. Map fitting — ridge sweep"),
    md("`python -m src.fit_tok shards` then `python -m src.fit_tok fit`"),
    code("""
sweep = pd.read_csv(RES / "ridge_sweep.csv")
display(sweep)
m = np.load(ROOT / "data" / "map_vi_en.npz")
print("chosen lambda", float(m["lam"]), "| shards", int(m["shards"]),
      "| pairs", f"{int(m['n_pairs']):,}", "| tokens", f"{int(m['n_tokens']):,}")
"""),
    md("## 3. Geometry, with the controls that make cosine interpretable"),
    md("`python -m src.eval_geometry`"),
    code("""
geo = pd.read_csv(RES / "mapping_metrics.csv")
meta = json.loads((RES / "mapping_metrics.json").read_text(encoding="utf-8"))
display(geo)
print({k: meta[k] for k in ("n_test_pairs", "chance_r1", "protocol")})
"""),
    md("""
The identity row is the point: without a map the two spaces are nearly orthogonal. The mean row is the
trap: predicting the dataset mean earns cosine for free, which is why retrieval@1 is the number that
carries the claim.
"""),
    md("## 4. Generation — three channels, one decoder"),
    md("`python -m src.train_decoder --pairs 100000` then `python -m src.generate` then `python -m src.score`"),
    code("""
met = json.loads((RES / "metrics.json").read_text(encoding="utf-8"))
rows = [{"channel": k, **{kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}}
        for k, v in met.items() if isinstance(v, dict) and "bleu" in v]
tbl = pd.DataFrame(rows).set_index("channel")
cols = ["bleu", "chrf", "comet", "degenerate_rate", "vi_word_rate", "empty_rate", "mean_hyp_words"]
display(tbl[[c for c in cols if c in tbl.columns]].round(4))
if "paired_native_minus_mapped" in met:
    print(json.dumps(met["paired_native_minus_mapped"], indent=2))
"""),
    md("### 4.1 Examples (one per channel, same Vietnamese source)"),
    code("""
gens = {c: [json.loads(l) for l in (RES / f"gen_{c}.jsonl").open(encoding="utf-8")]
        for c in met if (RES / f"gen_{c}.jsonl").exists()}
i = 0
print("VI :", gens["mapped"][i]["vi"])
print("REF:", gens["mapped"][i]["ref"])
for c, rows in gens.items():
    print(f"{c:9s}:", rows[i]["hyp"])
"""),
    md("## 5. Food probe on the frozen flan encoder (§3.1)"),
    md("`python -m src.probe`"),
    code("""
if (RES / "probe_metrics.json").exists():
    display(pd.Series(json.loads((RES / "probe_metrics.json").read_text(encoding="utf-8"))))
else:
    print("probe not run")
"""),
    md("## 6. Data track (M1)"),
    code("""
dw = ROOT / "data-work"
for p in sorted(dw.glob("structured/*")):
    print(f"{p.relative_to(ROOT)}  {p.stat().st_size/1024:.0f} KB")
for p in sorted(dw.glob("markdown/*.md")):
    print(f"{p.relative_to(ROOT)}  {p.stat().st_size/1024:.0f} KB")
"""),
    md("""
## 7. What this does not show

* en → vi is measured only geometrically; generating Vietnamese needs the vit5 decoder.
* The decoder is base-size and trained for one epoch on 100k pairs; the BLEU ceiling is set by that,
  not by the map. The claim is the *gap between channels*, not the absolute BLEU.
* COMET is reported only if `python -m src.score --comet` was run; the da reference-free variant
  requires the English source, so be explicit about which COMET checkpoint produced the number.
"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "research-phase01", "language": "python", "name": "research-phase01"},
    "language_info": {"name": "python", "version": "3.12.10"},
}
nbf.write(nb, OUT)
print(f"wrote {OUT}")
