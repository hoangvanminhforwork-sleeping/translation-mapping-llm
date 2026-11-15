"""Print the paper's result tables straight from results/*.json — no hand transcription.

Usage:  python tools/make_tables.py > agent-work/tables.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.stdout.reconfigure(encoding="utf-8")


def gen_table(direction: str):
    p = RES / f"metrics_{direction}.json"
    if not p.exists():
        return f"_(missing {p.name})_"
    m = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for c, v in m.items():
        if not (isinstance(v, dict) and "bleu" in v):
            continue
        lo, hi = v.get("bleu_ci95", [float("nan")] * 2)
        rows.append({"channel": c, "BLEU": v["bleu"], "95% CI": f"[{lo:.2f}, {hi:.2f}]",
                     "chrF": v["chrf"], "COMET": v.get("comet", float("nan")),
                     "NLL": v.get("mean_nll", float("nan")),
                     "empty": v["empty_rate"], "degen": v["degenerate_rate"],
                     "vi-words": v["vi_word_rate"], "words": v["mean_hyp_words"], "n": v["n"]})
    df = pd.DataFrame(rows).set_index("channel")
    out = ["| channel | BLEU | 95% CI | chrF | COMET | ref NLL | empty | degen | vi-words | hyp words | n |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in df.iterrows():
        out.append(f"| `{c}` | {r['BLEU']:.2f} | {r['95% CI']} | {r['chrF']:.2f} | "
                   f"{r['COMET']:.3f} | {r['NLL']:.2f} | {r['empty']:.3f} | {r['degen']:.3f} | "
                   f"{r['vi-words']:.4f} | {r['words']:.1f} | {r['n']} |")
    out.append("")
    for k, v in m.items():
        if k.startswith("paired_"):
            out.append(f"* {k}: {v['mean']:+.2f} BLEU, 95% CI [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}], "
                       f"P(diff<=0)={v['p_other_le_mapped']:.3f}")
    return "\n".join(out)


def geo_table(direction: str):
    p = RES / f"geometry_{direction}.csv"
    if not p.exists():
        return f"_(missing {p.name})_"
    d = pd.read_csv(p)
    out = ["| variant | token cos | token MSE | sentence cos | R@1 | margin+ |", "|---|---|---|---|---|---|"]
    for r in d.itertuples():
        out.append(f"| {r.variant} | {r.token_cosine:.4f} | {r.token_mse:.4f} | {r.sentence_cosine:.4f} | "
                   f"{r.retrieval_at_1:.4f} | {r.margin_frac_positive:.3f} |")
    return "\n".join(out)


def main():
    for d in ("vi_en", "en_vi"):
        print(f"\n## Generation {d}\n")
        print(gen_table(d))
        print(f"\n## Geometry {d}\n")
        print(geo_table(d))
    p = RES / "mining.json"
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        print("\n## M3 decoder tolerance\n")
        for d in ("vi_en", "en_vi"):
            for c, v in m.get(f"m3_decoder_tolerance_{d}", {}).items():
                print(f"* {d} {c}: pearson {v['pearson_cos_nll']:+.3f}, spearman {v['spearman_cos_nll']:+.3f}, "
                      f"NLL by cos quartile {[round(x,2) for x in v['nll_quartiles']]}")
        print("\n## M4 layer transfer\n")
        for d in ("vi_en", "en_vi"):
            v = m.get(f"m4_layer_transfer_{d}", {})
            if v:
                print(f"* {d}: " + " ".join(f"L{l}={v['layers'][str(l)]['token_cosine_mean']:.4f}"
                                             for l in range(len(v["layers"]))))


if __name__ == "__main__":
    main()
