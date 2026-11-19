"""Emit the markdown tables of the report from the saved JSON files (no number is typed by hand).

Usage
    python -m src.tables            # prints tables to stdout, also writes results/TABLES.md
"""
from __future__ import annotations

import glob
import json
import sys

from . import config as C

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:      # inside a notebook kernel stdout is an OutStream without reconfigure
    pass

ARM_ORDER = ["A0_greedy", "A1_beam", "A3_lenratio", "A5_mbr_chrf", "A6_rt_str", "A7_rt_lat",
             "A7c_rt_lat_centred", "A8_learned_string", "A8_learned_latent", "A9_oracle",
             "A5_lora_beam8", "A5_lora_e1_beam8", "A5_lora_mbr", "A5_lora_greedy"]


def load_all() -> dict:
    merged: dict[str, dict] = {}
    for path in sorted(glob.glob(str(C.RESULTS / "metrics_*.json"))):
        if "probe" in path:
            continue
        d = json.load(open(path, encoding="utf-8"))
        tag = path.split("metrics_")[-1].replace(".json", "")
        for k, v in d.items():
            if not isinstance(v, dict) or "bleu" not in v:
                continue
            e = merged.setdefault(k, {})
            for f in ("bleu", "chrf", "comet", "bleu_ci95", "comet_sent", "degenerate_rate",
                      "vi_word_rate", "mean_hyp_words"):
                if f in v:
                    e[f"{f}@{tag}"] = v[f]
            for f in v:
                if f.startswith("paired_vs_"):
                    e[f"{f}@{tag}"] = v[f]
    return merged


def pick(e: dict, field: str, order=("10k", "lora", "lora_e1", "all", "core", "lp", "lora_core")):
    for tag in order:
        if f"{field}@{tag}" in e:
            return e[f"{field}@{tag}"]
    return None


def main() -> None:
    m = load_all()
    rows = []
    for arm in ARM_ORDER + [a for a in m if a not in ARM_ORDER]:
        if arm not in m or arm.startswith(("MIX_", "RLORA_", "A10_", "A12_")):
            continue
        e = m[arm]
        rows.append((arm, pick(e, "bleu"), pick(e, "chrf"),
                     pick(e, "comet", ("lora_e1_core", "lora_core", "core", "all", "10k", "lora"))))

    lines = ["| arm | BLEU | chrF | COMET | % of frozen-oracle headroom |", "|---|---|---|---|---|"]
    base = pick(m.get("A1_beam", {}), "bleu")
    orac = pick(m.get("A9_oracle", {}), "bleu")
    for arm, b, c, co in rows:
        cap = ""
        if None not in (base, orac, b) and orac > base:
            cap = f"{100 * (b - base) / (orac - base):+.0f} %"
        bb = f"{b:.2f}" if b is not None else "—"
        cc = f"{c:.2f}" if c is not None else "—"
        coo = f"{co:.4f}" if co is not None else "—"
        lines.append(f"| {arm} | {bb} | {cc} | {coo} | {cap} |")

    diag = C.RESULTS / "rank_diag.json"
    if diag.exists():
        d = json.load(open(diag, encoding="utf-8"))
        lines += ["", "| utility | Spearman vs COMET | top-1 accuracy |", "|---|---|---|"]
        for k, v in d.get("utilities", {}).items():
            lines.append(f"| {k} | {v['spearman']:+.3f} | {v['top1_acc']:.3f} |")
        for k in ("learned_string", "learned_latent"):
            if k in d:
                v = d[k]
                lines.append(f"| {k} (MLP, {v['n_params']} params) | "
                             f"{v['spearman_mean']:+.3f} ± {v['spearman_sd']:.3f} | "
                             f"{v['top1_acc']:.3f} |")

    out = "\n".join(lines)
    print(out)
    (C.RESULTS / "TABLES.md").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
