"""Every figure in the paper, generated from the artefacts in results/ (never from constants).

Run after the pipeline:  python -m src.figures
Outputs: results/figures/*.png (200 dpi) and results/figures/architecture.svg
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

sys.stdout.reconfigure(encoding="utf-8")
FIG = C.RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 200, "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
C_MAIN, C_ALT, C_CTRL = "#1f4e79", "#c55a11", "#7f7f7f"

ARCH_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="960" height="250" viewBox="0 0 960 250">
<defs><marker id="a" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
<path d="M0,0 L10,4 L0,8 z" fill="#333"/></marker></defs>
<rect x="10" y="95" width="120" height="60" rx="10" fill="#eef3f8" stroke="#1f4e79"/>
<text x="70" y="120" text-anchor="middle" font-family="Segoe UI" font-size="13">Source text</text>
<text x="70" y="138" text-anchor="middle" font-family="Segoe UI" font-size="11" fill="#555">x (vi)</text>
<rect x="170" y="95" width="140" height="60" rx="10" fill="#dbe7f3" stroke="#1f4e79"/>
<text x="240" y="118" text-anchor="middle" font-family="Segoe UI" font-size="13">M_s encoder</text>
<text x="240" y="136" text-anchor="middle" font-family="Segoe UI" font-size="11" fill="#555">t5-small-vi (60M)</text>
<rect x="350" y="95" width="150" height="60" rx="10" fill="#fbe5d6" stroke="#c55a11"/>
<text x="425" y="118" text-anchor="middle" font-family="Segoe UI" font-size="13">Map f (512x512)</text>
<text x="425" y="136" text-anchor="middle" font-family="Segoe UI" font-size="11" fill="#555">ridge | LightGBM</text>
<rect x="540" y="95" width="150" height="60" rx="10" fill="#dbe7f3" stroke="#1f4e79"/>
<text x="615" y="118" text-anchor="middle" font-family="Segoe UI" font-size="13">M_t decoder</text>
<text x="615" y="136" text-anchor="middle" font-family="Segoe UI" font-size="11" fill="#555">t5-small (60M) + LoRA</text>
<rect x="730" y="95" width="120" height="60" rx="10" fill="#eef3f8" stroke="#1f4e79"/>
<text x="790" y="120" text-anchor="middle" font-family="Segoe UI" font-size="13">Target text</text>
<text x="790" y="138" text-anchor="middle" font-family="Segoe UI" font-size="11" fill="#555">y-hat (en)</text>
<line x1="130" y1="125" x2="166" y2="125" stroke="#333" stroke-width="2" marker-end="url(#a)"/>
<line x1="310" y1="125" x2="346" y2="125" stroke="#333" stroke-width="2" marker-end="url(#a)"/>
<line x1="500" y1="125" x2="536" y2="125" stroke="#333" stroke-width="2" marker-end="url(#a)"/>
<line x1="690" y1="125" x2="726" y2="125" stroke="#333" stroke-width="2" marker-end="url(#a)"/>
<text x="425" y="80" text-anchor="middle" font-family="Segoe UI" font-size="11" fill="#c55a11">H_s -> H_t (token-aligned)</text>
<text x="425" y="185" text-anchor="middle" font-family="Segoe UI" font-size="11" fill="#555">fitted on 80% of one shard, never sees the test 10%</text>
<text x="240" y="60" text-anchor="middle" font-family="Segoe UI" font-size="11" fill="#555">frozen</text>
<text x="615" y="60" text-anchor="middle" font-family="Segoe UI" font-size="11" fill="#555">frozen except LoRA</text>
</svg>"""


def save(fig, name):
    p = FIG / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p.name)


def fig_coreset():
    p = C.RESULTS / "coreset.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.k == d.k.min()]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    w = 0.38
    ax.bar(x - w / 2, d.representativeness, w, label="representativeness", color=C_MAIN)
    ax.bar(x + w / 2, d.diversity, w, label="diversity (cluster entropy)", color=C_ALT)
    for i, r in enumerate(d.itertuples()):
        if r.chosen:
            ax.annotate("chosen", (i, max(r.representativeness, r.diversity) + 0.02), ha="center",
                        fontsize=7, color="#c00000")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("train-", "").replace("-of-00014", "") for f in d.file], fontsize=7)
    ax.set_xlabel("MTet training shard")
    ax.set_ylabel("score")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    save(fig, "fig_coreset.png")


def fig_ridge():
    p = C.RESULTS / "ridge_sweep.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for dirn, g in d.groupby("direction"):
        ax.plot(np.clip(g["lambda"], 1e-5, None), g.mse_val, marker="o", ms=3,
                label=dirn.replace("_", "→"), color=C_MAIN if dirn == "vi_en" else C_ALT)
    ax.set_xscale("log")
    ax.set_xlabel("ridge λ")
    ax.set_ylabel("validation MSE (per token)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig_ridge_sweep.png")


def fig_r2():
    p = C.RESULTS / "lgbm_metrics.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    r, l = np.asarray(d["r2_ridge_per_dim"]), np.asarray(d["r2_lgbm_per_dim"])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.scatter(r, l, s=6, alpha=0.6, color=C_MAIN, edgecolors="none")
    lo, hi = min(r.min(), l.min()), max(r.max(), l.max())
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color=C_CTRL)
    ax.set_xlabel("ridge $R^2$ (per dimension)")
    ax.set_ylabel("LightGBM $R^2$ (per dimension)")
    ax.set_title(f"512 output dimensions, Δmean {l.mean()-r.mean():+.4f}", fontsize=8)
    save(fig, "fig_r2_ridge_vs_lgbm.png")


def fig_tolerance():
    out = {}
    for dirn in ("vi_en", "en_vi"):
        p = C.DATA / f"per_sent_{dirn}.npz"
        f = C.RESULTS / f"gen_{dirn}_mapped.jsonl"
        if not (p.exists() and f.exists()):
            continue
        z = np.load(p)
        if "map_sent_cos" not in z:
            continue
        rows = [json.loads(line) for line in f.open(encoding="utf-8")]
        cos = np.asarray(z["map_sent_cos"])[:len(rows)]
        nll = np.array([r["nll"] for r in rows])
        out[dirn] = (cos, nll)
    if not out:
        return
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    for i, (dirn, (cos, nll)) in enumerate(out.items()):
        ax.scatter(cos, nll, s=4, alpha=0.35, color=[C_MAIN, C_ALT][i], edgecolors="none",
                   label=dirn.replace("_", "→"))
        q = np.argsort(cos)
        k = max(len(cos) // 25, 1)
        xs = [cos[q[j:j + k]].mean() for j in range(0, len(q), k)]
        ys = [nll[q[j:j + k]].mean() for j in range(0, len(q), k)]
        ax.plot(xs, ys, color=[C_MAIN, C_ALT][i], lw=2)
    ax.set_xlabel("per-sentence cosine (mapped vs true target)")
    ax.set_ylabel("decoder NLL of the reference")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig_tolerance.png")


def fig_layers():
    p = C.RESULTS / "mining.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.8, 2.6))
    for dirn, col in (("vi_en", C_MAIN), ("en_vi", C_ALT)):
        m = d.get(f"m4_layer_transfer_{dirn}", {})
        if not m:
            continue
        ys = [m["layers"][str(l)]["token_cosine_mean"] for l in range(len(m["layers"]))]
        ax.plot(range(len(ys)), ys, marker="o", ms=3, color=col, label=dirn.replace("_", "→"))
    ax.set_xlabel("source encoder layer (0 = embeddings)")
    ax.set_ylabel("token cosine to target final state")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig_layers.png")


def fig_training():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for side, col in (("en", C_MAIN), ("vi", C_ALT)):
        p = C.RESULTS / f"copy_train_{side}.csv"
        if p.exists():
            d = pd.read_csv(p)
            axes[0].plot(d.step, d.loss, lw=1, color=col, label=side)
    axes[0].set_title("copy LoRA (identity task)", fontsize=8)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].legend(frameon=False, fontsize=7)
    for dirn, col in (("vi_en", C_MAIN), ("en_vi", C_ALT)):
        p = C.RESULTS / f"decoder_train_{dirn}.csv"
        if p.exists():
            d = pd.read_csv(p)
            for cond, ls in (("native", ":"), ("mapped", "-")):
                g = d[d.cond == cond]
                axes[1].plot(g.step, g.loss, ls, lw=1, color=col, alpha=0.9,
                             label=f"{dirn} {cond}")
    axes[1].set_title("translation LoRA (native vs mapped)", fontsize=8)
    axes[1].set_xlabel("step")
    axes[1].legend(frameon=False, fontsize=6)
    save(fig, "fig_training.png")


def fig_bleu():
    rows = []
    for dirn in ("vi_en", "en_vi"):
        p = C.RESULTS / f"metrics_{dirn}.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            for c, v in m.items():
                if isinstance(v, dict) and "bleu" in v:
                    lo, hi = v.get("bleu_ci95", [np.nan, np.nan])
                    rows.append({"direction": dirn, "channel": c, "bleu": v["bleu"],
                                 "lo": lo, "hi": hi, "chrf": v["chrf"]})
    if not rows:
        return
    d = pd.DataFrame(rows)
    order = ["mapped", "lgbm", "native", "identity", "copy"]
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    w = 0.38
    for i, dirn in enumerate(d.direction.unique()):
        g = d[d.direction == dirn].set_index("channel").reindex(order).dropna(subset=["bleu"])
        x = np.arange(len(g)) + (i - 0.5) * w
        ax.bar(x, g.bleu, w, yerr=[g.bleu - g.lo, g.hi - g.bleu], capsize=2,
               color=[C_MAIN, C_ALT][i], label=dirn.replace("_", "→"))
        ax.set_xticks(np.arange(len(g)))
        ax.set_xticklabels(g.index, rotation=20, fontsize=7)
    ax.set_ylabel("BLEU (95% bootstrap CI)")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig_bleu.png")


def main():
    (FIG / "architecture.svg").write_text(ARCH_SVG, encoding="utf-8")
    print("wrote architecture.svg")
    for f in (fig_coreset, fig_ridge, fig_r2, fig_tolerance, fig_layers, fig_training, fig_bleu):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print(f"{f.__name__}: skipped ({e})")


if __name__ == "__main__":
    main()
