"""Neural mining: why does a linear map keep winning, and what does the decoder actually tolerate?

Four measurements, each one a falsifiable statement about the *mechanism*, not the score:

  M1 nonlinearity gap    per-dimension R^2 of ridge vs 512 LightGBM boosters, on identical tokens.
                         If trees beat ridge, the residual is nonlinear and the linear map is a
                         limitation. If they tie, the residual is irreducible cross-lingual variance.
  M2 tree complexity     trees and features actually used per dimension. A booster that stops after
                         a few splits is a linear model with extra steps.
  M3 decoder tolerance   per-sentence mapped cosine vs the decoder NLL of the reference. A flat
                         relation means the decoder does not care about ~0.7 cosine; a steep one
                         means representation error is the bottleneck.
  M4 layer-wise transfer cosine between the mapped state at source layer l and the target's final
                         state. Shows how much of the target is already present before the last layer.

Outputs: results/mining.json, results/mining.md (tables for the paper)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from . import config as C
from .embed import Encoders, load_chosen
from .tokenbatch import align_torch
from .train_decoder import direction_sides
from .tokutil import load_tok

sys.stdout.reconfigure(encoding="utf-8")


def m1_m2() -> dict:
    lg = json.loads((C.RESULTS / "lgbm_metrics.json").read_text(encoding="utf-8"))
    r2r = np.asarray(lg["r2_ridge_per_dim"])
    r2l = np.asarray(lg["r2_lgbm_per_dim"])
    d = r2l - r2r
    return {
        "r2_ridge_mean": float(r2r.mean()), "r2_lgbm_mean": float(r2l.mean()),
        "delta_mean": float(d.mean()), "delta_median": float(np.median(d)),
        "frac_dims_improved": float((d > 1e-4).mean()), "frac_dims_worse": float((d < -1e-4).mean()),
        "delta_p90": float(np.percentile(d, 90)), "delta_p10": float(np.percentile(d, 10)),
        "corr_ridge_r2_vs_delta": float(pearsonr(r2r, d)[0]) if r2r.std() > 0 else None,
        "trees_per_dim_mean": float(np.mean(lg["trees_per_dim"])),
        "features_used_mean": float(np.mean(lg["features_used_per_dim"])),
        "token_mse_ridge": lg["token_mse_ridge"], "token_mse_lgbm": lg["token_mse_lgbm"],
        "token_cosine_ridge": lg["token_cosine_ridge"], "token_cosine_lgbm": lg["token_cosine_lgbm"],
        "n_tokens_train": lg["n_tokens_train"],
    }


def m3(direction: str) -> dict:
    p = C.DATA / f"per_sent_{direction}.npz"
    if not p.exists():
        return {}
    z = np.load(p)
    ids = z["ids"]
    out = {}
    for cond, key in (("mapped", "map_sent_cos"), ("lgbm", "lgbm_sent_cos")):
        f = C.RESULTS / f"gen_{direction}_{cond}.jsonl"
        if not f.exists() or key not in z:
            continue
        rows = [json.loads(line) for line in f.open(encoding="utf-8")]
        nll = np.array([r["nll"] for r in rows])
        cos = np.asarray(z[key])
        # gen rows are the first len(rows) test ids in the same order
        m = min(len(nll), len(cos))
        nll, cos = nll[:m], cos[:m]
        out[cond] = {"n": int(m),
                     "pearson_cos_nll": float(pearsonr(cos, nll)[0]),
                     "spearman_cos_nll": float(spearmanr(cos, nll)[0]),
                     "nll_quartiles": [float(np.mean(nll[q])) for q in
                                       np.array_split(np.argsort(cos), 4)],
                     "cos_quartiles": [float(np.mean(cos[q])) for q in np.array_split(np.argsort(cos), 4)]}
    return out


@torch.no_grad()
def m4(direction: str, n: int = 2000, batch: int = 128) -> dict:
    src, tgt = direction_sides(direction)
    d = load_chosen(("vi", "en", "split"))
    te = [i for i, s in enumerate(d["split"]) if s == "test"][:n]
    texts_src = [d[src][i] for i in te]
    texts_tgt = [d[tgt][i] for i in te]
    enc = Encoders()
    tok_s, tok_t = load_tok(src), load_tok(tgt)
    z = np.load(C.MAP[direction])
    W = torch.from_numpy(z["W"].astype(np.float32)).cuda()
    b = torch.from_numpy(z["b"].astype(np.float32)).cuda()
    per_layer = {}
    for s in range(0, len(texts_src), batch):
        bs = texts_src[s:s + batch]
        bt = texts_tgt[s:s + batch]
        es = tok_s.backend_tokenizer.encode_batch(bs, add_special_tokens=True)
        et = tok_t.backend_tokenizer.encode_batch(bt, add_special_tokens=True)
        Ls = min(C.MAX_LEN, max(len(e.ids) for e in es))
        Lt = min(C.MAX_LEN, max(len(e.ids) for e in et))
        ids_s = torch.zeros((len(bs), Ls), dtype=torch.long)
        am_s = torch.zeros((len(bs), Ls), dtype=torch.long)
        off_s = torch.zeros((len(bs), Ls, 2), dtype=torch.long)
        ids_t = torch.zeros((len(bt), Lt), dtype=torch.long)
        am_t = torch.zeros((len(bt), Lt), dtype=torch.long)
        off_t = torch.zeros((len(bt), Lt, 2), dtype=torch.long)
        for r, e in enumerate(es):
            m = min(len(e.ids), Ls)
            ids_s[r, :m] = torch.tensor(e.ids[:m]); am_s[r, :m] = 1
            off_s[r, :m] = torch.tensor(np.asarray(e.offsets[:m]))
        for r, e in enumerate(et):
            m = min(len(e.ids), Lt)
            ids_t[r, :m] = torch.tensor(e.ids[:m]); am_t[r, :m] = 1
            off_t[r, :m] = torch.tensor(np.asarray(e.offsets[:m]))
        ids_s, am_s, off_s = ids_s.cuda(), am_s.cuda(), off_s.cuda()
        ids_t, am_t, off_t = ids_t.cuda(), am_t.cuda(), off_t.cuda()
        hs = enc.model[src].encoder(input_ids=ids_s, attention_mask=am_s, output_hidden_states=True).hidden_states
        ht = enc.model[tgt].encoder(input_ids=ids_t, attention_mask=am_t).last_hidden_state
        Y, keep, _ = align_torch(off_s, am_s, off_t, am_t, ht)
        Yn = Y[keep].float()
        Yu = Yn / Yn.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        for l, H in enumerate(hs):
            P = (H[keep].float() @ W + b)
            cos = (P / P.norm(dim=-1, keepdim=True).clamp(min=1e-9) * Yu).sum(-1)
            per_layer.setdefault(l, []).append(cos.cpu().numpy())
    out = {"layers": {str(l): {"token_cosine_mean": float(np.concatenate(v).mean()),
                              "n_tokens": int(sum(len(x) for x in v))}
                      for l, v in per_layer.items()},
           "n_pairs": len(texts_src)}
    print(f"m4 {direction}: " + "  ".join(f"L{l} {d['token_cosine_mean']:.4f}" for l, d in out["layers"].items()))
    return out


def main():
    out = {"m1_m2_nonlinearity": m1_m2()}
    for direction in ("vi_en", "en_vi"):
        out[f"m3_decoder_tolerance_{direction}"] = m3(direction)
        out[f"m4_layer_transfer_{direction}"] = m4(direction)
    (C.RESULTS / "mining.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    a = out["m1_m2_nonlinearity"]
    lines = ["# Neural-mining tables (generated by src/mine.py)", "",
             "## M1 — nonlinearity gap (512 dims, identical token sample)", "",
             "| metric | ridge | LightGBM |", "|---|---|---|",
             f"| mean per-dim R² | {a['r2_ridge_mean']:.4f} | {a['r2_lgbm_mean']:.4f} |",
             f"| token MSE | {a['token_mse_ridge']:.4f} | {a['token_mse_lgbm']:.4f} |",
             f"| token cosine | {a['token_cosine_ridge']:.4f} | {a['token_cosine_lgbm']:.4f} |",
             "", f"ΔR² mean {a['delta_mean']:+.5f}, median {a['delta_median']:+.5f}; "
                 f"improved {a['frac_dims_improved']*100:.1f}% of dims, worse {a['frac_dims_worse']*100:.1f}%.",
             f"Trees/dim {a['trees_per_dim_mean']:.0f}, features used/dim {a['features_used_mean']:.0f}/512.",
             "", "## M3 — decoder tolerance (mapped channel)", "",
             "| channel | Pearson(cos, NLL) | Spearman | NLL by cos quartile (low→high cos) |",
             "|---|---|---|---|"]
    for direction in ("vi_en", "en_vi"):
        for cond, d in out.get(f"m3_decoder_tolerance_{direction}", {}).items():
            q = ", ".join(f"{x:.2f}" for x in d["nll_quartiles"])
            lines.append(f"| {direction} {cond} | {d['pearson_cos_nll']:+.3f} | {d['spearman_cos_nll']:+.3f} | {q} |")
    lines += ["", "## M4 — layer-wise transfer (token cosine to the target's final state)", "",
              "| direction | " + " | ".join(f"L{l}" for l in range(7)) + " |", "|---|" + "---|" * 7]
    for direction in ("vi_en", "en_vi"):
        d = out.get(f"m4_layer_transfer_{direction}", {})
        if d:
            lines.append(f"| {direction} | " + " | ".join(
                f"{d['layers'][str(l)]['token_cosine_mean']:.4f}" for l in range(7)) + " |")
    (C.RESULTS / "mining.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {C.RESULTS / 'mining.json'} and mining.md")


if __name__ == "__main__":
    main()
