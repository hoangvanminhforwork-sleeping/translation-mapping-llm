"""Phase 2/3 — fit the *same* 263 200-parameter linear map on decoder behaviour, not on vector error.

    ridge_full   the published E9 map (fitted on ~85 M aligned token pairs, MSE)
    ridge_sub    the same MSE objective refitted on the SAME 20 k pairs the CE arms see
                 (isolates "objective" from "amount of data")
    ce_ridge     W initialised from ridge_sub, trained to minimise the decoder's CE on the reference
    ce_scratch   W initialised at identity, same CE objective (is the ridge basin load-bearing?)
    distill      CE + KL(student || oracle), oracle = decoder reading E_ref  ("the answer")

Everything is teacher-forced against the gold English sentence, the decoder is frozen (the E8/E9
adapter), and only W,b move. 20 k sentences x 2 epochs is a decoder-only job on cached states.

Why this is the right test of the coincidence idea: ridge minimises ||WE - H_target||, which the
project has already shown is not what the decoder reads. If CE-fitted W beats it *while being
further from H_target in representation terms*, then the objective, not the geometry, was the bug.

Usage
    python -m src.fit_behaviour --arms ridge_sub,ce_ridge,ce_scratch --epochs 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from . import cc
from .coincidence import grad_dirs

sys.stdout.reconfigure(encoding="utf-8")
MAPS = cc.WORK / "maps"
MAPS.mkdir(parents=True, exist_ok=True)
D = 512


# ------------------------------------------------------------------ data
def cache_arrays(set_name: str, keys):
    d = cc.CACHE / set_name
    return {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in keys}


def batch_tensor(mm, idx, dtype=torch.float32):
    return torch.from_numpy(np.asarray(mm[idx]).astype(np.float32)).cuda()


# ------------------------------------------------------------------ ridge control
def fit_ridge_sub(vi, en, E_vi, E_ref, lam_rel=1e-8):
    """Closed-form ridge with E9's convention on the same 20 k pairs (see fit_ridge.py).

    Returns (W, b, stats) where stats carries the moments `recolor` needs, so the E9 covariance
    transform can be folded into an affine (W,b) and evaluated by the same code path.
    """
    _, m_vi, o_vi = cc.token_offsets("vi", vi)
    _, m_en, o_en = cc.token_offsets("en", en)
    A = np.zeros((D, D), dtype=np.float64)
    B = np.zeros((D, D), dtype=np.float64)
    C = np.zeros((D, D), dtype=np.float64)
    sx = np.zeros(D, dtype=np.float64)
    sy = np.zeros(D, dtype=np.float64)
    n = 0
    t0 = time.time()
    for s in range(0, len(vi), 256):
        b = slice(s, s + 256)
        X = np.asarray(E_vi[b]).astype(np.float32)                 # (b,64,512)
        Yt = np.asarray(E_ref[b]).astype(np.float32)
        Y, ok = cc.align_np(o_vi[b], m_vi[b], o_en[b], m_en[b], Yt)
        keep = (m_vi[b] > 0) & ok
        x = X[keep].astype(np.float64)
        y = Y[keep].astype(np.float64)
        A += x.T @ x
        B += x.T @ y
        C += y.T @ y
        sx += x.sum(0)
        sy += y.sum(0)
        n += len(x)
    mx, my = sx / n, sy / n
    Ac = A - n * np.outer(mx, mx)
    Bc = B - n * np.outer(mx, my)
    Cc = C - n * np.outer(my, my)
    lam = lam_rel * np.trace(Ac) / D
    W = np.linalg.solve(Ac + lam * np.eye(D), Bc)
    b = my - mx @ W
    print(f"ridge_sub: {n:,} token pairs, lam {lam:.3g}, {time.time() - t0:.0f}s", flush=True)
    stats = {"mx": mx, "my": my, "Cx": Ac / n, "Cy": Cc / n, "n": n}
    return W.astype(np.float32), b.astype(np.float32), stats


def fold_recolor(W, b, stats, t):
    """Fold E9's `ridge + recolor(t)` into one affine map, exactly as `gen_variants.ridge_color`:

        M  = xW + b
        M' = (M - mu_M) A + mu_Y,   A = Sigma_Y^1/2 Sigma_M^-1/2
        out = (1-t) M + t M'   ==  x[W((1-t)I + tA)] + [(1-t)b + t((b-mu_M)A + mu_Y)]

    `mu_M = mu_X W + b` and `Sigma_M = W^T Sigma_X W` (no need to materialise M).
    """
    def sqrtm(S):
        w, V = np.linalg.eigh(S)
        w = np.clip(w, 1e-8, None)
        return (V * np.sqrt(w)) @ V.T

    def inv_sqrtm(S):
        w, V = np.linalg.eigh(S)
        w = np.clip(w, 1e-8, None)
        return (V / np.sqrt(w)) @ V.T

    Cm = W.T @ stats["Cx"] @ W
    Cm = Cm + np.eye(D) * 1e-3 * np.trace(Cm) / D
    Cy = stats["Cy"] + np.eye(D) * 1e-3 * np.trace(stats["Cy"]) / D
    Amat = sqrtm(Cy) @ inv_sqrtm(Cm)
    mu_M = stats["mx"] @ W + b
    W2 = W @ ((1 - t) * np.eye(D) + t * Amat)
    b2 = (1 - t) * b + t * ((b - mu_M) @ Amat + stats["my"])
    return W2.astype(np.float32), b2.astype(np.float32)


# ------------------------------------------------------------------ CE objective
def ce_kl(model, h, am, lab, teacher=None):
    """Mean per-token CE of the reference, plus optional mean per-token KL to a teacher."""
    o = model(encoder_outputs=BaseModelOutput(last_hidden_state=h), attention_mask=am, labels=lab)
    lp = torch.log_softmax(o.logits.float(), -1)
    tgt = lab.clone()
    tgt[tgt == -100] = 0
    tok = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    m = (lab != -100).float()
    ce = (-(tok * m).sum() / m.sum().clamp(min=1))
    if teacher is None:
        return ce, ce.detach(), None
    with torch.no_grad():
        ht, amt = teacher
        ot = model(encoder_outputs=BaseModelOutput(last_hidden_state=ht), attention_mask=amt)
        lt = torch.log_softmax(ot.logits.float(), -1)
    kl = F.kl_div(lp, lt, log_target=True, reduction="none").sum(-1)
    kl = (kl * m).sum() / m.sum().clamp(min=1)
    return ce + kl, ce.detach(), kl.detach()


def train_arm(model, arm, E_vi, m_vi, lab, W0, b0, epochs, batch, lr, val, teacher=None,
              log_path=None):
    N = E_vi.shape[0]
    W = nn.Parameter(torch.from_numpy(W0.copy()).cuda())
    bb = nn.Parameter(torch.from_numpy(b0.copy()).cuda())
    opt = torch.optim.AdamW([W, bb], lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(0)
    t0 = time.time()
    hist = []
    with (log_path or (cc.RESULTS / f"train_{arm}.csv")).open("w", encoding="utf-8") as f:
        f.write("epoch,step,loss,ce,kl,val_ce,sec\n")
    for ep in range(epochs):
        order = rng.permutation(N)
        run = []
        for s in range(0, N, batch):
            idx = order[s:s + batch]
            h = batch_tensor(E_vi, idx) @ W + bb
            am = batch_tensor(m_vi, idx)
            lb = torch.from_numpy(np.asarray(lab[idx]).astype(np.int64)).cuda()
            if teacher is not None:
                th = batch_tensor(teacher["E"], idx)
                tam = batch_tensor(teacher["m"], idx)
                loss, ce, kl = ce_kl(model, h, am, lb, teacher=(th, tam))
            else:
                loss, ce, kl = ce_kl(model, h, am, lb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W, bb], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            run.append(float(ce))
            if len(run) >= 25:
                kls = "nan" if kl is None else f"{float(kl):.5f}"
                with (log_path or (cc.RESULTS / f"train_{arm}.csv")).open("a", encoding="utf-8") as f:
                    f.write(f"{ep},{s},{np.mean(run):.5f},{np.mean(run):.5f},{kls},,"
                            f"{time.time() - t0:.0f}\n")
                print(f"  [{arm}] ep {ep} {s}/{N} ce {np.mean(run):.4f} "
                      f"{len(run) / (time.time() - t0):.1f} b/s", flush=True)
                run = []
        vce = eval_ce(model, val, W, bb)
        hist.append({"epoch": ep, "val_ce": vce, "sec": round(time.time() - t0, 1)})
        print(f"  [{arm}] epoch {ep} val CE {vce:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return W.detach().cpu().numpy(), bb.detach().cpu().numpy(), hist


@torch.no_grad()
def eval_ce(model, val, W, b, batch=16):
    E, m, lab = val["E_vi"], val["m_vi"], val["lab"]
    out = []
    for s in range(0, E.shape[0], batch):
        idx = np.arange(s, min(s + batch, E.shape[0]))
        h = batch_tensor(E, idx) @ W + b
        am = batch_tensor(m, idx)
        lb = torch.from_numpy(np.asarray(lab[idx]).astype(np.int64)).cuda()
        _, ce, _ = ce_kl(model, h, am, lb)
        out.append(float(ce))
    return float(np.mean(out))


# ------------------------------------------------------------------ evaluation
def eval_arm(model, name, W, b, test, refs, lab, mu_ref, sd_ref, basis=None):
    E, m = test["E_vi"], test["m_vi"]
    H = batch_tensor(E, np.arange(E.shape[0])) @ torch.from_numpy(W).cuda() + \
        torch.from_numpy(b).cuda()
    am = batch_tensor(m, np.arange(E.shape[0]))
    hyps = cc.generate(model, H, am)
    ce = cc.ce_per_sentence(model, H, am, lab=lab)
    with torch.no_grad():
        z = (H - torch.from_numpy(mu_ref).cuda()) / torch.from_numpy(sd_ref).cuda()
        zd = (z.pow(2) * am.unsqueeze(-1)).sum(-1).sqrt()      # (B, T) per-token z-distance
        zd = zd[am > 0]                                        # token-level, like ANOMALY.md
    row = {"arm": name,
           "bleu": round(float(cc.bleu(hyps, refs)), 3),
           "chrf": round(float(cc.chrf(hyps, refs)), 3),
           "ce": round(float(ce.mean()), 4),
           "degen": round(float(cc.degen_rate(hyps)), 3),
           "norm": round(float((H * am.unsqueeze(-1)).pow(2).sum((1, 2)).sqrt().mean()), 2),
           "zdist_med": round(float(zd.median()), 2),
           "W_fro": round(float(np.linalg.norm(W)), 2),
           "b_norm": round(float(np.linalg.norm(b)), 2)}
    row["_hyps"] = hyps
    row["_ce"] = ce
    return row


def boot_delta(a, b, refs, metric="bleu", n=1000, seed=0):
    """Paired bootstrap of metric(a) - metric(b) over sentences."""
    rng = np.random.default_rng(seed)
    fn = cc.bleu if metric == "bleu" else cc.chrf
    base = fn(a, refs) - fn(b, refs)
    diffs = []
    N = len(refs)
    for _ in range(n):
        i = rng.integers(0, N, N)
        ra = [refs[j] for j in i]
        diffs.append(fn([a[j] for j in i], ra) - fn([b[j] for j in i], ra))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta": round(float(base), 3), "ci": [round(float(lo), 3), round(float(hi), 3)],
            "p_gt0": round(float(np.mean(np.array(diffs) > 0)), 3)}


# ------------------------------------------------------------------ main
def main(arms, epochs, batch, lr, n_eval, n_boot, force, init_map=None):
    model = cc.load_decoder(dtype=cc.DT_A)
    fit = cache_arrays("fit", ("E_vi", "m_vi", "E_ref", "m_ref", "lab"))
    val = cache_arrays("val", ("E_vi", "m_vi", "lab"))
    test = cache_arrays("test", ("E_vi", "m_vi", "E_nat", "m_nat", "E_ref", "m_ref", "lab"))
    txt = json.loads((cc.CACHE / "fit" / "texts.json").read_text(encoding="utf-8"))
    ttxt = json.loads((cc.CACHE / "test" / "texts.json").read_text(encoding="utf-8"))
    n_eval = min(n_eval, len(ttxt["vi"]))
    refs = ttxt["en"][:n_eval]
    test = {k: v[:n_eval] for k, v in test.items()}
    lab = torch.from_numpy(np.asarray(test["lab"]).astype(np.int64)).cuda()

    HYPS = cc.WORK / "hyps"
    HYPS.mkdir(parents=True, exist_ok=True)
    (HYPS / "refs.json").write_text(json.dumps(refs, ensure_ascii=False), encoding="utf-8")

    # target-space mean and per-dim sd from the *fit* set (E9's "genuine" reference frame)
    E_ref_fit, m_ref_fit = fit["E_ref"], fit["m_ref"]
    acc, acc2, cnt = np.zeros(D, dtype=np.float64), np.zeros(D, dtype=np.float64), 0
    for s in range(0, len(m_ref_fit), 512):
        h = np.asarray(E_ref_fit[s:s + 512]).astype(np.float64)
        mk = np.asarray(m_ref_fit[s:s + 512]) > 0
        acc += (h * mk[..., None]).sum((0, 1))
        acc2 += (h * h * mk[..., None]).sum((0, 1))
        cnt += mk.sum()
    mu_ref = (acc / cnt).astype(np.float32)
    sd_ref = np.sqrt(np.maximum(acc2 / cnt - (acc / cnt) ** 2, 1e-12)).astype(np.float32)

    W_full, b_full = (z.cpu().numpy() for z in cc.load_map())

    def get_sub():
        p = MAPS / "ridge_sub.npz"
        if p.exists() and not force:
            d = np.load(p)
            if "mx" in d:
                return d["W"], d["b"], {k: d[k] for k in ("mx", "my", "Cx", "Cy")}
        W, b, st = fit_ridge_sub(txt["vi"], txt["en"], fit["E_vi"], fit["E_ref"])
        np.savez(p, W=W, b=b, **st)
        return W, b, st

    out_p = cc.RESULTS / "behaviour_maps.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() and not force else {}
    out.update({"n_eval": n_eval, "epochs": epochs, "batch": batch, "lr": lr,
                "n_fit": int(fit["E_vi"].shape[0])})
    out.setdefault("arms", {})

    for arm in arms:
        p = MAPS / f"{arm}.npz"
        if p.exists() and not force:
            d = np.load(p)
            W, b = d["W"], d["b"]
            hist = json.loads(str(d["hist"])) if "hist" in d else []
            print(f"{arm}: loaded {p.name}", flush=True)
        elif arm == "ridge_full":
            W, b, hist = W_full, b_full, []
        elif arm == "ridge_sub":
            W, b, _ = get_sub()
            hist = []
        elif arm.startswith("ridge_recolor"):
            t = float(arm.replace("ridge_recolor", "")) / 100
            W0, b0, st = get_sub()
            W, b = fold_recolor(W0, b0, st, t)
            hist = [{"recolor_t": t}]
        elif arm.startswith("ridge_scale"):
            k = float(arm.replace("ridge_scale", "")) / 100
            W0, b0, _ = get_sub()
            W, b = (W0 * k).astype(np.float32), (b0 * k).astype(np.float32)
            hist = [{"scale_k": k}]
        elif arm in ("ce_ridge", "ce_scratch", "distill") or arm.startswith("ce_"):
            if init_map:
                d0 = np.load(MAPS / f"{init_map}.npz")
                W0, b0 = d0["W"], d0["b"]
                print(f"{arm}: init from {init_map}.npz", flush=True)
            elif arm == "ce_ridge" or arm.startswith("ce_recolor"):
                W0, b0, _ = get_sub()
            else:
                W0, b0 = np.eye(D, dtype=np.float32), np.zeros(D, dtype=np.float32)
            teacher = {"E": fit["E_ref"], "m": fit["m_ref"]} if arm == "distill" else None
            W, b, hist = train_arm(model, arm, fit["E_vi"], fit["m_vi"], fit["lab"],
                                   W0, b0, epochs, batch, lr, val, teacher=teacher)
        else:
            raise SystemExit(f"unknown arm {arm}")
        if not p.exists() or force:
            np.savez(p, W=W, b=b, hist=json.dumps(hist))

        row = eval_arm(model, arm, W, b, test, refs, lab, mu_ref, sd_ref)
        hyps = row.pop("_hyps")
        row.pop("_ce", None)
        (HYPS / f"{arm}.json").write_text(json.dumps(hyps, ensure_ascii=False), encoding="utf-8")
        row["val_ce"] = round(float(eval_ce(model, val, torch.from_numpy(W).cuda(),
                                            torch.from_numpy(b).cuda())), 4)
        row["hist"] = hist
        out["arms"][arm] = row
        print(f"{arm:16s} bleu {row['bleu']:.2f}  chrf {row['chrf']:.2f}  ce {row['ce']:.3f}  "
              f"degen {row['degen']:.3f}  |H| {row['norm']:.1f}  zdist {row['zdist_med']:.1f}  "
              f"val_ce {row['val_ce']:.3f}", flush=True)
        out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        torch.cuda.empty_cache()

    # ---- paired bootstrap against the published map (from persisted hypotheses)
    base = HYPS / "ridge_full.json"
    if base.exists():
        bh = json.loads(base.read_text(encoding="utf-8"))
        for arm in list(out["arms"]):
            f = HYPS / f"{arm}.json"
            if not f.exists() or arm == "ridge_full":
                continue
            h = json.loads(f.read_text(encoding="utf-8"))
            out["arms"][arm]["boot_vs_ridge_full"] = {
                "bleu": boot_delta(h, bh, refs, "bleu", n_boot),
                "chrf": boot_delta(h, bh, refs, "chrf", n_boot)}
            print(f"{arm:16s} dBLEU {out['arms'][arm]['boot_vs_ridge_full']['bleu']}", flush=True)

    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {out_p}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="ridge_sub,ce_ridge,ce_scratch")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--init-map", default=None)
    a = ap.parse_args()
    main([x for x in a.arms.split(",") if x], a.epochs, a.batch, a.lr, a.n_eval, a.n_boot, a.force, a.init_map)
