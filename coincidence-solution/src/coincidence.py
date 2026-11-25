"""Phase 1 — the coincidence measurement: how far can the memory move before the decoder changes
its mind, and is the map's error in the directions the decoder reads?

The mental model asks: does there exist H' != H with d(H') ~ d(H)?  A many-to-one map always has
such level sets, so the *measurable* questions are

    Q1  how large is the level set, in units of memory displacement?
    Q2  is it isotropic?  (random directions vs the direction of the map's actual error)
    Q3  does the map's error lie inside the level set, and if not, is the residual concentrated in
        the few directions the decoder is sensitive to (the "null space" claim)?

Protocol, per oracle memory H0 (see cc.py for `ref` / `nat`):
    * decode greedily -> y0, compute CE on the reference
    * move H0 by  eps * delta  for three direction families
          random   gaussian, per-sentence Frobenius-normalised  (isotropic null hypothesis)
          grad     +grad of CE wrt H0, same norm                (worst case at that norm)
          ridge    H_ridge - H0, so eps=1 is exactly the E9 map (the real error)
    * per (family, eps): exact-match rate against y0, chrF(y, y0), BLEU(y, ref), CE

Q3 is answered twice: by the `ridge` curve (where does it land?), and by projecting the map's
error onto the top-k singular subspace of the per-sentence CE gradients, against the k/d chance
level.

Usage
    python -m src.coincidence --n 300 [--oracles ref,nat] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from . import cc

sys.stdout.reconfigure(encoding="utf-8")

EPS_RANDOM = [0.05, 0.1, 0.2, 0.4, 0.8]
EPS_GRAD = [0.05, 0.1, 0.2, 0.4, 0.8]
EPS_RIDGE = [0.25, 0.5, 0.75, 1.0, 1.5]


# ------------------------------------------------------------------ helpers
def load_set(name: str):
    d = cc.CACHE / name
    t = json.loads((d / "texts.json").read_text(encoding="utf-8"))
    return {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in
            ("E_vi", "m_vi", "E_nat", "m_nat", "E_ref", "m_ref", "lab")}, t


def masked_norm(h: torch.Tensor, am: torch.Tensor) -> torch.Tensor:
    """Per-sentence Frobenius norm over *unmasked* positions -> (B,1,1)."""
    return ((h * am.unsqueeze(-1)).pow(2).sum((1, 2)).sqrt().clamp(min=1e-6)
            .view(-1, 1, 1))


def evaluate(model, H, am, refs, y0, lab, batch=16):
    hyps = cc.generate(model, H, am)
    ce = cc.ce_per_sentence(model, H, am, lab=lab, batch=batch)
    exact = float(np.mean([h.strip() == b.strip() for h, b in zip(hyps, y0)]))
    agree = float(cc.sent_chrf(hyps, y0).mean()) if y0 else float("nan")
    return {"exact": exact, "chrf_vs_base": agree, "bleu": float(cc.bleu(hyps, refs)),
            "ce": float(ce.mean()), "degen": float(cc.degen_rate(hyps)),
            "hyps": hyps}


def grad_dirs(model, H, am, refs, batch=8):
    """Per-sentence d CE / d H, flattened to (n, T*d). Also returns the CE values."""
    from transformers.modeling_outputs import BaseModelOutput
    n = H.shape[0]
    G, ces = [], []
    for s in range(0, n, batch):
        b = slice(s, s + batch)
        lab = cc.labels(refs[b])
        h = H[b].detach().clone().requires_grad_(True)
        o = model(encoder_outputs=BaseModelOutput(last_hidden_state=h),
                  attention_mask=am[b], labels=lab)
        lp = torch.log_softmax(o.logits.float(), -1)
        tgt = lab.clone()
        tgt[tgt == -100] = 0
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        m = (lab != -100).float()
        per = -(tok_lp * m).sum(1) / m.sum(1).clamp(min=1)
        per.sum().backward()
        G.append(h.grad.detach().reshape(h.shape[0], -1).cpu().numpy().copy())
        ces += per.detach().cpu().tolist()
    return np.concatenate(G), np.array(ces)


def analyse_oracle(name, E0, m0, H_ridge, am_ridge, refs, model, aligned=False):
    """Q1/Q2/Q3 for one oracle memory."""
    t0 = time.time()
    H0 = E0.float()
    am0 = m0.float()
    norm0 = masked_norm(H0, am0)
    lab = cc.labels(refs)
    out = {"oracle": name}

    base = evaluate(model, H0, am0, refs, y0=[], lab=lab)   # y0 unknown -> absolute metrics only
    y0 = base.pop("hyps")
    out["base"] = {k: v for k, v in base.items() if k != "chrf_vs_base"}
    out["base"]["norm"] = float(norm0.mean())
    print(f"[{name}] base  bleu {out['base']['bleu']:.2f}  ce {out['base']['ce']:.3f}  "
          f"degen {out['base']['degen']:.3f}  |H0| {out['base']['norm']:.1f}  "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- the map's error, measured.  Only position-meaningful when the oracle memory shares the
    # *vi tokenisation* with the map (i.e. `nat`): then H_ridge and H0 are two readings of the same
    # token at the same slot, and a position-wise difference is a real difference. Against `ref` the
    # two memories have different lengths and tokenisations, so any Euclidean distance is arbitrary;
    # there the `ridge` curve below is the only valid evidence.
    D = H_ridge - H0
    if aligned:
        dn = masked_norm(D, am_ridge)
        out["ridge_error"] = {"abs": float(dn.mean()),
                              "rel": float((dn / norm0).mean()),
                              "cos_sent": float((D * H0).sum((1, 2)).div(
                                  (masked_norm(D, am_ridge).squeeze() * norm0.squeeze())
                              ).mean())}

    def curve(family, eps_list, delta_fn):
        rows = []
        for eps in eps_list:
            H = H0 + eps * delta_fn()
            r = evaluate(model, H, am0, refs, y0, lab)
            r.pop("hyps")
            rows.append({"eps": eps, **r})
            print(f"  {family:6s} eps {eps:<5} exact {r['exact']:.3f}  "
                  f"chrF(y,y0) {r['chrf_vs_base']:.2f}  bleu {r['bleu']:.2f}  "
                  f"ce {r['ce']:.3f}", flush=True)
        return rows

    # random: same generator per family so the curves are comparable across oracles
    rng = np.random.default_rng(7)
    R = torch.from_numpy(rng.standard_normal(tuple(H0.shape)).astype(np.float32)).cuda()
    R = R * (norm0 / masked_norm(R, am0))

    out["curves"] = {}
    out["curves"]["random"] = curve("random", EPS_RANDOM, lambda: R)
    if aligned:
        g, _ = grad_dirs(model, H0, am0, refs)
        Gt = torch.from_numpy(g.reshape(tuple(H0.shape)).astype(np.float32)).cuda()
        Gt = Gt * (norm0 / masked_norm(Gt, am0).clamp(min=1e-9))
        out["grad_norm"] = float(np.linalg.norm(g, axis=1).mean())
        out["curves"]["grad"] = curve("grad", EPS_GRAD, lambda: Gt)
    out["curves"]["ridge"] = curve("ridge", EPS_RIDGE, lambda: D)

    if aligned:
        # ---- Q3: energy of the map's error inside the decoder's sensitive subspace
        d = g.shape[1]
        _, sv, vt = np.linalg.svd(g, full_matrices=False)
        Df = D.reshape(D.shape[0], -1).cpu().numpy()
        Dn = np.linalg.norm(Df, axis=1)
        sub = {}
        for k in (1, 8, 64, 256):
            V = vt[:k]
            proj = Df @ V.T
            frac = (proj ** 2).sum(1) / np.maximum(Dn ** 2, 1e-12)
            sub[str(k)] = {"energy_frac": float(frac.mean()), "chance": k / d}
        gs = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        cos2 = ((gs * Df).sum(1) / np.maximum(Dn, 1e-12)) ** 2
        out["sensitive_subspace"] = {"singular_values_top5": sv[:5].tolist(),
                                     "energy_in_topk": sub,
                                     "cos2_grad_ridge_per_sent": float(cos2.mean()),
                                     "cos2_chance": 1.0 / d}
        print(f"[{name}] ridge error |D| {out['ridge_error']['abs']:.1f} "
              f"({out['ridge_error']['rel'] * 100:.0f}% of |H0|); "
              f"energy in top-1/8/64/256 = "
              + " / ".join(f"{sub[str(k)]['energy_frac'] * 100:.1f}%" for k in (1, 8, 64, 256))
              + f"  (chance {1 / d * 100:.2f}%)", flush=True)
    print(f"[{name}] done ({time.time() - t0:.0f}s)", flush=True)
    return out


def main(n: int, oracles: list[str], force: bool):
    out_p = cc.RESULTS / "coincidence.json"
    if out_p.exists() and not force:
        print(f"{out_p.name} exists, skip (use --force)")
        return
    model = cc.load_decoder(dtype=cc.DT_A)
    sets, txt = load_set("test")
    n = min(n, len(txt["vi"]))
    refs = txt["en"][:n]
    W, b = cc.load_map()

    E_vi = torch.from_numpy(np.asarray(sets["E_vi"][:n]).astype(np.float32)).cuda()
    m_vi = torch.from_numpy(np.asarray(sets["m_vi"][:n]).astype(np.float32)).cuda()
    H_ridge = E_vi @ W + b
    am_ridge = m_vi

    result = {"n": n, "oracles": {}}
    for name in oracles:
        key = {"ref": ("E_ref", "m_ref"), "nat": ("E_nat", "m_nat")}[name]
        E0 = torch.from_numpy(np.asarray(sets[key[0]][:n]).astype(np.float32)).cuda()
        m0 = torch.from_numpy(np.asarray(sets[key[1]][:n]).astype(np.float32)).cuda()
        result["oracles"][name] = analyse_oracle(
            name, E0, m0, H_ridge, am_ridge, refs, model, aligned=(name == "nat"))
        torch.cuda.empty_cache()

    out_p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_p}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--oracles", default="ref,nat")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.n, [o for o in a.oracles.split(",") if o], a.force)
