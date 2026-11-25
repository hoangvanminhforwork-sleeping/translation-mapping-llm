"""Analysis — where do the fitted maps sit relative to (a) the oracle's *behaviour* and
(b) the decoder's sensitive subspace?

Phase 1 measured the decoder in isolation. This script connects it to the maps:

  overlap_*   exact-match and chrF of an arm's greedy output against the oracle's greedy output
              (the oracle = decoder reading the English reference). This is "behavioural distance",
              the quantity the mission's mental model actually names, and it is well defined across
              memories of different lengths, unlike any vector distance.
  energy_topK fraction of ||H_arm - mu_ref||^2 inside the top-K singular subspace of dCE/dH at
              H_nat (same vi tokenisation, so positions correspond). chance = K/dim.
  spectrum    singular-value decay of the per-sentence gradient matrix -> how many directions the
              decoder actually reads.

Usage
    python -m src.analyse_basis [--n 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import torch

from . import cc
from .coincidence import grad_dirs

sys.stdout.reconfigure(encoding="utf-8")
ARMS = ("ridge_full", "ridge_sub", "ridge_recolor10", "ridge_scale122", "ce_ridge",
        "ce_ridge_e6", "ce_scratch")


def main(n: int):
    model = cc.load_decoder(dtype=cc.DT_A)
    t = cc.CACHE / "test"
    txt = json.loads((t / "texts.json").read_text(encoding="utf-8"))
    n = min(n, len(txt["vi"]))
    refs = txt["en"][:n]

    def load(k):
        return np.asarray(np.load(t / f"{k}.npy", mmap_mode="r")[:n])

    E_vi, m_vi = load("E_vi").astype(np.float32), load("m_vi").astype(np.float32)
    E_nat, m_nat = load("E_nat").astype(np.float32), load("m_nat").astype(np.float32)
    E_ref, m_ref = load("E_ref").astype(np.float32), load("m_ref").astype(np.float32)
    lab = torch.from_numpy(load("lab").astype(np.int64)).cuda()

    # target frame from the fit split
    fd = cc.CACHE / "fit"
    Ef, mf = (np.asarray(np.load(fd / f"{k}.npy", mmap_mode="r"))
              for k in ("E_ref", "m_ref"))
    acc, acc2, cnt = np.zeros(512), np.zeros(512), 0
    for s in range(0, len(mf), 512):
        h = Ef[s:s + 512].astype(np.float64)
        mk = mf[s:s + 512] > 0
        acc += (h * mk[..., None]).sum((0, 1))
        acc2 += (h * h * mk[..., None]).sum((0, 1))
        cnt += mk.sum()
    mu = (acc / cnt).astype(np.float32)
    sd = np.sqrt(np.maximum(acc2 / cnt - (acc / cnt) ** 2, 1e-12)).astype(np.float32)

    # oracle behaviour (regenerate: not persisted by coincidence.py)
    t0 = time.time()
    H_ref = torch.from_numpy(E_ref).cuda()
    am_ref = torch.from_numpy(m_ref).cuda()
    hyps_ref = cc.generate(model, H_ref, am_ref)
    ce_ref = cc.ce_per_sentence(model, H_ref, am_ref, lab=lab)
    print(f"oracle ref: bleu {cc.bleu(hyps_ref, refs):.2f} ce {ce_ref.mean():.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # decoder sensitivity at H_nat (position-valid)
    H_nat = torch.from_numpy(E_nat).cuda()
    am_nat = torch.from_numpy(m_nat).cuda()
    g, _ = grad_dirs(model, H_nat, am_nat, refs, batch=8)
    print(f"gradients: {g.shape} ({time.time() - t0:.0f}s)", flush=True)
    from sklearn.utils.extmath import randomized_svd
    U, S, Vt = randomized_svd(g, n_components=256, n_iter=4, random_state=0)
    dim = g.shape[1]
    tot = float((S ** 2).sum())
    out = {"n": n, "dim": dim,
           "oracle": {"bleu": round(float(cc.bleu(hyps_ref, refs)), 3),
                      "ce": round(float(ce_ref.mean()), 4)},
           "spectrum": {"top5": [round(float(x), 3) for x in S[:5]],
                        "sv_ratio_1_64": round(float(S[0] / S[63]), 2),
                        "sv_ratio_1_256": round(float(S[0] / S[255]), 2),
                        "energy_top64": round(float((S[:64] ** 2).sum() / tot), 4),
                        "energy_top256": round(float((S ** 2).sum() / tot), 4),
                        "chance_top64": round(64 / dim, 8)},
           "arms": {}}

    for arm in ARMS:
        p = cc.WORK / "maps" / f"{arm}.npz"
        hp = cc.WORK / "hyps" / f"{arm}.json"
        if not p.exists():
            continue
        d = np.load(p)
        H = torch.from_numpy(E_vi @ d["W"] + d["b"]).cuda()
        am = torch.from_numpy(m_vi).cuda()
        if hp.exists():
            hyps = json.loads(hp.read_text(encoding="utf-8"))[:n]
        else:
            hyps = cc.generate(model, H, am)
        ce = cc.ce_per_sentence(model, H, am, lab=lab)
        row = {"bleu": round(float(cc.bleu(hyps, refs)), 3),
               "ce": round(float(ce.mean()), 4),
               "exact_vs_oracle": round(float(np.mean(
                   [a.strip() == b.strip() for a, b in zip(hyps, hyps_ref)])), 4),
               "chrf_vs_oracle": round(float(cc.sent_chrf(hyps, hyps_ref).mean()), 2)}
        Df = (H - torch.from_numpy(mu).cuda()).reshape(H.shape[0], -1).cpu().numpy()
        Dn = np.linalg.norm(Df, axis=1)
        for k in (8, 64, 256):
            frac = ((Df @ Vt[:k].T) ** 2).sum(1) / np.maximum(Dn ** 2, 1e-12)
            row[f"energy_top{k}"] = round(float(frac.mean()), 6)
            row[f"chance_top{k}"] = round(k / dim, 8)
        with torch.no_grad():
            z = (H - torch.from_numpy(mu).cuda()) / torch.from_numpy(sd).cuda()
            zd = (z.pow(2) * am.unsqueeze(-1)).sum(-1).sqrt()[am > 0]
        row["zdist_med"] = round(float(zd.median()), 2)
        out["arms"][arm] = row
        print(f"{arm:16s} bleu {row['bleu']:.2f} ce {row['ce']:.3f} "
              f"exact_vs_oracle {row['exact_vs_oracle']:.3f} "
              f"chrf_vs_oracle {row['chrf_vs_oracle']:.1f} zdist {row['zdist_med']:.1f} "
              f"E@8/64/256 {row['energy_top8'] * 100:.3f}/{row['energy_top64'] * 100:.3f}/"
              f"{row['energy_top256'] * 100:.2f}% ({time.time() - t0:.0f}s)", flush=True)

    (cc.RESULTS / "basis_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote results/basis_analysis.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    main(ap.parse_args().n)
